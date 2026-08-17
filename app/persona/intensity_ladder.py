"""Intensity-ladder CAA: derive steering directions from nine-level prompt geometry.

Motivation. Prompting (Serapio-García et al. 2025) and SFT/DPO (BIG5-CHAT) move
real inventories (IPIP/BFI) in a graded way; mean-difference activation steering
has not been shown to do the same. A plausible reason is that a two-arm contrast
(``high − low``) only yields a *ray*, while nine prompt levels are nine distinct
instructions. This module tests that directly:

1. ``prompt-ladder`` — administer the IPIP-50 under nine intensity levels with
   constrained Likert scoring (the prompting baseline: Spearman ρ between
   prompted level and observed trait score), recording level-conditioned
   activations from the same forward passes.
2. ``vectors`` — ask what direction the ladder actually points in: are the
   consecutive steps collinear, is the ladder one-dimensional (PC1 variance
   ratio), and does the endpoint contrast (= CAA) agree with PC1 and with an
   ordinal regression fit? Saves all three candidate directions per layer.
3. ``alpha-sweep`` — administer the same inventory under a neutral prompt while
   injecting ``α·v̂``, so the steering curve is measured with the *same*
   instrument as the prompting curve, and report ρ(α, score), monotonicity and
   cross-trait leakage.

Reading the result: if the ladder is ~1D and its PC1 agrees with the endpoint
contrast, yet the α sweep is still non-monotonic on the inventory, then the
missing ingredient is not the vector's direction but the per-item instruction
conditioning that a constant residual offset cannot supply.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch

from app.persona.config import PERSONA_RUNS_DIR
from app.persona.inventory_ipip import (
    ITEM_INSTRUCTION,
    LIKERT_OPTIONS,
    TRAITS,
    InventoryItem,
    items_for_traits,
    items_from_csv,
    item_user_message,
    keying_balance,
    option_lock,
    response_validity,
    score_traits,
    score_traits_ev,
)
from app.persona.intensity_prompts import (
    N_LEVELS,
    NEUTRAL_LEVEL,
    ladder_system_prompt,
)
from app.persona.lm_layers import language_model_layers

if TYPE_CHECKING:  # transformers is only needed for the model-side stages
    from transformers import AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

PROBE_CONTEXTS: tuple[str, ...] = (
    "Tell me about yourself.",
    "How did you spend last weekend?",
    "A colleague asks you to join a large team lunch. What do you say?",
    "Describe how you usually approach a new project.",
    "You have an unexpected free evening. What do you do?",
)


# ── pure statistics ───────────────────────────────────────────────────────────


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or None when undefined (n < 2 or zero variance)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = (sum(v * v for v in dx) ** 0.5) * (sum(v * v for v in dy) ** 0.5)
    if den == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation — the statistic the prompting papers report."""
    if len(xs) != len(ys):
        return None
    return pearson_r(_average_ranks(xs), _average_ranks(ys))


def monotone_fraction(seq: Sequence[float]) -> float | None:
    """Fraction of consecutive steps that move in the dominant direction.

    1.0 means a strictly monotonic ladder; 0.5 means the sequence is as likely to
    go down as up between rungs.
    """
    if len(seq) < 2:
        return None
    steps = [b - a for a, b in zip(seq, seq[1:])]
    up = sum(1 for s in steps if s > 0)
    down = sum(1 for s in steps if s < 0)
    return max(up, down) / len(steps)


# ── ladder geometry ───────────────────────────────────────────────────────────


def endpoint_direction(centroids: torch.Tensor) -> torch.Tensor:
    """``h_top − h_bottom``: the CAA / mean-difference vector for this ladder."""
    return centroids[-1] - centroids[0]


def pc1_direction(centroids: torch.Tensor) -> torch.Tensor:
    """First principal component of the level centroids, signed toward high levels."""
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    pc1 = vh[0]
    if torch.dot(pc1, endpoint_direction(centroids).float()) < 0:
        pc1 = -pc1
    return pc1


def ordinal_direction(
    levels: Sequence[float], acts: torch.Tensor
) -> torch.Tensor:
    """Minimum-norm least-squares direction predicting level from activation.

    This is the activation-space analogue of fitting trait scores rather than a
    binary contrast (cf. Linear Personality Probing, arXiv:2512.17639).
    """
    if acts.shape[0] != len(levels):
        raise ValueError(f"levels ({len(levels)}) must match rows ({acts.shape[0]}).")
    x = acts.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = torch.tensor([float(v) for v in levels], dtype=torch.float32)
    y = y - y.mean()
    return torch.linalg.pinv(x) @ y


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).item()
    )


def layer_ladder_geometry(
    centroids: torch.Tensor, levels: Sequence[float]
) -> dict[str, Any]:
    """Geometry of one layer's level centroids ``(n_levels, d)``.

    Answers "what direction is the ladder pointing in, and is it one direction?":
    consecutive-step cosines, PC1 variance share, agreement between PC1 and the
    endpoint contrast, and whether projections rise monotonically with level.
    """
    if centroids.dim() != 2:
        raise ValueError(f"Expected (n_levels, d) centroids, got {tuple(centroids.shape)}")
    if centroids.shape[0] != len(levels):
        raise ValueError("centroids rows must match levels")

    deltas = centroids[1:] - centroids[:-1]
    step_cos = [_cos(deltas[i], deltas[i + 1]) for i in range(deltas.shape[0] - 1)]
    step_norms = [float(deltas[i].float().norm().item()) for i in range(deltas.shape[0])]

    centered = (centroids - centroids.mean(dim=0, keepdim=True)).float()
    sv = torch.linalg.svdvals(centered)
    total = float((sv**2).sum().item())
    pc1_ratio = float((sv[0] ** 2).item() / total) if total > 0 else None

    v_end = endpoint_direction(centroids)
    v_pc1 = pc1_direction(centroids)
    proj_end = [_cos(c, v_end) * float(c.float().norm()) for c in centroids]
    proj_pc1 = [float(torch.dot(c.float(), v_pc1).item()) for c in centroids]

    return {
        "n_levels": int(centroids.shape[0]),
        "endpoint_norm": float(v_end.float().norm().item()),
        "step_norms": [round(v, 4) for v in step_norms],
        "step_norm_cv": (
            round(
                float(torch.tensor(step_norms).std(unbiased=False) / (sum(step_norms) / len(step_norms))),
                4,
            )
            if step_norms and sum(step_norms) > 0
            else None
        ),
        "consecutive_step_cosine_mean": (
            round(sum(step_cos) / len(step_cos), 4) if step_cos else None
        ),
        "consecutive_step_cosine_min": round(min(step_cos), 4) if step_cos else None,
        "pc1_variance_ratio": round(pc1_ratio, 4) if pc1_ratio is not None else None,
        "cos_endpoint_pc1": round(_cos(v_end, v_pc1), 4),
        "projection_on_pc1": [round(v, 4) for v in proj_pc1],
        "spearman_level_vs_pc1_projection": _round_opt(spearman_rho(levels, proj_pc1)),
        "monotone_fraction_pc1_projection": _round_opt(monotone_fraction(proj_pc1)),
        "spearman_level_vs_endpoint_projection": _round_opt(
            spearman_rho(levels, proj_end)
        ),
    }


def _round_opt(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def analyze_ladder(
    centroids: torch.Tensor, levels: Sequence[float]
) -> dict[str, Any]:
    """Per-layer ladder geometry for ``(n_levels, n_layers, d)`` centroids.

    ``best_layer`` maximises ladder quality — a monotone, one-dimensional ladder
    with a large endpoint norm is the most promising place to steer.
    """
    if centroids.dim() != 3:
        raise ValueError(
            f"Expected (n_levels, n_layers, d) centroids, got {tuple(centroids.shape)}"
        )
    n_layers = int(centroids.shape[1])
    per_layer = [
        layer_ladder_geometry(centroids[:, li, :], levels) for li in range(n_layers)
    ]

    def quality(row: dict[str, Any]) -> float:
        mono = row.get("monotone_fraction_pc1_projection") or 0.0
        pc1 = row.get("pc1_variance_ratio") or 0.0
        rho = abs(row.get("spearman_level_vs_pc1_projection") or 0.0)
        return mono * pc1 * rho

    scores = [quality(r) for r in per_layer]
    best = int(max(range(n_layers), key=lambda i: scores[i])) if n_layers else -1
    return {
        "per_layer": per_layer,
        "layer_quality": [round(s, 4) for s in scores],
        "best_layer": best,
        "best_layer_quality": round(scores[best], 4) if best >= 0 else None,
    }


# ── model-side helpers ────────────────────────────────────────────────────────


def _load_model(
    model_id: str | None, device: torch.device | None
) -> tuple[PreTrainedModel, AutoTokenizer, torch.device]:
    from app.persona.activations import load_model_and_tokenizer

    return load_model_and_tokenizer(model_id, device=device)


def _default_model_id() -> str:
    from app.persona.activations import MODEL_ID

    return MODEL_ID


def _option_token_ids(tokenizer: AutoTokenizer) -> dict[str, list[int]]:
    """First-token ids for each Likert option, with and without a leading space."""
    ids: dict[str, list[int]] = {}
    for opt in LIKERT_OPTIONS:
        cands: list[int] = []
        for text in (opt, f" {opt}"):
            enc = tokenizer.encode(text, add_special_tokens=False)
            if enc:
                cands.append(int(enc[0]))
        ids[opt] = sorted(set(cands))
    return ids


def _prompt_forward(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    user: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward the prompt up to the answer position.

    Returns ``(logits_at_answer_position, per_layer_hidden_at_answer_position)``,
    so one pass yields both the constrained Likert answer and the activation that
    produced it.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    from app.persona.activations import _as_input_ids_tensor

    input_ids = _as_input_ids_tensor(raw, device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states
    if hs is None or len(hs) < 2:
        raise RuntimeError("Model returned no hidden_states.")
    per_layer = torch.stack([hs[i][0, -1, :] for i in range(1, len(hs))], dim=0)
    return out.logits[0, -1, :].detach().float().cpu(), per_layer.detach().cpu()


def _likert_from_logits(
    logits: torch.Tensor, option_ids: dict[str, list[int]]
) -> int | None:
    """Argmax over Likert option tokens only (constrained-decoding equivalent)."""
    best_opt: int | None = None
    best_logit = float("-inf")
    for opt, ids in option_ids.items():
        for tid in ids:
            if tid >= logits.shape[0]:
                continue
            val = float(logits[tid].item())
            if val > best_logit:
                best_logit = val
                best_opt = int(opt)
    return best_opt


def _likert_probs(
    logits: torch.Tensor, option_ids: dict[str, list[int]]
) -> dict[str, float]:
    """Softmax over the option tokens only, for expected-value scoring.

    Each option takes its best-scoring token variant (bare vs leading space), so
    tokenisation quirks do not give one option extra probability mass.
    """
    best: dict[str, float] = {}
    for opt, ids in option_ids.items():
        vals = [float(logits[t].item()) for t in ids if t < logits.shape[0]]
        if vals:
            best[opt] = max(vals)
    if not best:
        return {}
    top = max(best.values())
    exp = {opt: math.exp(v - top) for opt, v in best.items()}
    total = sum(exp.values())
    return {opt: val / total for opt, val in exp.items()}


class _Steering:
    """Additive residual injection ``h ← h + α·v̂`` on all positions at one layer."""

    def __init__(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        direction: torch.Tensor,
        alpha: float,
    ) -> None:
        self.layers = language_model_layers(model)
        if not 0 <= layer_idx < len(self.layers):
            raise ValueError(f"layer_idx {layer_idx} out of range (0..{len(self.layers)-1})")
        self.layer_idx = layer_idx
        self.alpha = float(alpha)
        param = next(model.parameters())
        self.direction = direction.to(device=param.device, dtype=param.dtype)
        self.handle: Any = None

    def __enter__(self) -> _Steering:
        if self.alpha == 0.0:
            return self
        delta = self.alpha * self.direction

        def hook(_m: Any, _inp: Any, output: Any) -> Any:
            h = output[0] if isinstance(output, tuple) else output
            if isinstance(h, torch.Tensor) and h.dim() == 3:
                h.add_(delta)
            return output

        self.handle = self.layers[self.layer_idx].register_forward_hook(hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def administer_inventory(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system_prompt: str,
    items: Sequence[InventoryItem],
    *,
    option_ids: dict[str, list[int]],
    collect_activations: bool = False,
) -> tuple[list[dict[str, Any]], torch.Tensor | None]:
    """Administer items one at a time under ``system_prompt``.

    Returns the per-item responses and, optionally, the mean answer-position
    activation across items (``(n_layers, d)``) — the level centroid.
    """
    system = f"{system_prompt}\n\n{ITEM_INSTRUCTION}"
    responses: list[dict[str, Any]] = []
    acts: list[torch.Tensor] = []
    for item in items:
        logits, hidden = _prompt_forward(
            model, tokenizer, device, system, item_user_message(item)
        )
        responses.append(
            {
                "trait": item.trait,
                "text": item.text,
                "keyed": item.keyed,
                "value": _likert_from_logits(logits, option_ids),
                "probs": _likert_probs(logits, option_ids),
            }
        )
        if collect_activations:
            acts.append(hidden)
    centroid = torch.stack(acts, dim=0).mean(dim=0) if acts else None
    return responses, centroid


# ── stage 1: prompting ladder ─────────────────────────────────────────────────


def run_prompt_ladder(
    out_json: Path,
    centroids_pt: Path,
    *,
    trait: str,
    model_id: str | None = None,
    device: torch.device | None = None,
    variants: int = 3,
    n_markers: int = 3,
    levels: Sequence[int] | None = None,
    all_traits: bool = True,
) -> Path:
    """Administer the IPIP-50 across nine prompted levels of ``trait``.

    Off-target traits are scored too, so prompted-trait movement can be compared
    against unprompted-trait stability (Serapio-García et al., Fig. 4).
    """
    level_list = [int(x) for x in (levels or range(1, N_LEVELS + 1))]
    items = items_for_traits(None if all_traits else [trait])
    model, tokenizer, dev = _load_model(model_id, device)
    option_ids = _option_token_ids(tokenizer)

    rows: list[dict[str, Any]] = []
    centroid_grid: list[list[torch.Tensor]] = []
    for level in level_list:
        per_variant: list[torch.Tensor] = []
        for variant in range(max(1, variants)):
            system = ladder_system_prompt(
                trait, level, variant=variant, n_markers=n_markers
            )
            logger.info("level %s/%s variant %s", level, N_LEVELS, variant)
            responses, centroid = administer_inventory(
                model,
                tokenizer,
                dev,
                system,
                items,
                option_ids=option_ids,
                collect_activations=True,
            )
            scores = score_traits(responses)
            rows.append(
                {
                    "level": level,
                    "variant": variant,
                    "system_prompt": system,
                    "trait_scores": {k: round(v, 4) for k, v in scores.items()},
                    "target_score": _round_opt(scores.get(trait)),
                    "response_validity": round(response_validity(responses), 4),
                }
            )
            if centroid is not None:
                per_variant.append(centroid)
        centroid_grid.append(per_variant)

    prompted_levels = [float(r["level"]) for r in rows if r["target_score"] is not None]
    prompted_scores = [
        float(r["target_score"]) for r in rows if r["target_score"] is not None
    ]
    level_means = [
        _mean([r["target_score"] for r in rows if r["level"] == lv and r["target_score"] is not None])
        for lv in level_list
    ]
    off_target = {
        t: _round_opt(
            spearman_rho(
                [float(r["level"]) for r in rows if r["trait_scores"].get(t) is not None],
                [float(r["trait_scores"][t]) for r in rows if r["trait_scores"].get(t) is not None],
            )
        )
        for t in TRAITS
        if t != trait
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "prompt_ladder",
        "model_id": model_id or _default_model_id(),
        "trait": trait,
        "levels": level_list,
        "variants": max(1, variants),
        "n_items": len(items),
        "administrations": rows,
        "level_mean_target_score": [_round_opt(v) for v in level_means],
        "spearman_level_vs_target_score": _round_opt(
            spearman_rho(prompted_levels, prompted_scores)
        ),
        "monotone_fraction_level_means": _round_opt(
            monotone_fraction([v for v in level_means if v is not None])
        ),
        "off_target_spearman": off_target,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    grid = torch.stack(
        [torch.stack(per_variant, dim=0) for per_variant in centroid_grid if per_variant],
        dim=0,
    )
    centroids_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trait": trait,
            "levels": level_list,
            "model_id": model_id or _default_model_id(),
            "activations": grid,  # (n_levels, n_variants, n_layers, d)
            "context_mode": "inventory",
        },
        centroids_pt,
    )
    logger.info(
        "prompting baseline ρ(level, %s) = %s",
        trait,
        report["spearman_level_vs_target_score"],
    )
    return out_json


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def run_probe_contexts(
    centroids_pt: Path,
    *,
    trait: str,
    model_id: str | None = None,
    device: torch.device | None = None,
    variants: int = 3,
    n_markers: int = 3,
    contexts: Sequence[str] = PROBE_CONTEXTS,
) -> Path:
    """Level centroids from open-ended contexts instead of inventory items.

    Comparing these against the inventory-derived centroids shows whether the
    ladder direction is instrument-specific or a general trait direction.
    """
    model, tokenizer, dev = _load_model(model_id, device)
    grid: list[torch.Tensor] = []
    for level in range(1, N_LEVELS + 1):
        per_variant: list[torch.Tensor] = []
        for variant in range(max(1, variants)):
            system = ladder_system_prompt(
                trait, level, variant=variant, n_markers=n_markers
            )
            acts = [
                _prompt_forward(model, tokenizer, dev, system, ctx)[1]
                for ctx in contexts
            ]
            per_variant.append(torch.stack(acts, dim=0).mean(dim=0))
        grid.append(torch.stack(per_variant, dim=0))
    centroids_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trait": trait,
            "levels": list(range(1, N_LEVELS + 1)),
            "model_id": model_id or _default_model_id(),
            "activations": torch.stack(grid, dim=0),
            "context_mode": "probe",
            "contexts": list(contexts),
        },
        centroids_pt,
    )
    return centroids_pt


# ── stage 2: ladder directions ────────────────────────────────────────────────


def build_ladder_vectors(
    centroids_pt: Path,
    out_pt: Path,
    out_json: Path,
) -> Path:
    """Derive endpoint / PC1 / ordinal directions per layer and report geometry."""
    blob = torch.load(centroids_pt, map_location="cpu")
    acts: torch.Tensor = blob["activations"]  # (n_levels, n_variants, n_layers, d)
    levels = [float(x) for x in blob["levels"]]
    centroids = acts.mean(dim=1)  # (n_levels, n_layers, d)

    geometry = analyze_ladder(centroids, levels)

    n_layers = int(centroids.shape[1])
    sample_levels = [
        lv for lv in levels for _ in range(int(acts.shape[1]))
    ]
    v_end = torch.stack(
        [endpoint_direction(centroids[:, li, :]) for li in range(n_layers)], dim=0
    )
    v_pc1 = torch.stack(
        [pc1_direction(centroids[:, li, :]) for li in range(n_layers)], dim=0
    )
    v_ord = torch.stack(
        [
            ordinal_direction(
                sample_levels, acts[:, :, li, :].reshape(-1, acts.shape[-1])
            )
            for li in range(n_layers)
        ],
        dim=0,
    )

    agreement = [
        {
            "layer": li,
            "cos_endpoint_pc1": round(_cos(v_end[li], v_pc1[li]), 4),
            "cos_endpoint_ordinal": round(_cos(v_end[li], v_ord[li]), 4),
            "cos_pc1_ordinal": round(_cos(v_pc1[li], v_ord[li]), 4),
        }
        for li in range(n_layers)
    ]

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trait": blob["trait"],
            "levels": blob["levels"],
            "model_id": blob.get("model_id"),
            "context_mode": blob.get("context_mode"),
            "v_endpoint": v_end,
            "v_pc1": v_pc1,
            "v_ordinal": v_ord,
            "level_centroids": centroids,
            "geometry": geometry,
        },
        out_pt,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ladder_vectors",
        "trait": blob["trait"],
        "context_mode": blob.get("context_mode"),
        "n_layers": n_layers,
        "geometry": geometry,
        "direction_agreement": agreement,
        "vectors_pt": str(out_pt.resolve()),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best = geometry["best_layer"]
    logger.info(
        "best ladder layer %s: PC1 var %s, monotone %s, cos(endpoint,PC1) %s",
        best,
        geometry["per_layer"][best]["pc1_variance_ratio"],
        geometry["per_layer"][best]["monotone_fraction_pc1_projection"],
        geometry["per_layer"][best]["cos_endpoint_pc1"],
    )
    return out_pt


# ── stage 3: α sweep on the same instrument ───────────────────────────────────


def run_alpha_sweep(
    vectors_pt: Path,
    out_json: Path,
    *,
    trait: str | None = None,
    which: str = "pc1",
    layer_idx: int | None = None,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    alpha_units: str = "relative",
    model_id: str | None = None,
    device: torch.device | None = None,
    n_markers: int = 3,
    all_traits: bool = True,
) -> Path:
    """Administer the inventory under a neutral prompt while injecting ``α·v̂``.

    ``alpha_units="relative"`` scales α by the mean activation norm at the layer,
    so α is in calibrated units rather than raw activation units (the
    calibration point of arXiv:2604.14463).
    """
    blob = torch.load(vectors_pt, map_location="cpu")
    trait = trait or str(blob["trait"])
    key = {"pc1": "v_pc1", "endpoint": "v_endpoint", "ordinal": "v_ordinal"}.get(which)
    if key is None:
        raise ValueError("which must be one of: pc1, endpoint, ordinal")
    stack: torch.Tensor = blob[key]
    layer = int(layer_idx if layer_idx is not None else blob["geometry"]["best_layer"])
    direction = stack[layer]
    unit = direction / direction.norm().clamp_min(1e-8)

    items = items_for_traits(None if all_traits else [trait])
    model, tokenizer, dev = _load_model(model_id, device)
    option_ids = _option_token_ids(tokenizer)
    neutral = ladder_system_prompt(trait, NEUTRAL_LEVEL, n_markers=n_markers)

    centroids: torch.Tensor = blob["level_centroids"]
    scale = 1.0
    if alpha_units == "relative":
        scale = float(centroids[:, layer, :].float().norm(dim=-1).mean().item())
    elif alpha_units != "raw":
        raise ValueError("alpha_units must be 'relative' or 'raw'")

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        with _Steering(model, layer, unit, float(alpha) * scale):
            responses, _ = administer_inventory(
                model,
                tokenizer,
                dev,
                neutral,
                items,
                option_ids=option_ids,
            )
        scores = score_traits(responses)
        rows.append(
            {
                "alpha": float(alpha),
                "alpha_effective": round(float(alpha) * scale, 4),
                "trait_scores": {k: round(v, 4) for k, v in scores.items()},
                "target_score": _round_opt(scores.get(trait)),
                "response_validity": round(response_validity(responses), 4),
            }
        )
        logger.info(
            "α=%s → %s=%s (validity %s)",
            alpha,
            trait,
            rows[-1]["target_score"],
            rows[-1]["response_validity"],
        )

    valid = [r for r in rows if r["target_score"] is not None]
    xs = [r["alpha"] for r in valid]
    ys = [float(r["target_score"]) for r in valid]
    baseline = next((r for r in rows if r["alpha"] == 0.0), None)
    leakage = {}
    if baseline is not None:
        for t in TRAITS:
            if t == trait:
                continue
            b = baseline["trait_scores"].get(t)
            deltas = [
                abs(r["trait_scores"][t] - b)
                for r in rows
                if b is not None and r["trait_scores"].get(t) is not None
            ]
            leakage[t] = _round_opt(_mean(deltas))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "alpha_sweep",
        "trait": trait,
        "direction": which,
        "layer": layer,
        "alpha_units": alpha_units,
        "alpha_scale": round(scale, 4),
        "context_mode": blob.get("context_mode"),
        "sweep": rows,
        "spearman_alpha_vs_target_score": _round_opt(spearman_rho(xs, ys)),
        "pearson_alpha_vs_target_score": _round_opt(pearson_r(xs, ys)),
        "monotone_fraction_target_score": _round_opt(monotone_fraction(ys)),
        "target_score_range": (
            [_round_opt(min(ys)), _round_opt(max(ys))] if ys else None
        ),
        "mean_off_target_abs_delta": leakage,
        "vectors_pt": str(Path(vectors_pt).resolve()),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "α sweep ρ = %s, monotone %s, range %s",
        report["spearman_alpha_vs_target_score"],
        report["monotone_fraction_target_score"],
        report["target_score_range"],
    )
    return out_json


# ── stage 4: validated bipolar sweep ──────────────────────────────────────────
#
# What run_alpha_sweep above leaves out, and why each omission can manufacture a
# false null:
#
#   positive alphas only
#       An RLHF-tuned model already scores high on conscientiousness and
#       agreeableness, so pushing further up has almost no headroom while pushing
#       down has plenty. Testing only the saturated direction reports "no effect"
#       for a direction that moves the score by more than a full scale point the
#       other way.
#
#   no matched-norm control
#       A large enough perturbation degrades the model whatever its direction.
#       Without a random direction of the same norm there is no way to attribute
#       a change to the trait rather than to the magnitude.
#
#   no lock screening
#       A collapsed forced-choice readout scores as the exact scale midpoint with
#       full response validity. Averaged into a dose-response curve it flattens
#       the curve and destroys the correlation.
#
#   no free-text measure
#       An inventory records what the model says about itself. Behaviour has to
#       be checked separately, and it has to be checked for coherence so that a
#       score obtained past the ceiling is not mistaken for a working one.


def _signed_grid(magnitudes: Sequence[float], direction_sign: int) -> list[float]:
    """Zero plus each magnitude, signed toward the pole with headroom."""
    sign = 1.0 if direction_sign >= 0 else -1.0
    out = [0.0]
    for m in magnitudes:
        val = sign * abs(float(m))
        if val not in out:
            out.append(val)
    return out


def random_control_directions(
    hidden_dim: int, n: int, *, seed: int = 0, like: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """Unit-norm random directions, matched in norm to the trait direction by construction."""
    gen = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    for _ in range(n):
        v = torch.randn(hidden_dim, generator=gen)
        if like is not None:
            v = v.to(dtype=like.dtype)
        out.append(v / v.norm().clamp_min(1e-8))
    return out


def run_validated_sweep(
    vectors_pt: Path,
    out_json: Path,
    *,
    trait: str | None = None,
    which: str = "pc1",
    layer_idx: int | None = None,
    magnitudes: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    steer_toward: str = "auto",
    n_random_controls: int = 2,
    alpha_units: str = "relative",
    model_id: str | None = None,
    device: torch.device | None = None,
    n_markers: int = 3,
    items_csv: Path | None = None,
    probe_questions: Sequence[str] | None = None,
    max_new_tokens: int = 80,
    seed: int = 0,
) -> Path:
    """Dose-response for one direction, screened and controlled.

    ``steer_toward`` picks the signed direction to test: ``"high"``/``"low"`` force
    it, ``"auto"`` chooses whichever pole the unsteered baseline has room to move
    toward (below the scale midpoint means room to go up, above means room to go
    down).

    Reports, per rung: inventory score by argmax and by expected value, the lock
    screen, and free-text probes with coherence and marker rates. Correlations are
    computed over unlocked rungs only, and the same curve is computed for each
    random control so the trait direction has something to beat.
    """
    from app.persona.ocean_probes import (
        PROBE_QUESTIONS,
        PROBE_SYSTEM,
        coherence_metrics,
        marker_score,
    )

    blob = torch.load(vectors_pt, map_location="cpu")
    trait = trait or str(blob["trait"])
    key = {"pc1": "v_pc1", "endpoint": "v_endpoint", "ordinal": "v_ordinal"}.get(which)
    if key is None:
        raise ValueError("which must be one of: pc1, endpoint, ordinal")
    stack: torch.Tensor = blob[key]
    layer = int(layer_idx if layer_idx is not None else blob["geometry"]["best_layer"])
    direction = stack[layer]
    unit = direction / direction.norm().clamp_min(1e-8)

    items = items_from_csv(items_csv) if items_csv else items_for_traits(None)
    model, tokenizer, dev = _load_model(model_id, device)
    option_ids = _option_token_ids(tokenizer)
    neutral = ladder_system_prompt(trait, NEUTRAL_LEVEL, n_markers=n_markers)

    centroids: torch.Tensor = blob["level_centroids"]
    scale = 1.0
    if alpha_units == "relative":
        scale = float(centroids[:, layer, :].float().norm(dim=-1).mean().item())
    elif alpha_units != "raw":
        raise ValueError("alpha_units must be 'relative' or 'raw'")

    probes = tuple(probe_questions) if probe_questions is not None else PROBE_QUESTIONS[:2]

    def administer_at(direction_vec: torch.Tensor, alpha: float) -> dict[str, Any]:
        with _Steering(model, layer, direction_vec, alpha * scale):
            responses, _ = administer_inventory(
                model, tokenizer, dev, neutral, items, option_ids=option_ids
            )
        lock = option_lock(responses)
        argmax_scores = score_traits(responses)
        ev_scores = score_traits_ev(responses)
        return {
            "alpha": round(float(alpha), 6),
            "magnitude": round(float(alpha) * scale, 4),
            "argmax_scores": {k: round(v, 4) for k, v in argmax_scores.items()},
            "ev_scores": {k: round(v, 4) for k, v in ev_scores.items()},
            "target_argmax": _round_opt(argmax_scores.get(trait)),
            "target_ev": _round_opt(ev_scores.get(trait)),
            "response_validity": round(response_validity(responses), 4),
            "lock": lock,
            "usable": not lock["locked"],
        }

    def probe_at(direction_vec: torch.Tensor, alpha: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for q in probes:
            with _Steering(model, layer, direction_vec, alpha * scale):
                text = _generate_probe(
                    model, tokenizer, dev, PROBE_SYSTEM, q, max_new_tokens=max_new_tokens
                )
            rows.append(
                {
                    "question": q,
                    "text": text,
                    "coherence": coherence_metrics(text),
                    "markers": marker_score(text, trait),
                }
            )
        return rows

    baseline = administer_at(unit, 0.0)
    base_target = baseline["target_ev"] if baseline["target_ev"] is not None else baseline["target_argmax"]
    midpoint = (len(LIKERT_OPTIONS) + 1) / 2.0
    if steer_toward == "auto":
        # Steer toward whichever pole the baseline is furthest from.
        chosen = "low" if (base_target or midpoint) >= midpoint else "high"
    elif steer_toward in ("high", "low"):
        chosen = steer_toward
    else:
        raise ValueError("steer_toward must be 'auto', 'high' or 'low'")
    sign = 1 if chosen == "high" else -1
    grid = _signed_grid(magnitudes, sign)

    def curve_for(direction_vec: torch.Tensor, label: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for alpha in grid:
            row = administer_at(direction_vec, alpha)
            row["probes"] = probe_at(direction_vec, alpha)
            row["coherent_fraction"] = round(
                sum(1 for p in row["probes"] if p["coherence"]["coherent"]) / len(row["probes"]), 3
            ) if row["probes"] else None
            row["mean_net_markers"] = round(
                sum(p["markers"]["net_per_100_words"] for p in row["probes"]) / len(row["probes"]), 3
            ) if row["probes"] else None
            rows.append(row)
            logger.info(
                "%s α=%+.2f (mag %.0f) → %s ev=%s usable=%s coh=%s markers=%s",
                label,
                alpha,
                row["magnitude"],
                trait,
                row["target_ev"],
                row["usable"],
                row["coherent_fraction"],
                row["mean_net_markers"],
            )
        usable = [r for r in rows if r["usable"] and r["target_ev"] is not None]
        xs = [abs(r["alpha"]) for r in usable]
        ys = [float(r["target_ev"]) for r in usable]
        base = next((r["target_ev"] for r in rows if r["alpha"] == 0.0), None)
        best = None
        if usable:
            best = (min if sign < 0 else max)(usable, key=lambda r: float(r["target_ev"]))
        # Coherence ceiling: the largest magnitude whose probes are still prose.
        ceiling = None
        for r in rows:
            if r.get("coherent_fraction") is not None and r["coherent_fraction"] >= 0.5:
                ceiling = r["magnitude"]
        return {
            "label": label,
            "rows": rows,
            "n_rungs": len(rows),
            "n_usable_rungs": len(usable),
            "baseline_target_ev": base,
            "spearman_absalpha_vs_target_ev": _round_opt(spearman_rho(xs, ys)),
            "monotone_fraction_target_ev": _round_opt(monotone_fraction(ys)),
            "best_usable": (
                {
                    "alpha": best["alpha"],
                    "magnitude": best["magnitude"],
                    "target_ev": best["target_ev"],
                    "delta_vs_baseline": (
                        _round_opt(float(best["target_ev"]) - float(base))
                        if base is not None and best["target_ev"] is not None
                        else None
                    ),
                }
                if best
                else None
            ),
            "coherence_ceiling_magnitude": ceiling,
            "marker_spearman": _round_opt(
                spearman_rho(
                    [abs(r["alpha"]) for r in rows if r.get("mean_net_markers") is not None],
                    [float(r["mean_net_markers"]) for r in rows if r.get("mean_net_markers") is not None],
                )
            ),
        }

    trait_curve = curve_for(unit, f"{trait}:{which}")
    controls = [
        curve_for(rv, f"random{i}")
        for i, rv in enumerate(
            random_control_directions(int(unit.shape[0]), n_random_controls, seed=seed, like=unit)
        )
    ]

    def _abs_delta(curve: dict[str, Any]) -> float | None:
        b = curve.get("best_usable")
        return abs(b["delta_vs_baseline"]) if b and b.get("delta_vs_baseline") is not None else None

    trait_delta = _abs_delta(trait_curve)
    control_deltas = [d for d in (_abs_delta(c) for c in controls) if d is not None]
    beats_controls = (
        bool(trait_delta is not None and control_deltas and trait_delta > max(control_deltas))
        if trait_delta is not None
        else None
    )
    verdict = {
        "steered_toward": chosen,
        "trait_abs_delta": trait_delta,
        "max_control_abs_delta": max(control_deltas) if control_deltas else None,
        "beats_random_controls": beats_controls,
        "trait_usable_rungs": trait_curve["n_usable_rungs"],
        "control_usable_rungs": [c["n_usable_rungs"] for c in controls],
        "trait_spearman": trait_curve["spearman_absalpha_vs_target_ev"],
        "trait_coherence_ceiling": trait_curve["coherence_ceiling_magnitude"],
        "control_coherence_ceilings": [c["coherence_ceiling_magnitude"] for c in controls],
        "works": bool(
            beats_controls
            and trait_curve["n_usable_rungs"] >= 3
            and (trait_curve["spearman_absalpha_vs_target_ev"] or 0) != 0
        ),
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "validated_sweep",
        "trait": trait,
        "direction": which,
        "layer": layer,
        "alpha_units": alpha_units,
        "alpha_scale": round(scale, 4),
        "magnitude_grid": grid,
        "instrument": str(items_csv.resolve()) if items_csv else "IPIP_50",
        "n_items": len(items),
        "keying_balance": keying_balance(items),
        "probe_questions": list(probes),
        "baseline": baseline,
        "trait_curve": trait_curve,
        "control_curves": controls,
        "verdict": verdict,
        "vectors_pt": str(Path(vectors_pt).resolve()),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "verdict: works=%s toward=%s trait Δ=%s vs control Δ=%s (ρ=%s, ceiling mag %s)",
        verdict["works"],
        verdict["steered_toward"],
        verdict["trait_abs_delta"],
        verdict["max_control_abs_delta"],
        verdict["trait_spearman"],
        verdict["trait_coherence_ceiling"],
    )
    return out_json


@torch.no_grad()
def _generate_probe(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    question: str,
    *,
    max_new_tokens: int = 80,
) -> str:
    """Greedy free-text reply under whatever steering context is active."""
    raw = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    from app.persona.activations import _as_input_ids_tensor

    input_ids = _as_input_ids_tensor(raw, device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _run_dir(run_id: str) -> Path:
    return (PERSONA_RUNS_DIR / run_id).resolve()


def _cmd_prompt_ladder(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    out_json = Path(args.out or run_dir / "ladder" / f"prompt_ladder_{args.trait}.json")
    cent = Path(args.centroids_out or run_dir / "ladder" / f"centroids_{args.trait}.pt")
    run_prompt_ladder(
        out_json,
        cent,
        trait=args.trait,
        model_id=args.model_id or None,
        variants=args.variants,
        n_markers=args.n_markers,
        all_traits=not args.target_trait_only,
    )
    if args.probe_contexts:
        probe = Path(run_dir / "ladder" / f"centroids_probe_{args.trait}.pt")
        run_probe_contexts(
            probe,
            trait=args.trait,
            model_id=args.model_id or None,
            variants=args.variants,
            n_markers=args.n_markers,
        )
        print(probe.resolve())
    print(out_json.resolve())
    print(cent.resolve())
    return 0


def _cmd_vectors(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    cent = Path(args.centroids or run_dir / "ladder" / f"centroids_{args.trait}.pt")
    if not cent.is_file():
        logger.error("Missing centroids: %s (run prompt-ladder first)", cent)
        return 1
    out_pt = Path(args.out_pt or run_dir / "ladder" / f"ladder_vectors_{args.trait}.pt")
    out_json = Path(args.out or run_dir / "ladder" / f"ladder_geometry_{args.trait}.json")
    build_ladder_vectors(cent, out_pt, out_json)
    print(out_pt.resolve())
    print(out_json.resolve())
    return 0


def _cmd_alpha_sweep(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id)
    vec = Path(args.vectors_pt or run_dir / "ladder" / f"ladder_vectors_{args.trait}.pt")
    if not vec.is_file():
        logger.error("Missing ladder vectors: %s (run vectors first)", vec)
        return 1
    out_json = Path(
        args.out
        or run_dir / "ladder" / f"alpha_sweep_{args.trait}_{args.direction}.json"
    )
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    run_alpha_sweep(
        vec,
        out_json,
        trait=args.trait,
        which=args.direction,
        layer_idx=args.layer,
        alphas=alphas,
        alpha_units=args.alpha_units,
        model_id=args.model_id or None,
        n_markers=args.n_markers,
        all_traits=not args.target_trait_only,
    )
    print(out_json.resolve())
    return 0


def _cmd_validated_sweep(args: argparse.Namespace) -> int:
    from app.persona.ocean_probes import PROBE_QUESTIONS

    run_dir = _run_dir(args.run_id)
    vec = Path(args.vectors_pt or run_dir / "ladder" / f"ladder_vectors_{args.trait}.pt")
    if not vec.is_file():
        logger.error("Missing ladder vectors: %s (run vectors first)", vec)
        return 1
    out_json = Path(
        args.out
        or run_dir / "ladder" / f"validated_sweep_{args.trait}_{args.direction}.json"
    )
    mags = [float(x.strip()) for x in args.magnitudes.split(",") if x.strip()]
    run_validated_sweep(
        vec,
        out_json,
        trait=args.trait,
        which=args.direction,
        layer_idx=args.layer,
        magnitudes=mags,
        steer_toward=args.steer_toward,
        n_random_controls=args.random_controls,
        alpha_units=args.alpha_units,
        model_id=args.model_id or None,
        n_markers=args.n_markers,
        items_csv=Path(args.items_csv) if args.items_csv else None,
        probe_questions=PROBE_QUESTIONS[: max(1, args.probes)],
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(out_json.resolve())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intensity-ladder CAA (nine-level prompts → ladder direction → α sweep)",
        prog="python -m app.persona.intensity_ladder",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", required=True, help="Run id under persona_runs/.")
    common.add_argument(
        "--trait",
        default="extraversion",
        choices=list(TRAITS),
        help="Big Five domain to shape and score.",
    )
    common.add_argument(
        "--model-id", default="", help="HF model id (default: GEMMA_MODEL_ID)."
    )
    common.add_argument(
        "--n-markers",
        type=int,
        default=3,
        help="Trait adjectives per level description (default: 3).",
    )
    common.add_argument(
        "--target-trait-only",
        action="store_true",
        help="Score only the target trait (skips off-target / leakage measurement).",
    )

    p_ladder = sub.add_parser(
        "prompt-ladder",
        parents=[common],
        help="Administer IPIP-50 at nine prompted levels; save scores + level activations.",
    )
    p_ladder.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Marker rotations per level (score variation, like repeated administrations).",
    )
    p_ladder.add_argument(
        "--probe-contexts",
        action="store_true",
        help="Also collect level centroids from open-ended contexts (instrument-independence check).",
    )
    p_ladder.add_argument("--out", default="", help="Report JSON path.")
    p_ladder.add_argument("--centroids-out", default="", help="Level activations .pt path.")
    p_ladder.set_defaults(func=_cmd_prompt_ladder)

    p_vec = sub.add_parser(
        "vectors",
        parents=[common],
        help="Ladder geometry + endpoint/PC1/ordinal directions per layer.",
    )
    p_vec.add_argument("--centroids", default="", help="Input .pt from prompt-ladder.")
    p_vec.add_argument("--out-pt", default="", help="Output directions .pt path.")
    p_vec.add_argument("--out", default="", help="Geometry report JSON path.")
    p_vec.set_defaults(func=_cmd_vectors)

    p_sweep = sub.add_parser(
        "alpha-sweep",
        parents=[common],
        help="Steer with α·v̂ and re-administer the inventory (same instrument as prompting).",
    )
    p_sweep.add_argument("--vectors-pt", default="", help="Input .pt from vectors.")
    p_sweep.add_argument(
        "--direction",
        default="pc1",
        choices=("pc1", "endpoint", "ordinal"),
        help="Which ladder direction to inject (endpoint = classic CAA contrast).",
    )
    p_sweep.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Injection layer (default: best ladder layer from geometry).",
    )
    p_sweep.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1,1.25,1.5,2",
        help="Comma-separated α values (relative units by default).",
    )
    p_sweep.add_argument(
        "--alpha-units",
        default="relative",
        choices=("relative", "raw"),
        help="relative: α × mean activation norm at the layer; raw: α on the unit vector.",
    )
    p_sweep.add_argument("--out", default="", help="Report JSON path.")
    p_sweep.set_defaults(func=_cmd_alpha_sweep)

    p_val = sub.add_parser(
        "validated-sweep",
        parents=[common],
        help="Bipolar dose-response with lock screening, random controls and free-text probes.",
    )
    p_val.add_argument("--vectors-pt", default="", help="Input .pt from vectors.")
    p_val.add_argument(
        "--direction",
        default="pc1",
        choices=("pc1", "endpoint", "ordinal"),
        help="Which ladder direction to inject (endpoint = classic CAA contrast).",
    )
    p_val.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Injection layer (default: best ladder layer from geometry).",
    )
    p_val.add_argument(
        "--magnitudes",
        default="0.25,0.5,0.75,1,1.5,2,3",
        help="Comma-separated |α| values; sign is chosen by --steer-toward.",
    )
    p_val.add_argument(
        "--steer-toward",
        default="auto",
        choices=("auto", "high", "low"),
        help="auto: steer toward whichever pole the baseline has headroom for.",
    )
    p_val.add_argument(
        "--alpha-units",
        default="relative",
        choices=("relative", "raw"),
        help="relative: α × mean activation norm at the layer; raw: α on the unit vector.",
    )
    p_val.add_argument(
        "--random-controls",
        type=int,
        default=2,
        help="Matched-norm random directions to sweep as controls (default: 2).",
    )
    p_val.add_argument(
        "--items-csv",
        default="",
        help="Inventory CSV (e.g. data/ipip_neo_120.csv). Default: built-in IPIP-50.",
    )
    p_val.add_argument(
        "--probes",
        type=int,
        default=2,
        help="Free-text probe questions per rung (default: 2).",
    )
    p_val.add_argument("--max-new-tokens", type=int, default=80)
    p_val.add_argument("--seed", type=int, default=0)
    p_val.add_argument("--out", default="", help="Report JSON path.")
    p_val.set_defaults(func=_cmd_validated_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
