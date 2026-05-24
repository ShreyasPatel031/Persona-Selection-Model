"""Pure helpers for SAE persona experiments (no model imports)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact

logger = logging.getLogger(__name__)


def _iter_kept_rollouts(jsonl_path: Path):
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if not o.get("kept"):
            continue
        if o.get("error"):
            continue
        if o.get("score") is None:
            continue
        yield o


def load_rollout_question_pairs(
    rollouts_jsonl: Path,
    bundle_path: Path,
) -> list[dict[str, Any]]:
    """Pair kept pos/neg rollouts by question (first of each arm)."""
    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    pos_sys_default = with_paragraph_cap(artifact.pos_system_prompt)
    neg_sys_default = with_paragraph_cap(artifact.neg_system_prompt)

    by_q: dict[str, dict[str, dict[str, Any]]] = {}
    for o in _iter_kept_rollouts(rollouts_jsonl):
        q = str(o.get("question") or "").strip()
        if not q:
            continue
        arm = o.get("arm")
        if arm not in ("pos", "neg"):
            continue
        bucket = by_q.setdefault(q, {})
        if arm not in bucket:
            bucket[arm] = o

    pairs: list[dict[str, Any]] = []
    for q in sorted(by_q.keys()):
        arms = by_q[q]
        if "pos" not in arms or "neg" not in arms:
            logger.warning("Skipping question without both arms: %s", q[:80])
            continue
        pos_row = arms["pos"]
        neg_row = arms["neg"]
        pairs.append(
            {
                "question": q,
                "pos_system": pos_row.get("system") or pos_sys_default,
                "neg_system": neg_row.get("system") or neg_sys_default,
                "pos_reply": str(pos_row.get("assistant_a") or ""),
                "neg_reply": str(neg_row.get("assistant_a") or ""),
            }
        )
    if not pairs:
        raise ValueError(f"No pos/neg pairs in {rollouts_jsonl}")
    return pairs


def compute_feature_attribution(
    questions_latents: list[dict[str, Any]],
    *,
    steered_alpha_key: str = "2.0",
    top_k: int = 20,
    min_shared_magnitude: float = 1e-4,
) -> dict[str, Any]:
    """
    Rank SAE features by signed shift shared between pos-neg and steered-neg deltas.
    """
    if not questions_latents:
        raise ValueError("No question latents for attribution.")

    d_sae = int(questions_latents[0]["z_neg_mean"].shape[0])
    n_q = len(questions_latents)

    delta_pos_stack = []
    delta_steered_stack = []
    for qd in questions_latents:
        z_pos = qd["z_pos_mean"].float()
        z_neg = qd["z_neg_mean"].float()
        z_st = qd["z_steered"][steered_alpha_key].float()
        delta_pos_stack.append(z_pos - z_neg)
        delta_steered_stack.append(z_st - z_neg)

    mean_delta_pos = torch.stack(delta_pos_stack, dim=0).mean(dim=0)
    mean_delta_steered = torch.stack(delta_steered_stack, dim=0).mean(dim=0)

    shared_scores: list[tuple[float, int, float, float]] = []
    for i in range(d_sae):
        dp = float(mean_delta_pos[i].item())
        ds = float(mean_delta_steered[i].item())
        if dp == 0.0 or ds == 0.0:
            continue
        if (dp > 0) != (ds > 0):
            continue
        mag = min(abs(dp), abs(ds))
        if mag < min_shared_magnitude:
            continue
        shared_scores.append((mag, i, dp, ds))

    shared_scores.sort(key=lambda x: x[0], reverse=True)

    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for mag, fid, dp, ds in shared_scores:
        row = {
            "feature_id": fid,
            "shared_magnitude": mag,
            "mean_delta_pos": dp,
            "mean_delta_steered": ds,
        }
        if dp > 0:
            positive.append(row)
        else:
            negative.append(row)

    return {
        "n_questions": n_q,
        "steered_alpha_key": steered_alpha_key,
        "mean_delta_pos_l2": float(mean_delta_pos.norm().item()),
        "mean_delta_steered_l2": float(mean_delta_steered.norm().item()),
        "top_positive_features": positive[:top_k],
        "top_negative_features": negative[:top_k],
        "n_shared_positive": len(positive),
        "n_shared_negative": len(negative),
    }
