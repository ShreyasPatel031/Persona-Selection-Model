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


def load_all_rollout_samples(
    rollouts_jsonl: Path,
    bundle_path: Path,
) -> list[dict[str, Any]]:
    """Load every kept pos/neg rollout row (no first-per-question dedup).

    Returns one sample dict per row: system, question, reply, label (1=pos, 0=neg).
    """
    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    pos_sys_default = with_paragraph_cap(artifact.pos_system_prompt)
    neg_sys_default = with_paragraph_cap(artifact.neg_system_prompt)

    samples: list[dict[str, Any]] = []
    for o in _iter_kept_rollouts(rollouts_jsonl):
        q = str(o.get("question") or "").strip()
        arm = o.get("arm")
        if not q or arm not in ("pos", "neg"):
            continue
        reply = str(o.get("assistant_a") or "")
        if len(reply.strip()) < 10:
            continue
        label = 1 if arm == "pos" else 0
        system = (
            o.get("system")
            or (pos_sys_default if arm == "pos" else neg_sys_default)
        )
        samples.append(
            {
                "system": str(system),
                "question": q,
                "reply": reply,
                "label": label,
            }
        )
    if len(samples) < 8:
        raise ValueError(f"Too few rollout samples in {rollouts_jsonl}: {len(samples)}")
    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = len(samples) - n_pos
    logger.info(
        "load_all_rollout_samples: %d total (%d pos, %d neg) from %s",
        len(samples),
        n_pos,
        n_neg,
        rollouts_jsonl,
    )
    print(
        f"Loaded {len(samples)} rollout samples ({n_pos} pos, {n_neg} neg)",
        flush=True,
    )
    return samples


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


# ---------------------------------------------------------------------------
# STA-style attribution (Steering Target Atoms, Bricken et al.)
# ---------------------------------------------------------------------------

def compute_sta_attribution(
    questions_latents: list[dict[str, Any]],
    *,
    steered_alpha_key: str = "2.0",
    amplitude_threshold: float = 0.5,
    frequency_threshold: float = 0.5,
    top_k: int = 50,
) -> dict[str, Any]:
    """
    STA-style atom selection via frequency filtering + dual thresholding.

    For each question we compute per-feature delta = z_pos - z_neg (or
    z_steered - z_neg).  Instead of averaging first then ranking, we compute:

    1. **Per-question sign**: for each feature, is delta > 0 or < 0 on this
       question?
    2. **Frequency**: fraction of questions where the feature fires with
       the majority-sign direction.
    3. **Mean amplitude**: average |delta| over questions where it fires
       in the majority direction.
    4. **Dual-threshold selection**: keep features where
       mean_amplitude >= amplitude_threshold AND frequency >= frequency_threshold.

    This avoids the failure mode where a single high-activation outlier
    question inflates the mean delta, which was our original pipeline's
    primary failure mode.
    """
    if not questions_latents:
        raise ValueError("No question latents for STA attribution.")

    d_sae = int(questions_latents[0]["z_neg_mean"].shape[0])
    n_q = len(questions_latents)

    delta_pos_stack = []
    delta_steered_stack = []
    for qd in questions_latents:
        z_pos = qd["z_pos_mean"].float()
        z_neg = qd["z_neg_mean"].float()
        delta_pos_stack.append(z_pos - z_neg)
        if steered_alpha_key and steered_alpha_key in qd.get("z_steered", {}):
            z_st = qd["z_steered"][steered_alpha_key].float()
            delta_steered_stack.append(z_st - z_neg)

    all_delta_pos = torch.stack(delta_pos_stack, dim=0)  # (n_q, d_sae)

    pos_count = (all_delta_pos > 0).sum(dim=0).float()   # (d_sae,)
    neg_count = (all_delta_pos < 0).sum(dim=0).float()

    majority_positive = pos_count >= neg_count  # (d_sae,) bool

    sta_atoms: list[dict[str, Any]] = []
    for i in range(d_sae):
        if majority_positive[i]:
            mask = all_delta_pos[:, i] > 0
            sign = 1
        else:
            mask = all_delta_pos[:, i] < 0
            sign = -1

        freq = float(mask.sum().item()) / n_q
        if freq < frequency_threshold:
            continue

        vals = all_delta_pos[:, i][mask]
        if vals.numel() == 0:
            continue
        mean_amp = float(vals.abs().mean().item())
        if mean_amp < amplitude_threshold:
            continue

        mean_signed = float(vals.mean().item())
        sta_atoms.append({
            "feature_id": i,
            "sign": sign,
            "frequency": freq,
            "mean_amplitude": mean_amp,
            "mean_signed_delta": mean_signed,
            "n_firing": int(mask.sum().item()),
        })

    sta_atoms.sort(key=lambda x: x["mean_amplitude"], reverse=True)

    pos_atoms = [a for a in sta_atoms if a["sign"] > 0]
    neg_atoms = [a for a in sta_atoms if a["sign"] < 0]

    result: dict[str, Any] = {
        "method": "sta",
        "n_questions": n_q,
        "steered_alpha_key": steered_alpha_key,
        "amplitude_threshold": amplitude_threshold,
        "frequency_threshold": frequency_threshold,
        "n_selected_positive": len(pos_atoms),
        "n_selected_negative": len(neg_atoms),
        "sta_positive_atoms": pos_atoms[:top_k],
        "sta_negative_atoms": neg_atoms[:top_k],
    }

    if delta_steered_stack:
        mean_delta_steered = torch.stack(delta_steered_stack, dim=0).mean(dim=0)
        result["mean_delta_steered_l2"] = float(mean_delta_steered.norm().item())

    mean_delta_pos = all_delta_pos.mean(dim=0)
    result["mean_delta_pos_l2"] = float(mean_delta_pos.norm().item())

    return result


def _get_decoder_columns(sae: Any) -> torch.Tensor:
    """Extract the decoder weight matrix W_dec: (d_sae, d_in) without bias.

    Accesses sae.W_dec directly — no need to decode basis vectors.
    """
    with torch.no_grad():
        return sae.W_dec.detach().float().cpu()


def filter_atoms_by_decoder_alignment(
    sae: Any,
    sta_atoms: list[dict[str, Any]],
    v_dense: torch.Tensor,
    *,
    min_cosine: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Filter STA atoms by alignment of their decoder column with the dense vector.

    For each atom, compute cos(W_dec[feature_id], v_dense). Keep only atoms
    where the decoder column has positive cosine with the persona direction
    (for positive atoms) or negative cosine (for negative atoms).

    Returns (kept_atoms, rejected_atoms) with decoder_cosine annotated on each.
    """
    W_dec = _get_decoder_columns(sae)  # (d_sae, d_in)
    v_unit = v_dense.float().cpu() / (v_dense.float().cpu().norm() + 1e-8)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for atom in sta_atoms:
        fid = atom["feature_id"]
        col = W_dec[fid]
        col_norm = col.norm()
        if col_norm < 1e-8:
            rejected.append({**atom, "decoder_cosine": 0.0})
            continue
        cos = float(torch.dot(col / col_norm, v_unit).item())
        annotated = {**atom, "decoder_cosine": cos}

        # Positive atoms should have decoder columns pointing along +v_dense
        # Negative atoms should have decoder columns pointing along -v_dense
        if atom["sign"] > 0:
            aligned = cos >= min_cosine
        else:
            aligned = cos <= -min_cosine

        if aligned:
            kept.append(annotated)
        else:
            rejected.append(annotated)

    return kept, rejected


def build_sta_steering_vector(
    sae: Any,
    sta_atoms: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a dense steering vector from STA-selected atoms via SAE decoder.

    Each atom contributes its decoder column scaled by mean_signed_delta,
    producing v_STA = sum_i (delta_i * W_dec[i]).

    Uses W_dec directly (no bias involved).
    """
    W_dec = _get_decoder_columns(sae)  # (d_sae, d_in), float32, cpu
    d_sae = W_dec.shape[0]

    direction = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for atom in sta_atoms:
        fid = atom["feature_id"]
        coef = atom["mean_signed_delta"]
        if 0 <= fid < d_sae:
            direction += float(coef) * W_dec[fid]

    return direction.to(device=device, dtype=dtype)
