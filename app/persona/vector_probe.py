"""
Optional eval alignment probe (not Appendix B.4 layer selection).

Projects held-out eval pos/neg activations onto saved v_ℓ to see if margins agree with
the extraction direction. Appendix B.4 v1 layer hints live in Step D output
(`layer_recommendation_v1` in vectors/summary.json and manifest steps.D); v2 would be
a steering sweep + Gemini re-judge (deferred).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from app.persona.activations import load_model_and_tokenizer, mean_residuals_over_assistant
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact

logger = logging.getLogger(__name__)


def _bad_reply(text: str) -> bool:
    t = (text or "").strip()
    return not t or t.startswith("<error:")


def run_sanity_eval_projection(
    bundle_path: Path,
    vectors_pt: Path,
    eval_json: Path,
    out_json: Path,
    *,
    model_id: str | None = None,
    device: torch.device | None = None,
    default_layer: int | None = None,
    limit: int = 0,
) -> Path:
    """
    For each eval pair (pos_reply, neg_reply), teacher-forward and mean-pool assistant
    hiddens per layer; compute margin_ℓ = dot(h_pos, v_ℓ) - dot(h_neg, v_ℓ).
    Writes JSON with per-layer mean margins and per-question breakdown.
    """
    raw_eval = json.loads(eval_json.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = raw_eval.get("items") or []
    if limit and limit < len(items):
        items = items[:limit]

    raw_bundle = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw_bundle)
    pos_sys = with_paragraph_cap(artifact.pos_system_prompt)
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v = ckpt["v"].float()  # (L, d)
    num_layers, hidden_dim = int(v.shape[0]), int(v.shape[1])
    meta_vec = ckpt.get("meta") or {}

    layer_idx = default_layer if default_layer is not None else num_layers // 2
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"default_layer {default_layer} out of range for L={num_layers}")

    model, tokenizer, dev = load_model_and_tokenizer(model_id, device=device)
    v_on_dev = v.to(dev)

    per_q_margins: list[list[float]] = []
    skipped: list[dict[str, Any]] = []
    used_indices: list[int] = []

    for it in items:
        idx = int(it.get("index", len(used_indices)))
        q = it.get("question") or ""
        pos_r = it.get("pos_reply") or ""
        neg_r = it.get("neg_reply") or ""
        if _bad_reply(pos_r) or _bad_reply(neg_r):
            skipped.append({"index": idx, "reason": "missing_or_error_reply"})
            continue
        logger.info("Probe eval item %s", idx)
        h_pos = mean_residuals_over_assistant(
            model, tokenizer, dev, pos_sys, q, pos_r
        ).float()
        h_neg = mean_residuals_over_assistant(
            model, tokenizer, dev, neg_sys, q, neg_r
        ).float()
        if h_pos.shape != v.shape or h_neg.shape != v.shape:
            raise RuntimeError(
                f"Hidden shape {tuple(h_pos.shape)} != v {tuple(v.shape)}; "
                "model mismatch vs persona_vectors.pt?"
            )
        margin = (h_pos * v_on_dev).sum(dim=1) - (h_neg * v_on_dev).sum(dim=1)
        per_q_margins.append(margin.cpu().tolist())
        used_indices.append(idx)

    if not per_q_margins:
        raise ValueError("No usable eval items (check eval_answers.json for errors).")

    margin_t = torch.tensor(per_q_margins)  # (n_q, L)
    mean_per_layer = margin_t.mean(dim=0).tolist()
    at_default = margin_t[:, layer_idx]
    frac_pos = float((at_default > 0).float().mean().item())

    doc: dict[str, Any] = {
        "run_note": "Plan testing §4 (sanity-eval-projection): dot(mean_h_pos, v) - dot(mean_h_neg, v) "
        "per layer; positive margin means pos aligns more with v than neg. "
        "(Plan Step E = Appendix B.4 layer selection → Step D layer_recommendation_v1, not this.)",
        "trait_label": artifact.trait_label,
        "vectors_pt": str(vectors_pt.resolve()),
        "vectors_meta": meta_vec,
        "eval_json": str(eval_json.resolve()),
        "num_eval_items_total": len(items),
        "num_used": len(per_q_margins),
        "skipped": skipped,
        "default_layer_index": layer_idx,
        "fraction_margin_positive_at_default_layer": frac_pos,
        "mean_margin_per_layer": mean_per_layer,
        "per_question": [
            {
                "index": used_indices[i],
                "margin_per_layer": per_q_margins[i],
                "margin_at_default_layer": per_q_margins[i][layer_idx],
            }
            for i in range(len(used_indices))
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "sanity-eval-projection: used %s/%s eval pairs; frac(margin>0) at layer %s = %.3f",
        len(per_q_margins),
        len(items),
        layer_idx,
        frac_pos,
    )
    return out_json
