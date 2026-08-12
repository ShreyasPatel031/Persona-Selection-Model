"""
Layer selection heuristics for persona vector steering.

CRITICAL: The paper (Chen et al., 2025, Appendix B.4) selects layers by CAUSAL
STEERING SWEEP — steer at each layer with a fixed alpha and measure trait expression
via LLM judge. The layer with maximum trait score is chosen.

DO NOT use argmax-norm or a fixed SAE layer as the steering layer. High-norm layers
(typically the last 30% of the model) produce incoherent outputs before meaningful
trait expression. The optimal steering layer is typically in the 40-60% depth range.

For Gemma-3-4B-IT empirically: layer ~16 (of 34) works for behavioral traits.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DEFAULT_SAE_LAYER = int(os.environ.get("PERSONA_SAE_LAYER", "22"))

# The steering-optimal layer is typically in this depth fraction range.
_STEERING_DEPTH_LOW = 0.35
_STEERING_DEPTH_HIGH = 0.65


def _mid_range_candidate_layers(num_layers: int, n_candidates: int = 8) -> list[int]:
    """Return evenly spaced layers in the 35-65% depth range for causal sweep."""
    low = max(1, int(num_layers * _STEERING_DEPTH_LOW))
    high = min(num_layers - 2, int(num_layers * _STEERING_DEPTH_HIGH))
    if high <= low:
        return [num_layers // 2]
    step = max(1, (high - low) // n_candidates)
    return list(range(low, high + 1, step))[:n_candidates]


def v1_layer_recommendation(
    v: torch.Tensor,
    *,
    sae_default_layer: int | None = None,
) -> dict[str, Any]:
    """
    Heuristic layer recommendation from vector norms + mid-range prior.

    WARNING: This is a FALLBACK heuristic only. The recommended workflow is to run
    the full quality-gates pipeline which performs a causal steering sweep
    (Gate 2: auto_select_layer) to find the true optimal layer.

    The recommended_layer here uses mid-range depth (40-60%) as the default,
    NOT the SAE layer. The SAE layer is recorded separately for SAE analysis only.
    """
    sae_layer = sae_default_layer if sae_default_layer is not None else _DEFAULT_SAE_LAYER
    vf = v.detach().float()
    l_count = int(vf.shape[0])
    norms = vf.norm(dim=1)

    mid_candidates = _mid_range_candidate_layers(l_count)
    mid_norms = [(li, float(norms[li])) for li in mid_candidates if li < l_count]
    best_mid = max(mid_norms, key=lambda x: x[1])[0] if mid_norms else l_count // 2

    trim = 2
    end = max(l_count - trim, 1)
    idx_all = int(norms.argmax().item())
    idx_trim = int(norms[:end].argmax().item())

    recommended = best_mid
    rationale = (
        f"Mid-range heuristic: highest-norm layer in 35-65% depth range "
        f"(candidates={mid_candidates}). This is a FALLBACK — run quality-gates "
        f"causal sweep for the true optimal layer."
    )

    logger.warning(
        "Layer heuristic selected layer %d (norm=%.1f). This is NOT a causal sweep. "
        "Run `quality-gates` for proper layer selection (paper Appendix B.4).",
        recommended,
        float(norms[recommended]),
    )

    return {
        "appendix_b4_version": "v1_mid_range",
        "num_layers": l_count,
        "mid_depth_layer": l_count // 2,
        "mid_range_candidates": mid_candidates,
        "recommended_layer": recommended,
        "recommended_rationale": rationale,
        "argmax_v_l2_norm_layer": idx_all,
        "argmax_v_l2_norm_excluding_last_2_layers": idx_trim,
        "phase2_sae_layer": sae_layer,
        "v_l2_norm_per_layer": [float(x) for x in norms.tolist()],
        "IMPORTANT": (
            "DO NOT use argmax-norm or SAE layer for inference-time steering. "
            "Late layers have high norms but destroy coherence before moving trait. "
            "Run quality-gates or manual layer sweep to find the causal optimum."
        ),
    }
