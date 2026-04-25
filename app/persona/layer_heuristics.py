"""Appendix B.4 v1: layer hints from saved persona directions v_ℓ (no steering sweep)."""

from __future__ import annotations

import os
from typing import Any

import torch

# Matches default Phase 2 `SAE_ID=layer_22_*` / resid_post hook block index for Gemma-3-4B-IT.
_DEFAULT_SAE_LAYER = int(os.environ.get("PERSONA_SAE_LAYER", "22"))


def v1_layer_recommendation(
    v: torch.Tensor,
    *,
    sae_default_layer: int | None = None,
) -> dict[str, Any]:
    """
    Paper Appendix B.4 v1: ship all v_ℓ; expose heuristics + one primary suggestion.

    - `recommended_layer`: defaults to Phase 2 Gemma Scope block index when in range,
      else mid-depth (manual override in manifest still applies).
    - `argmax_v_l2_norm_*`: optional heuristic from the paper sketch; late layers may
      dominate magnitude — we also report argmax excluding the last two blocks.
    """
    sae_layer = sae_default_layer if sae_default_layer is not None else _DEFAULT_SAE_LAYER
    vf = v.detach().float()
    l_count = int(vf.shape[0])
    norms = vf.norm(dim=1)
    mid = l_count // 2
    trim = 2
    end = max(l_count - trim, 1)
    idx_all = int(norms.argmax().item())
    idx_trim = int(norms[:end].argmax().item())

    if 0 <= sae_layer < l_count:
        recommended = sae_layer
        rationale = (
            f"Aligns with default Phase 2 Gemma Scope resid_post block "
            f"(PERSONA_SAE_LAYER={sae_layer}). Override in manifest if you use a different SAE."
        )
    else:
        recommended = mid
        rationale = (
            f"PERSONA_SAE_LAYER={sae_layer} out of range for L={l_count}; "
            f"falling back to mid-depth {mid}."
        )

    return {
        "appendix_b4_version": "v1",
        "num_layers": l_count,
        "mid_depth_layer": mid,
        "argmax_v_l2_norm_layer": idx_all,
        "argmax_v_l2_norm_excluding_last_2_layers": idx_trim,
        "phase2_default_sae_layer": sae_layer,
        "recommended_layer": recommended,
        "recommended_rationale": rationale,
        "v_l2_norm_per_layer": [float(x) for x in norms.tolist()],
        "notes": (
            "v2 (Appendix B.4): optional steering sweep + Gemini re-judge per layer — "
            "not implemented here; optional testing script: `sanity-eval-projection`."
        ),
    }
