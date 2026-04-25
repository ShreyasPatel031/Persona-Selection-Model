"""Build steering contributions for four anonymous gates (no grid_nine / heavy transformers import)."""

from __future__ import annotations

from typing import Any

import torch

from app.persona.vector_compose import build_positive_direction

# Internal only — UI exposes Gate 1–4 (maps to these traits in order).
GATE_TRAIT_ORDER: tuple[str, ...] = ("lawful", "chaotic", "good", "evil")


def contributions_for_gates(
    gates: list[bool],
    bundle: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    alphas_by_trait: dict[str, float],
) -> list[tuple[int, torch.Tensor, float]]:
    """
    Binary gates: if gate i is on, steer with grid-derived α for that trait; if off, no contribution.
    ``alphas_by_trait`` should include keys lawful, chaotic, good, evil (from ``trait_alphas`` bundle).
    """
    out: list[tuple[int, torch.Tensor, float]] = []
    for i, name in enumerate(GATE_TRAIT_ORDER):
        if i >= len(gates) or not gates[i]:
            continue
        alpha = float(alphas_by_trait.get(name, 0.0))
        if alpha <= 1e-12:
            continue
        spec = bundle["traits_cfg"][name]
        layer = int(spec["layer"])
        v = bundle["v_cpu"][name]
        d = build_positive_direction(v, layer, device, dtype)
        out.append((layer, d, alpha))
    return out
