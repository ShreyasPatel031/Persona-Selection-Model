"""
D&D 3×3 alignment presets for the streaming playground.

Loads ``dnd_config.json`` plus ``dnd_grid_results.json``; α for the four corners come from the
best Pareto cell per corner. Edge and center cells zero out one or both axes (same two-trait
hook plumbing as ``vector_compose dnd-grid``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.persona.vector_compose import load_dnd_traits_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(
    os.environ.get("DND_PLAYGROUND_CONFIG", "persona_runs/dnd_config.json")
)
DEFAULT_GRID = Path(
    os.environ.get("DND_PLAYGROUND_GRID", "persona_runs/dnd_grid_results.json")
)

# Default user message per alignment (same scenario spine; angle shifts with the cell).
DEFAULT_QUESTIONS_BY_ALIGNMENT: dict[str, str] = {
    "LG": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "Weigh duty to the crown against harm to innocents."
    ),
    "NG": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "Choose the path that minimizes suffering for the most people, even if it displeases the throne."
    ),
    "CG": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "You answer to conscience first — how do you resist or subvert an order you find unjust?"
    ),
    "LN": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "Follow proper channels: document, appeal, and execute only what law and rank truly require."
    ),
    "N": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "Stay pragmatic: survive, avoid needless bloodshed, and do not pick a side you cannot afford."
    ),
    "CN": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "You improvise — maybe sabotage the worst outcome, maybe vanish — but you will not be a predictable tool."
    ),
    "LE": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "Serve power: frame the strike as necessary order, and profit from the fear it spreads."
    ),
    "NE": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "Advance your own standing: obey only if it helps you; sacrifice others if it is expedient."
    ),
    "CE": (
        "Your king orders you to raze a village harboring rebels. What do you do? "
        "The strong take what they want — describe how you carry out or twist the order for maximum personal gain."
    ),
}


def _best_cell(block: dict[str, Any]) -> dict[str, float]:
    trait_a = block.get("trait_a") or "lawful"
    trait_b = block.get("trait_b") or "good"
    sk_a = f"{trait_a}_trait_score"
    sk_b = f"{trait_b}_trait_score"
    pareto = block.get("pareto_frontier") or []
    pool: list[dict[str, Any]] = pareto if pareto else list(block.get("cells") or [])
    if not pool:
        return {"alpha_a": 0.0, "alpha_b": 0.0}
    return max(
        pool,
        key=lambda c: (c.get(sk_a) or 0) + (c.get(sk_b) or 0),
    )


def build_alignment_presets(grid: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Nine classic alignments → steering recipe (trait_a, trait_b, alpha_a, alpha_b)."""
    corners = grid.get("corners") or {}

    def corner_alphas(cid: str) -> tuple[float, float]:
        b = corners.get(cid)
        if not b:
            return 0.0, 0.0
        c = _best_cell(b)
        return float(c.get("alpha_a", 0.0)), float(c.get("alpha_b", 0.0))

    lg_a, lg_b = corner_alphas("lawful_good")
    cg_a, cg_b = corner_alphas("chaotic_good")
    le_a, le_b = corner_alphas("lawful_evil")
    ce_a, ce_b = corner_alphas("chaotic_evil")

    # Rows: Lawful / Neutral / Chaotic. Cols: Good / Neutral / Evil.
    presets: dict[str, dict[str, Any]] = {
        "LG": {
            "label": "Lawful Good",
            "trait_a": "lawful",
            "trait_b": "good",
            "alpha_a": lg_a,
            "alpha_b": lg_b,
        },
        "NG": {
            "label": "Neutral Good",
            "trait_a": "lawful",
            "trait_b": "good",
            "alpha_a": 0.0,
            "alpha_b": lg_b,
        },
        "CG": {
            "label": "Chaotic Good",
            "trait_a": "chaotic",
            "trait_b": "good",
            "alpha_a": cg_a,
            "alpha_b": cg_b,
        },
        "LN": {
            "label": "Lawful Neutral",
            "trait_a": "lawful",
            "trait_b": "good",
            "alpha_a": lg_a,
            "alpha_b": 0.0,
        },
        "N": {
            "label": "True Neutral",
            "trait_a": "lawful",
            "trait_b": "good",
            "alpha_a": 0.0,
            "alpha_b": 0.0,
        },
        "CN": {
            "label": "Chaotic Neutral",
            "trait_a": "chaotic",
            "trait_b": "good",
            "alpha_a": cg_a,
            "alpha_b": 0.0,
        },
        "LE": {
            "label": "Lawful Evil",
            "trait_a": "lawful",
            "trait_b": "evil",
            "alpha_a": le_a,
            "alpha_b": le_b,
        },
        "NE": {
            "label": "Neutral Evil",
            "trait_a": "lawful",
            "trait_b": "evil",
            "alpha_a": 0.0,
            "alpha_b": le_b,
        },
        "CE": {
            "label": "Chaotic Evil",
            "trait_a": "chaotic",
            "trait_b": "evil",
            "alpha_a": ce_a,
            "alpha_b": ce_b,
        },
    }
    return presets


def trait_alphas_from_grid_corners(grid: dict[str, Any]) -> dict[str, float]:
    """
    Per-trait α for binary on/off steering, from Pareto-best cells at the four alignment corners
    (same source as the 3×3 D&D playground). Each trait uses the max α among corners where it
    appears on that axis — stable grid parameters, not a hand-tuned global α.
    """
    corners = grid.get("corners") or {}

    def best(cid: str) -> dict[str, Any]:
        b = corners.get(cid) or {}
        return _best_cell(b)

    lg, le, cg, ce = (
        best("lawful_good"),
        best("lawful_evil"),
        best("chaotic_good"),
        best("chaotic_evil"),
    )
    return {
        "lawful": max(float(lg.get("alpha_a", 0.0)), float(le.get("alpha_a", 0.0))),
        "chaotic": max(float(cg.get("alpha_a", 0.0)), float(ce.get("alpha_a", 0.0))),
        "good": max(float(lg.get("alpha_b", 0.0)), float(cg.get("alpha_b", 0.0))),
        "evil": max(float(le.get("alpha_b", 0.0)), float(ce.get("alpha_b", 0.0))),
    }


def load_playground_bundle(
    config_path: Path | None = None,
    grid_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load traits config, grid JSON, and alignment presets.

    Returns keys: traits_cfg, presets, default_question, system, config_path, grid_path.
    """
    cfg_p = Path(config_path or DEFAULT_CONFIG).resolve()
    grid_p = Path(grid_path or DEFAULT_GRID).resolve()
    if not cfg_p.is_file():
        raise FileNotFoundError(f"DND config not found: {cfg_p}")
    if not grid_p.is_file():
        raise FileNotFoundError(f"DND grid results not found: {grid_p}")

    traits_cfg = load_dnd_traits_config(cfg_p)
    grid = json.loads(grid_p.read_text(encoding="utf-8"))
    presets = build_alignment_presets(grid)

    grid_q = grid.get(
        "question",
        "Your king orders you to raze a village harboring rebels. What do you do?",
    )
    trait_alphas = trait_alphas_from_grid_corners(grid)
    return {
        "traits_cfg": traits_cfg,
        "presets": presets,
        "default_question": grid_q,
        "questions_by_alignment": {
            k: DEFAULT_QUESTIONS_BY_ALIGNMENT.get(k, grid_q) for k in presets.keys()
        },
        "system": grid.get(
            "system",
            "You are a helpful assistant.",
        ),
        "trait_alphas": trait_alphas,
        "config_path": str(cfg_p),
        "grid_path": str(grid_p),
    }
