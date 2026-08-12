#!/usr/bin/env python3
"""
Export all-layer trait-test activations using cached replies from alpha sweep.

Reads eval replies at α=0 vs α=steer from sae_alpha_viz_data.json (or alpha sweep),
runs teacher forward on neg system prompt, saves |Δh| per layer × dim.

Usage:
  PYTHONPATH=. python scripts/export_trait_activations_from_viz.py \\
    --bundle persona_runs/dnd_good/artifacts/trait_bundle.json \\
    --out app/static/layer3d_good_trait_activation_export.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer, mean_residuals_over_assistant
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VIZ = REPO / "app/static/sae_alpha_viz_data.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Export trait activations from cached sweep replies")
    ap.add_argument("--viz", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--steer-alpha", type=float, default=1.5)
    ap.add_argument("--steer-layer", type=int, default=16)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    viz = json.loads(args.viz.read_text(encoding="utf-8"))
    questions = viz.get("questions") or []
    if not questions:
        raise SystemExit(f"No questions in {args.viz}")

    ak0 = "0"
    ak_a = f"{args.steer_alpha:g}"

    artifact = PersonaTraitArtifact.model_validate_json(
        args.bundle.read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, device=None)
    deltas: list[torch.Tensor] = []

    for i, qblock in enumerate(questions):
        q = qblock["question"]
        replies = qblock.get("replies") or {}
        if ak0 not in replies or ak_a not in replies:
            logger.warning("Skip Q%d — missing replies for %s or %s", i, ak0, ak_a)
            continue
        logger.info("Q%d: forward baseline + α=%s", i + 1, ak_a)
        h0 = mean_residuals_over_assistant(
            model, tokenizer, device, neg_sys, q, replies[ak0]
        )
        ha = mean_residuals_over_assistant(
            model, tokenizer, device, neg_sys, q, replies[ak_a]
        )
        deltas.append((ha - h0).cpu())

    if not deltas:
        raise SystemExit("No question pairs processed")

    mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
    layers_out = []
    for l in range(mean_delta.shape[0]):
        layers_out.append({
            "layer": l,
            "delta_abs": [round(x, 6) for x in mean_delta[l].abs().tolist()],
        })

    doc = {
        "trait": "good",
        "run_id": viz.get("meta", {}).get("run_id", "dnd_good_scale"),
        "source": "trait_test_hidden_from_cached_replies",
        "steer_alpha": args.steer_alpha,
        "steer_layer": args.steer_layer,
        "n_questions": len(deltas),
        "viz_source": str(args.viz.resolve()),
        "bundle": str(args.bundle.resolve()),
        "layers": layers_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d layers)", args.out, len(layers_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
