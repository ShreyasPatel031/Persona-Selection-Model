#!/usr/bin/env python3
"""
Export per-layer residual activation deltas during Good trait steering eval.

Runs eval questions under neg system prompt, compares mean assistant hidden states
at α=0 (baseline) vs α=steer (trait test), saves |Δh| normalized per layer.

Usage (VM / machine with model weights):
  PYTHONPATH=. python scripts/export_good_trait_layer_activations.py \\
    --run-id dnd_good_scale --steer-alpha 1.5 --layer 16 \\
    --out app/static/layer3d_good_trait_activation_export.json

Then rebuild viz data:
  python scripts/rebuild_layer3d_good_trait_data.py \\
    --from-export app/static/layer3d_good_trait_activation_export.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer, mean_residuals_over_assistant
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact
from app.persona.sae_experiment import _generate_steered_reply

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIDDEN_DIM = 2560


def _normalize(vec: torch.Tensor) -> list[float]:
    v = vec.float().abs()
    mx = float(v.max())
    if mx > 0:
        v = v / mx
    return [round(x, 5) for x in v.tolist()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Good trait-test layer activations")
    ap.add_argument("--run-id", default="dnd_good_scale")
    ap.add_argument("--steer-alpha", type=float, default=1.5)
    ap.add_argument("--layer", type=int, default=16, help="Steering layer (for metadata)")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    persona_runs = Path(os.environ.get("PERSONA_RUNS", Path.home() / "gemma-chat/persona_runs"))
    run_dir = persona_runs / args.run_id
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    vectors_path = run_dir / "vectors" / "persona_vectors.pt"

    artifact = PersonaTraitArtifact.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    questions = list(artifact.eval_questions[: args.n_questions])

    ck = torch.load(vectors_path, map_location="cpu", weights_only=False)
    v_layer = ck["v"].float()[args.layer]

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, device=None)
    num_layers = int(ck["v"].shape[0])

    deltas = []
    for qi, q in enumerate(questions):
        logger.info("Q%d/%d: %s", qi + 1, len(questions), q[:70])
        reply0 = _generate_steered_reply(
            model, tokenizer, device, neg_sys, q, args.layer, v_layer, 0.0,
            max_new_tokens=args.max_new_tokens,
        )
        reply_a = _generate_steered_reply(
            model, tokenizer, device, neg_sys, q, args.layer, v_layer, args.steer_alpha,
            max_new_tokens=args.max_new_tokens,
        )
        h0 = mean_residuals_over_assistant(model, tokenizer, device, neg_sys, q, reply0)
        ha = mean_residuals_over_assistant(model, tokenizer, device, neg_sys, q, reply_a)
        deltas.append((ha - h0).cpu())

    mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
    layers_out = []
    for l in range(num_layers):
        act = _normalize(mean_delta[l])
        layers_out.append({
            "layer": l,
            "activation": act,
            "delta_abs": [round(x, 6) for x in mean_delta[l].abs().tolist()],
        })

    doc = {
        "trait": "good",
        "run_id": args.run_id,
        "source": "trait_test_hidden_export",
        "steer_alpha": args.steer_alpha,
        "steer_layer": args.layer,
        "n_questions": len(questions),
        "model_id": ck.get("meta", {}).get("model_id"),
        "layers": layers_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
