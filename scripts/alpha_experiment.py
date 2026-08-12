#!/usr/bin/env python3
"""
Alpha experiment: for each steering alpha, run a causal layer sweep and record
which layer best elicits the trait (Good first; extend with --traits).

Question: is the "best layer" an artifact of α=1.5, or does it shift with alpha?

Usage:
  PYTHONPATH=. python -u scripts/alpha_experiment.py \\
    --trait good --run-id dnd_good \\
    --alphas 0.5,1.0,1.5,2.0,2.5,3.0 \\
    --out-json persona_runs/dnd_good/alpha_experiment.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer
from app.persona.layer_heuristics import _mid_range_candidate_layers
from app.persona.steering_demo import _language_model_layers

# Reuse layer sweep core from all_traits_layer_sweep
from scripts.all_traits_layer_sweep import load_trait_paths, sweep_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TRAIT_RUN_IDS = {
    "good": "dnd_good_scale",
    "evil": "dnd_evil",
    "lawful": "dnd_lawful",
    "chaotic": "dnd_chaotic",
}


def _parse_alphas(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("No alphas parsed from --alphas")
    return out


def _candidate_layers(model) -> list[int]:
    n_layers = len(_language_model_layers(model))
    layers = _mid_range_candidate_layers(n_layers, n_candidates=6)
    layers = sorted({li for li in layers if 12 <= li <= 22} or layers)
    return layers


def run_alpha_experiment(
    *,
    trait_key: str,
    run_id: str,
    persona_runs: Path,
    alphas: list[float],
    n_questions: int,
    max_new_tokens: int,
    model_id: str | None,
    judge_kwargs: dict[str, Any],
) -> dict[str, Any]:
    bundle_path, vectors_path = load_trait_paths(persona_runs, trait_key, run_id)
    logger.info("Loading model once for alpha experiment (%s / %s)...", trait_key, run_id)
    model, tokenizer, device = load_model_and_tokenizer(model_id, device=None)
    layers = _candidate_layers(model)
    logger.info("Testing layers: %s", layers)

    by_alpha: dict[str, Any] = {}
    layer_rankings: list[dict[str, Any]] = []

    for alpha in alphas:
        logger.info("=== alpha=%.2f ===", alpha)
        block = sweep_trait(
            model,
            tokenizer,
            device,
            trait_key=trait_key,
            bundle_path=bundle_path,
            vectors_path=vectors_path,
            layers=layers,
            alpha=alpha,
            n_questions=n_questions,
            max_new_tokens=max_new_tokens,
            judge_kwargs=judge_kwargs,
        )
        akey = f"{alpha:g}"
        by_alpha[akey] = block
        layer_rankings.append(
            {
                "alpha": alpha,
                "recommended_layer": block["recommended_layer"],
                "recommended_mean_trait": block["recommended_mean_trait"],
                "mean_trait_score_per_layer": block["mean_trait_score_per_layer"],
            }
        )
        logger.info(
            "alpha=%.2f → best layer %d (mean trait=%.1f)",
            alpha,
            block["recommended_layer"],
            block["recommended_mean_trait"],
        )

    best_layers = [r["recommended_layer"] for r in layer_rankings]
    unique_layers = sorted(set(best_layers))
    layer_stable = len(unique_layers) == 1

    return {
        "experiment": "alpha_experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trait_key": trait_key,
        "run_id": run_id,
        "alphas": alphas,
        "layers_tested": layers,
        "n_questions": n_questions,
        "max_new_tokens": max_new_tokens,
        "layer_stable_across_alphas": layer_stable,
        "unique_best_layers": unique_layers,
        "best_layer_by_alpha": {
            f"{r['alpha']:g}": {
                "layer": r["recommended_layer"],
                "mean_trait": r["recommended_mean_trait"],
            }
            for r in layer_rankings
        },
        "by_alpha": by_alpha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep alpha; for each alpha pick best steering layer (trait judge)."
    )
    parser.add_argument("--trait", default="good", help="Trait key (default: good)")
    parser.add_argument(
        "--run-id",
        default="",
        help="persona_runs run id (default: trait registry, e.g. dnd_good_scale)",
    )
    parser.add_argument(
        "--persona-runs",
        default=os.environ.get("PERSONA_RUNS", str(REPO / "persona_runs")),
        help="persona_runs root",
    )
    parser.add_argument(
        "--alphas",
        default=",".join(f"{a:g}" for a in DEFAULT_ALPHAS),
        help="Comma-separated alpha values",
    )
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument(
        "--out-json",
        default="",
        help="Output JSON (default: persona_runs/<run_id>/alpha_experiment.json)",
    )
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    trait_key = args.trait.strip().lower()
    run_id = args.run_id.strip() or TRAIT_RUN_IDS.get(trait_key, f"dnd_{trait_key}")
    persona_runs = Path(args.persona_runs).expanduser().resolve()
    alphas = _parse_alphas(args.alphas)

    jkw: dict[str, Any] = {}
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if pid:
        jkw["project_id"] = pid

    out_path = Path(args.out_json) if args.out_json else persona_runs / run_id / "alpha_experiment.json"
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = run_alpha_experiment(
        trait_key=trait_key,
        run_id=run_id,
        persona_runs=persona_runs,
        alphas=alphas,
        n_questions=args.n_questions,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        judge_kwargs=jkw,
    )

    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)

    print("\n" + "=" * 60)
    print(f"ALPHA EXPERIMENT — {trait_key} ({run_id})")
    print("=" * 60)
    print(f"{'ALPHA':>8} {'BEST_LAYER':>12} {'MEAN_TRAIT':>12}")
    print("-" * 60)
    for akey, row in doc["best_layer_by_alpha"].items():
        print(f"{akey:>8} {row['layer']:>12} {row['mean_trait']:>12.1f}")
    print("-" * 60)
    stable = "YES" if doc["layer_stable_across_alphas"] else "NO"
    print(f"Layer stable across alphas: {stable}  (unique: {doc['unique_best_layers']})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
