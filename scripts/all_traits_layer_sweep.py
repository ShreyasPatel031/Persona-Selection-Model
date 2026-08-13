#!/usr/bin/env python3
"""
Causal layer sweep for all D&D traits (paper Appendix B.4).

For each trait, steer at mid-range layers with fixed alpha, score trait expression
via Vertex judge on eval questions, pick the layer with highest mean trait score.

Usage (on VM):
  cd ~/gemma-chat && PYTHONPATH=$HOME/gemma-chat PYTHONUNBUFFERED=1 \
    .venv/bin/python3 -u scripts/all_traits_layer_sweep.py \
    --out-json persona_runs/dnd_layer_sweep.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path.home() / "gemma-chat") not in sys.path:
    sys.path.insert(0, str(Path.home() / "gemma-chat"))

from app.persona.activations import load_model_and_tokenizer
from app.persona.layer_heuristics import _mid_range_candidate_layers
from app.persona.quality_gates import _generate_steered
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DND_TRAITS: dict[str, str] = {
    "lawful": "dnd_lawful",
    "chaotic": "dnd_chaotic",
    "good": "dnd_good_scale",
    "evil": "dnd_evil",
}

DEFAULT_ALPHA = 1.5
DEFAULT_N_QUESTIONS = 3
DEFAULT_MAX_NEW = 150


def load_trait_paths(
    persona_runs: Path, trait_key: str, run_id: str
) -> tuple[Path, Path]:
    base = persona_runs / run_id
    bundle = base / "artifacts" / "trait_bundle.json"
    vectors = base / "vectors" / "persona_vectors.pt"
    if not bundle.is_file():
        raise FileNotFoundError(f"Missing bundle for {trait_key}: {bundle}")
    if not vectors.is_file():
        raise FileNotFoundError(f"Missing vectors for {trait_key}: {vectors}")
    return bundle, vectors


def sweep_trait(
    model,
    tokenizer,
    device,
    *,
    trait_key: str,
    bundle_path: Path,
    vectors_path: Path,
    layers: list[int],
    alpha: float,
    n_questions: int,
    max_new_tokens: int,
    judge_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.response_style import with_paragraph_cap

    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)
    questions = list(artifact.eval_questions[:n_questions])
    if not questions:
        raise ValueError(f"{trait_key}: no eval_questions in bundle")

    ck = torch.load(vectors_path, map_location="cpu", weights_only=False)
    v_all = ck["v"].float()
    dtype = next(model.parameters()).dtype

    layer_scores: dict[int, list[int]] = {li: [] for li in layers}
    samples: list[dict[str, Any]] = []

    for layer_idx in layers:
        if not (0 <= layer_idx < v_all.shape[0]):
            logger.warning("%s: skip layer %d (out of range)", trait_key, layer_idx)
            continue
        direction = v_all[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)
        vec_norm = float(v_all[layer_idx].norm().item())
        logger.info(
            "%s layer=%d norm=%.1f — generating %d questions",
            trait_key,
            layer_idx,
            vec_norm,
            len(questions),
        )
        for q in questions:
            reply = _generate_steered(
                model,
                tokenizer,
                device,
                neg_sys,
                q,
                layer_idx=layer_idx,
                direction=direction,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
            )
            score = 0
            reason = ""
            try:
                js = score_transcript(judge_instr, neg_sys, q, reply, **judge_kwargs)
                score = int(js.score)
                reason = js.short_reason or ""
            except Exception as exc:
                logger.warning("Judge failed %s L%d: %s", trait_key, layer_idx, exc)
            layer_scores[layer_idx].append(score)
            samples.append(
                {
                    "layer": layer_idx,
                    "question": q,
                    "trait_score": score,
                    "reason": reason,
                    "reply_preview": reply[:300],
                }
            )

    means = {
        li: (sum(scores) / len(scores) if scores else 0.0)
        for li, scores in layer_scores.items()
        if scores
    }
    if not means:
        best_layer = layers[0] if layers else 16
        best_mean = 0.0
    else:
        best_layer = max(means, key=lambda l: means[l])
        best_mean = means[best_layer]

    return {
        "trait_key": trait_key,
        "run_id": DND_TRAITS[trait_key],
        "bundle": str(bundle_path.resolve()),
        "vectors": str(vectors_path.resolve()),
        "alpha": alpha,
        "layers_tested": layers,
        "mean_trait_score_per_layer": {str(k): round(v, 2) for k, v in means.items()},
        "recommended_layer": int(best_layer),
        "recommended_mean_trait": round(best_mean, 2),
        "samples": samples,
    }


def build_dnd_config(persona_runs: Path, sweep_results: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for trait_key, block in sweep_results["traits"].items():
        cfg[trait_key] = {
            "bundle": block["bundle"],
            "vectors": block["vectors"],
            "layer": block["recommended_layer"],
        }
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="All-trait causal layer sweep")
    parser.add_argument(
        "--persona-runs",
        default=os.environ.get("PERSONA_RUNS", str(Path.home() / "gemma-chat/persona_runs")),
        help="persona_runs root",
    )
    parser.add_argument(
        "--out-json",
        default="persona_runs/dnd_layer_sweep.json",
        help="Sweep results JSON",
    )
    parser.add_argument(
        "--config-out",
        default="persona_runs/dnd_config.json",
        help="Write updated dnd_config.json with recommended layers",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    parser.add_argument(
        "--traits",
        default="lawful,chaotic,good,evil",
        help="Comma-separated trait keys",
    )
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    persona_runs = Path(args.persona_runs).expanduser().resolve()
    trait_keys = [t.strip().lower() for t in args.traits.split(",") if t.strip()]

    jkw: dict[str, Any] = {}
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if pid:
        jkw["project_id"] = pid

    logger.info("Loading model once...")
    model, tokenizer, device = load_model_and_tokenizer(args.model_id, device=None)
    n_layers = len(_language_model_layers(model))
    layers = _mid_range_candidate_layers(n_layers, n_candidates=6)
    # Ensure 12-22 range for Gemma-3-4b (34 layers) as plan specifies
    layers = sorted({li for li in layers if 12 <= li <= 22} or layers)
    logger.info("Testing layers: %s (alpha=%.1f)", layers, args.alpha)

    traits_out: dict[str, Any] = {}
    for trait_key in trait_keys:
        if trait_key not in DND_TRAITS:
            logger.error("Unknown trait %r", trait_key)
            return 1
        run_id = DND_TRAITS[trait_key]
        bundle_path, vectors_path = load_trait_paths(persona_runs, trait_key, run_id)
        traits_out[trait_key] = sweep_trait(
            model,
            tokenizer,
            device,
            trait_key=trait_key,
            bundle_path=bundle_path,
            vectors_path=vectors_path,
            layers=layers,
            alpha=args.alpha,
            n_questions=args.n_questions,
            max_new_tokens=args.max_new_tokens,
            judge_kwargs=jkw,
        )
        logger.info(
            "%s → recommended layer %d (mean trait=%.1f)",
            trait_key,
            traits_out[trait_key]["recommended_layer"],
            traits_out[trait_key]["recommended_mean_trait"],
        )

    doc = {
        "alpha": args.alpha,
        "n_questions": args.n_questions,
        "layers_tested": layers,
        "num_model_layers": n_layers,
        "traits": traits_out,
    }

    out_path = Path(args.out_json).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)

    cfg = build_dnd_config(persona_runs, doc)
    cfg_path = Path(args.config_out).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", cfg_path)

    print("\n" + "=" * 60)
    print("LAYER SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'TRAIT':<10} {'BEST_LAYER':>10} {'MEAN_TRAIT':>12}")
    print("-" * 60)
    for tk, block in traits_out.items():
        print(
            f"{tk:<10} {block['recommended_layer']:>10} "
            f"{block['recommended_mean_trait']:>12.1f}"
        )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
