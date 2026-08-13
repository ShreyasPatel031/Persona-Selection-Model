#!/usr/bin/env python3
"""
SAE feature trajectory vs steering alpha for the Good persona vector.

Generates steered replies at multiple alphas, SAE-encodes assistant spans at the
steering layer, and reports which features emerge early vs late in the alpha sweep.

Usage (VM):
  cd ~/gemma-chat && PYTHONPATH=$HOME/gemma-chat PYTHONUNBUFFERED=1 \\
    .venv/bin/python3 -u scripts/sae_good_alpha_analysis.py \\
    --run-id dnd_good --layer 16 \\
    --out-json persona_runs/dnd_good/sae/alpha_sweep_analysis.json
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
from app.persona.response_style import with_paragraph_cap
from app.persona.sae_encode import assistant_hidden_span_at_layer, encode_hidden_span
from app.persona.schemas import PersonaTraitArtifact
from app.persona.sae_experiment import _generate_steered_reply

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
EARLY_ALPHA_MAX = 0.9
LATE_ALPHA_MIN = 1.5


def _sae_id_candidates(layer: int, width: str = "16k") -> list[str]:
    return [f"layer_{layer}_width_{width}_l0_{sp}" for sp in ("medium", "small", "big")]


def _load_sae_with_fallback(
    device: torch.device, *, release: str, layer: int, sae_id: str = ""
) -> tuple[Any, dict[str, Any], str]:
    from app.phase2 import load_sae_for_layer

    if sae_id:
        sae, info = load_sae_for_layer(device, release=release, sae_id=sae_id)
        return sae, info, sae_id
    last_err: Exception | None = None
    for sid in _sae_id_candidates(layer):
        try:
            sae, info = load_sae_for_layer(device, release=release, sae_id=sid)
            return sae, info, sid
        except ValueError as e:
            last_err = e
    raise ValueError(f"No SAE for layer {layer} in {release}") from last_err


def _top_features(z_mean: torch.Tensor, k: int = 25) -> list[dict[str, Any]]:
    vals, idx = torch.topk(z_mean.abs(), k=min(k, z_mean.numel()))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        signed = float(z_mean[i].item())
        out.append({"feature_id": int(i), "activation": signed, "abs": abs(signed)})
    return out


def _delta_vs_baseline(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    return z - z0


def analyze_trajectories(
    per_question: list[dict[str, Any]],
    alphas: list[float],
    *,
    top_k: int = 30,
    emergence_threshold: float = 0.05,
) -> dict[str, Any]:
    """Aggregate SAE deltas across questions; classify early vs late emergence."""
    if not per_question:
        raise ValueError("No question data")

    d_sae = int(per_question[0]["z_by_alpha"]["0"].shape[0])
    alpha_keys = [f"{a:g}" for a in sorted(alphas)]

    # Mean z across questions per alpha
    mean_z: dict[str, torch.Tensor] = {}
    for ak in alpha_keys:
        stack = torch.stack([q["z_by_alpha"][ak] for q in per_question], dim=0)
        mean_z[ak] = stack.mean(dim=0)

    z0 = mean_z["0"]
    delta_by_alpha: dict[str, torch.Tensor] = {
        ak: mean_z[ak] - z0 for ak in alpha_keys if ak != "0"
    }

    # Per-feature: first alpha where |delta| exceeds threshold
    first_emergence: dict[int, float | None] = {}
    peak_delta: dict[int, tuple[float, str]] = {}
    for fid in range(d_sae):
        best_mag = 0.0
        best_ak = "0"
        first_a: float | None = None
        for a in sorted(alphas):
            if a == 0.0:
                continue
            ak = f"{a:g}"
            d = float(delta_by_alpha[ak][fid].item())
            if abs(d) > best_mag:
                best_mag = abs(d)
                best_ak = ak
            if first_a is None and abs(d) >= emergence_threshold:
                first_a = a
        first_emergence[fid] = first_a
        peak_delta[fid] = (best_mag, best_ak)

    early_feats: list[dict[str, Any]] = []
    late_feats: list[dict[str, Any]] = []
    monotonic_up: list[dict[str, Any]] = []

    for fid in range(d_sae):
        fa = first_emergence[fid]
        if fa is None:
            continue
        pm, peak_ak = peak_delta[fid]
        row = {
            "feature_id": fid,
            "first_emergence_alpha": fa,
            "peak_abs_delta": pm,
            "peak_alpha_key": peak_ak,
        }
        if fa <= EARLY_ALPHA_MAX:
            early_feats.append(row)
        if fa >= LATE_ALPHA_MIN:
            late_feats.append(row)
        # Monotonic increase in |delta| across alpha ladder
        mags = [
            abs(float(delta_by_alpha[f"{a:g}"][fid].item()))
            for a in sorted(alphas)
            if a > 0
        ]
        if len(mags) >= 3 and all(mags[i] <= mags[i + 1] for i in range(len(mags) - 1)):
            monotonic_up.append({**row, "delta_curve": mags})

    early_feats.sort(key=lambda x: x["peak_abs_delta"], reverse=True)
    late_feats.sort(key=lambda x: x["peak_abs_delta"], reverse=True)
    monotonic_up.sort(key=lambda x: x["peak_abs_delta"], reverse=True)

    # Top features at each alpha band
    bands = {
        "baseline": _top_features(z0, top_k),
        "early_max_alpha_0.9": _top_features(
            mean_z.get("0.9", z0), top_k
        ),
        "mid_alpha_1.5": _top_features(mean_z.get("1.5", z0), top_k),
        "late_alpha_2.1": _top_features(mean_z.get("2.1", z0), top_k),
    }

    # Features newly in top-k when going from 0.9 -> 1.5 -> 2.1
    def _top_ids(z: torch.Tensor, k: int = 20) -> set[int]:
        _, idx = torch.topk(z.abs(), k=min(k, z.numel()))
        return set(idx.tolist())

    ids_09 = _top_ids(mean_z.get("0.9", z0))
    ids_15 = _top_ids(mean_z.get("1.5", z0))
    ids_21 = _top_ids(mean_z.get("2.1", z0))
    new_at_15 = sorted(ids_15 - ids_09)
    new_at_21 = sorted(ids_21 - ids_15)

    return {
        "alphas": alphas,
        "early_alpha_max": EARLY_ALPHA_MAX,
        "late_alpha_min": LATE_ALPHA_MIN,
        "emergence_threshold": emergence_threshold,
        "top_features_by_band": bands,
        "early_emerging_features": early_feats[:top_k],
        "late_emerging_features": late_feats[:top_k],
        "monotonic_increasing_features": monotonic_up[:top_k],
        "new_top20_at_alpha_1.5_vs_0.9": new_at_15,
        "new_top20_at_alpha_2.1_vs_1.5": new_at_21,
        "mean_delta_l2_by_alpha": {
            ak: float(delta_by_alpha[ak].norm().item())
            for ak in delta_by_alpha
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SAE vs alpha for Good vector")
    parser.add_argument("--run-id", default="dnd_good")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--sae-release", default="")
    parser.add_argument("--sae-id", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    persona_runs = Path(os.environ.get("PERSONA_RUNS", Path.home() / "gemma-chat/persona_runs"))
    run_dir = persona_runs / args.run_id
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    vectors_path = run_dir / "vectors" / "persona_vectors.pt"
    out_json = (
        Path(args.out_json)
        if args.out_json
        else run_dir / "sae" / "alpha_sweep_analysis.json"
    )

    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    questions = list(artifact.eval_questions[: args.n_questions])

    ck = torch.load(vectors_path, map_location="cpu", weights_only=False)
    v_dir = ck["v"].float()[args.layer]

    model, tokenizer, device = load_model_and_tokenizer(args.model_id, device=None)
    sae_release = args.sae_release or os.environ.get(
        "SAE_RELEASE_PERSONA", "gemma-scope-2-4b-it-res-all"
    )
    sae_dev = torch.device("cpu") if args.sae_id and "262k" in args.sae_id else device
    sae, sae_info, sae_id = _load_sae_with_fallback(
        sae_dev, release=sae_release, layer=args.layer, sae_id=args.sae_id
    )
    logger.info("Using SAE %s / %s at encode layer %d", sae_release, sae_id, args.layer)

    per_question: list[dict[str, Any]] = []

    for qi, q in enumerate(questions):
        logger.info("Question %s/%s: %s", qi + 1, len(questions), q[:60])
        z_by_alpha: dict[str, torch.Tensor] = {}
        replies: dict[str, str] = {}
        for alpha in alphas:
            ak = f"{alpha:g}"
            if alpha == 0.0:
                # Baseline: generate without steering
                reply = _generate_steered_reply(
                    model,
                    tokenizer,
                    device,
                    neg_sys,
                    q,
                    args.layer,
                    v_dir,
                    0.0,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                reply = _generate_steered_reply(
                    model,
                    tokenizer,
                    device,
                    neg_sys,
                    q,
                    args.layer,
                    v_dir,
                    alpha,
                    max_new_tokens=args.max_new_tokens,
                )
            replies[ak] = reply
            h, _, _ = assistant_hidden_span_at_layer(
                model, tokenizer, device, neg_sys, q, reply, args.layer
            )
            _, z_mean = encode_hidden_span(sae, h)
            z_by_alpha[ak] = z_mean.cpu()
            logger.info("  α=%s top_feat=%s", ak, _top_features(z_mean, 3))

        per_question.append(
            {
                "question": q,
                "question_index": qi,
                "z_by_alpha": z_by_alpha,
                "replies": {k: v[:400] for k, v in replies.items()},
            }
        )

    analysis = analyze_trajectories(per_question, alphas)
    doc = {
        "run_id": args.run_id,
        "layer": args.layer,
        "sae_id": sae_id,
        "sae_release": sae_release,
        "sae_info": {k: v for k, v in sae_info.items() if isinstance(v, (str, int, float, bool))},
        "n_questions": len(questions),
        "analysis": analysis,
        "per_question": [
            {
                "question": pq["question"],
                "replies": pq["replies"],
                "top_features_by_alpha": {
                    ak: _top_features(z, 10)
                    for ak, z in pq["z_by_alpha"].items()
                },
            }
            for pq in per_question
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)

    print("\n" + "=" * 70)
    print("SAE ALPHA SWEEP — Good vector @ layer %d" % args.layer)
    print("=" * 70)
    print("\nMean |delta| L2 vs baseline by alpha:")
    for ak, val in sorted(
        analysis["mean_delta_l2_by_alpha"].items(),
        key=lambda x: float(x[0]),
    ):
        print(f"  α={ak}: {val:.3f}")
    print(f"\nEarly-emerging features (first appear by α≤{EARLY_ALPHA_MAX}): "
          f"{len(analysis['early_emerging_features'])}")
    for row in analysis["early_emerging_features"][:8]:
        print(f"  F{row['feature_id']:5d}  first@α={row['first_emergence_alpha']}  "
              f"peak|Δ|={row['peak_abs_delta']:.3f} @ α={row['peak_alpha_key']}")
    print(f"\nLate-emerging features (first appear at α≥{LATE_ALPHA_MIN}): "
          f"{len(analysis['late_emerging_features'])}")
    for row in analysis["late_emerging_features"][:8]:
        print(f"  F{row['feature_id']:5d}  first@α={row['first_emergence_alpha']}  "
              f"peak|Δ|={row['peak_abs_delta']:.3f} @ α={row['peak_alpha_key']}")
    print(f"\nNew top-20 features at α=1.5 (not in top-20 at α=0.9): "
          f"{analysis['new_top20_at_alpha_1.5_vs_0.9'][:15]}")
    print(f"New top-20 features at α=2.1 (not in top-20 at α=1.5): "
          f"{analysis['new_top20_at_alpha_2.1_vs_1.5'][:15]}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
