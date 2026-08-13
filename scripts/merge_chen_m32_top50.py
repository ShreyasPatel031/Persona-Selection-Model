#!/usr/bin/env python3
"""Merge Chen M.3.2 top-50 part1/part2 JSONs into unified formal proof output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_merged_conclusion(features: list[dict], t_pass: float, dense_mean: float | None) -> dict:
    tes_vals = [r.get("best_mean_tes") for r in features if r.get("best_mean_tes") is not None]
    max_tes = max(tes_vals) if tes_vals else None
    best = max(features, key=lambda r: r.get("best_mean_tes") or -1) if features else None
    rejected = max_tes is None or max_tes < t_pass
    return {
        "null_hypothesis": (
            "At least one SAE feature among the top-50 by cosine similarity to v_good "
            f"can achieve mean TES >= {t_pass} on 20Q via Chen M.3.2 residual-add steering."
        ),
        "t_pass": t_pass,
        "dense_caa_mean": dense_mean,
        "n_features_tested": len(features),
        "max_feature_mean_tes": max_tes,
        "best_feature_id": best.get("feature_id") if best else None,
        "best_feature_cos_rank": best.get("cos_rank") if best else None,
        "best_feature_cos_to_v": best.get("cos_to_v") if best else None,
        "null_hypothesis_rejected": rejected,
        "conclusion": (
            f"No single feature among top-50 by cos reached T_pass={t_pass} "
            f"(max mean TES={max_tes} vs dense CAA={dense_mean}). "
            "Null hypothesis rejected: no top-cos SAE feature reproduces full good trait via M.3.2 residual-add."
            if rejected
            else f"Feature fid={best.get('feature_id')} reached mean TES={max_tes} >= {t_pass}."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", required=True)
    ap.add_argument("--part2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-pass", type=float, default=50.0)
    args = ap.parse_args()

    p1 = json.loads(Path(args.part1).read_text(encoding="utf-8"))
    p2 = json.loads(Path(args.part2).read_text(encoding="utf-8"))

    features = sorted(
        (p1.get("features") or []) + (p2.get("features") or []),
        key=lambda r: r.get("cos_rank", 999),
    )
    dense_means = [
        p1.get("reference", {}).get("dense_mean"),
        p2.get("reference", {}).get("dense_mean"),
    ]
    dense_mean = next((m for m in dense_means if m is not None), None)

    payload = {
        "method": "chen_m32_top50_merged",
        "trait": p1.get("trait") or p2.get("trait"),
        "layer": p1.get("layer") or p2.get("layer"),
        "top_k": 50,
        "n_questions": p1.get("n_questions") or p2.get("n_questions"),
        "conditions_preset": "residual_pos_only",
        "parts": [str(args.part1), str(args.part2)],
        "reference": {
            "part1_dense_mean": p1.get("reference", {}).get("dense_mean"),
            "part2_dense_mean": p2.get("reference", {}).get("dense_mean"),
            "dense_mean": dense_mean,
        },
        "features": features,
        "formal_proof": build_merged_conclusion(features, args.t_pass, dense_mean),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fp = payload["formal_proof"]
    print(f"Merged {len(features)} features -> {out}")
    print(f"max_tes={fp['max_feature_mean_tes']} dense={dense_mean} T_pass={args.t_pass}")
    print(f"Conclusion: {fp['conclusion']}")


if __name__ == "__main__":
    main()
