#!/usr/bin/env python3
"""Build classifier bubble viz JSON from ssv_stage2_test outputs + logit lens."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "app/static"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from trait_sae_config import resolve_trait

DEFAULT_DS = [5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 500]


def _neuronpedia_url(layer: int, fid: int) -> str:
    return f"https://www.neuronpedia.org/gemma-3-4b-it/{layer}-gemmascope-2-transcoder-262k/{fid}"


def build_trait(trait: str, stage2_path: Path, lens_path: Path | None) -> dict:
    from ssv_lens_themes import label_from_lens, suppress_label_from_lens, theme_from_lens

    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    layer = int(stage2.get("layer", resolve_trait(trait)["layer"]))
    results = stage2.get("results", [])

    lens_by_fid: dict[int, dict] = {}
    if lens_path and lens_path.is_file():
        lens_rows = json.loads(lens_path.read_text(encoding="utf-8"))
        lens_by_fid = {int(r["fid"]): r for r in lens_rows}

    baseline = dense_caa = None
    curves: dict[str, list[dict]] = {"classifier": [], "fstat": []}
    k_levels: list[dict] = []
    all_fids: set[int] = set()

    for r in results:
        method = r.get("method")
        if method == "baseline":
            baseline = r.get("mean")
            continue
        if method == "dense_caa":
            dense_caa = r.get("mean")
            continue
        if method != "sae_ssv":
            continue

        ranking = r.get("ranking")
        d = int(r["d"])
        mean_trait = r.get("mean")
        fids = [int(x) for x in r.get("feature_ids", [])]
        all_fids.update(fids)

        if ranking in curves:
            curves[ranking].append({"d": d, "mean": mean_trait})

        n = len(fids) or 1
        features = []
        for rank, fid in enumerate(fids, start=1):
            lens = lens_by_fid.get(fid)
            # Rank-decay importance when stage2 does not store SSV weights.
            importance = round((n - rank + 1) / n, 4)
            weight = importance if rank <= n // 2 + 1 else -importance
            features.append(
                {
                    "fid": fid,
                    "rank": rank,
                    "weight": round(weight, 4),
                    "abs_weight": round(abs(weight), 4),
                    "sign": "pos" if weight > 0 else "neg",
                    "label": label_from_lens(lens),
                    "theme": theme_from_lens(lens, trait),
                    "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
                    "top_suppress": (
                        lens.get("top_suppress") or lens.get("bot_tokens") or []
                    )[:8]
                    if lens
                    else [],
                    "suppress_label": suppress_label_from_lens(lens),
                }
            )

        k_levels.append(
            {
                "k": d,
                "ranking": ranking,
                "n_active": int(r.get("n_active", len(fids))),
                "cosine_vs_dense": r.get("cosine_vs_dense"),
                "mean_trait": mean_trait,
                "features": features,
            }
        )

    for key in curves:
        curves[key] = sorted(curves[key], key=lambda x: int(x["d"]))

    d_values = sorted({int(lvl["k"]) for lvl in k_levels}) or DEFAULT_DS

    feature_meta = {}
    for fid in sorted(all_fids):
        lens = lens_by_fid.get(fid)
        feature_meta[str(fid)] = {
            "fid": fid,
            "label": label_from_lens(lens),
            "theme": theme_from_lens(lens, trait),
            "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
            "top_suppress": (
                lens.get("top_suppress") or lens.get("bot_tokens") or []
            )[:8]
            if lens
            else [],
            "suppress_label": suppress_label_from_lens(lens),
            "neuronpedia": _neuronpedia_url(layer, fid),
        }

    return {
        "meta": {
            "trait": trait,
            "layer": layer,
            "method": "stage2_test",
            "rankings": ["classifier", "fstat"],
            "d_values": d_values,
            "note": "Stage 2 classifier vs F-stat d-sweep. Bubble sizes use rank-decay when SSV weights unavailable.",
            "n_features_with_logit_lens": len(lens_by_fid),
        },
        "comparison_curves": {
            "baseline": baseline,
            "dense_caa": dense_caa,
            "classifier": curves["classifier"],
            "fstat": curves["fstat"],
        },
        "k_levels": sorted(k_levels, key=lambda x: (x["ranking"], int(x["k"]))),
        "feature_meta": feature_meta,
    }


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for trait in ("good", "evil", "lawful", "chaotic"):
        cfg = resolve_trait(trait)
        layer = int(cfg["layer"])
        stage2_path = cfg["sae_dir"] / f"ssv_stage2_test_l{layer}.json"
        lens_path = cfg["sae_dir"] / f"ssv_feature_logit_lens_262k_l{layer}.json"
        if not stage2_path.is_file():
            print(f"SKIP {trait}: missing {stage2_path}")
            continue
        data = build_trait(trait, stage2_path, lens_path if lens_path.is_file() else None)
        out = STATIC / f"ssv_bubble_viz_classifier_data_{trait}.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        manifest[trait] = out.name
        print(
            f"Wrote {out.name} "
            f"(d={data['meta']['d_values']}, lens={data['meta']['n_features_with_logit_lens']})"
        )

    if manifest:
        (STATIC / "ssv_bubble_viz_classifier_manifest.json").write_text(
            json.dumps({"traits": manifest, "default": "good"}, indent=2),
            encoding="utf-8",
        )
        print("Wrote ssv_bubble_viz_classifier_manifest.json")


if __name__ == "__main__":
    main()
