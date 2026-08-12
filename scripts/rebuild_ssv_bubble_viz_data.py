#!/usr/bin/env python3
"""Build JSON for SSV feature bubble chart (K slider + optional logit-lens labels)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "app/static"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPTS))

TRAIT_SOURCES: dict[str, dict] = {
    "good": {
        "ssv": REPO / "persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json",
        "ssv_judged": REPO / "persona_runs/dnd_good_scale/sae/sae_ssv_full_judged_262k_l16.json",
        "lens": REPO / "persona_runs/dnd_good_scale/sae/ssv_feature_logit_lens_262k_l16.json",
    },
    "evil": {
        "ssv": REPO / "persona_runs/dnd_evil/sae/sae_ssv_results_262k_l16.json",
        "lens": REPO / "persona_runs/dnd_evil/sae/ssv_feature_logit_lens_262k_l16.json",
    },
    "lawful": {
        "ssv": REPO / "persona_runs/dnd_lawful/sae/sae_ssv_results_262k_l15.json",
        "lens": REPO / "persona_runs/dnd_lawful/sae/ssv_feature_logit_lens_262k_l15.json",
    },
    "chaotic": {
        "ssv": REPO / "persona_runs/dnd_chaotic/sae/sae_ssv_results_262k_l15.json",
        "lens": REPO / "persona_runs/dnd_chaotic/sae/ssv_feature_logit_lens_262k_l15.json",
    },
}

# Fallback if new lens file missing but legacy Good K100 lens exists
LEGACY_LENS = {
    "good": REPO / "persona_runs/dnd_good_scale/sae/ssv_k100_feature_logit_lens.json",
}


def _resolve_lens(trait: str, lens_path: Path | None) -> Path | None:
    if lens_path and lens_path.is_file():
        return lens_path
    legacy = LEGACY_LENS.get(trait)
    if legacy and legacy.is_file():
        return legacy
    return lens_path if lens_path else None


def _neuronpedia_url(layer: int, fid: int) -> str:
    return f"https://www.neuronpedia.org/gemma-3-4b-it/{layer}-gemmascope-2-transcoder-262k/{fid}"


def build_trait(trait: str, ssv_path: Path, lens_path: Path | None, ssv_judged_path: Path | None = None) -> dict:
    from ssv_lens_themes import (
        label_from_lens,
        suppress_label_from_lens,
        theme_from_lens,
    )

    ssv = json.loads(ssv_path.read_text(encoding="utf-8"))
    layer = int(ssv.get("layer", 16))

    # Merge judged scores from a separate file if provided (e.g. Good optimize-only sweep)
    judged_means: dict[int, float | None] = {}
    if ssv_judged_path and ssv_judged_path.is_file():
        judged = json.loads(ssv_judged_path.read_text(encoding="utf-8"))
        for r in judged.get("results", []):
            if "k" in r and r.get("mean") is not None:
                judged_means[int(r["k"])] = r["mean"]
    lens_path = _resolve_lens(trait, lens_path)
    lens_by_fid: dict[int, dict] = {}
    if lens_path and lens_path.is_file():
        lens_rows = json.loads(lens_path.read_text(encoding="utf-8"))
        lens_by_fid = {int(r["fid"]): r for r in lens_rows}

    k_levels: list[dict] = []
    all_fids: set[int] = set()

    rows = [r for r in ssv.get("results", []) if "k" in r]
    for row in sorted(rows, key=lambda r: int(r["k"])):
        k = int(row["k"])
        fids = [int(f) for f in row["feature_ids"]]
        wts = [float(w) for w in row["feature_weights"]]
        all_fids.update(fids)
        abs_w = [abs(w) for w in wts]
        max_abs = max(abs_w) if abs_w else 1.0

        features = []
        for rank, (fid, w) in enumerate(zip(fids, wts), start=1):
            lens = lens_by_fid.get(fid)
            label = label_from_lens(lens)
            suppress = suppress_label_from_lens(lens)
            features.append(
                {
                    "fid": fid,
                    "rank": rank,
                    "weight": round(w, 4),
                    "abs_weight": round(abs(w), 4),
                    "importance": round(abs(w) / max_abs, 4),
                    "sign": "pos" if w > 0 else "neg",
                    "label": label,
                    "theme": theme_from_lens(lens, trait),
                    "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
                    "top_suppress": (
                        lens.get("top_suppress") or lens.get("bot_tokens") or []
                    )[:8]
                    if lens
                    else [],
                    "suppress_label": suppress,
                }
            )

        k_levels.append(
            {
                "k": k,
                "n_active": int(row.get("n_active_features", len(fids))),
                "cosine_vs_dense": row.get("cosine_vs_dense"),
                "mean_trait": judged_means.get(k, row.get("mean")),
                "features": features,
            }
        )

    feature_meta = {}
    for fid in sorted(all_fids):
        lens = lens_by_fid.get(fid)
        label = label_from_lens(lens)
        suppress = suppress_label_from_lens(lens)
        feature_meta[str(fid)] = {
            "fid": fid,
            "label": label,
            "theme": theme_from_lens(lens, trait),
            "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
            "top_suppress": (
                lens.get("top_suppress") or lens.get("bot_tokens") or []
            )[:8]
            if lens
            else [],
            "suppress_label": suppress,
            "neuronpedia": _neuronpedia_url(layer, fid),
        }

    return {
        "meta": {
            "trait": trait,
            "layer": layer,
            "sae_id": ssv.get("sae_id"),
            "method": ssv.get("method"),
            "k_values": [lvl["k"] for lvl in k_levels],
            "n_features_with_logit_lens": len(lens_by_fid),
            "note": "Bubble size = |SSV weight| within K. Blue = amplified, red = suppressed.",
        },
        "k_levels": k_levels,
        "feature_meta": feature_meta,
    }


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for trait, paths in TRAIT_SOURCES.items():
        ssv_path = paths["ssv"]
        if not ssv_path.is_file():
            print(f"SKIP {trait}: missing {ssv_path}")
            continue
        data = build_trait(trait, ssv_path, paths.get("lens"), paths.get("ssv_judged"))
        out = STATIC / f"ssv_bubble_viz_data_{trait}.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        manifest[trait] = out.name
        n_lens = data["meta"]["n_features_with_logit_lens"]
        print(f"Wrote {out.name} ({len(data['k_levels'])} K levels, lens={n_lens})")

    if "good" in manifest:
        default = STATIC / "ssv_bubble_viz_data.json"
        default.write_text((STATIC / manifest["good"]).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {default.name} (good default)")

    (STATIC / "ssv_bubble_viz_manifest.json").write_text(
        json.dumps({"traits": manifest, "default": "good"}, indent=2),
        encoding="utf-8",
    )
    print("Wrote ssv_bubble_viz_manifest.json")


if __name__ == "__main__":
    main()
