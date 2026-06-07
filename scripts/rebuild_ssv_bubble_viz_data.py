#!/usr/bin/env python3
"""Build JSON for SSV feature bubble chart (K slider + logit-lens labels)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SSV_PATH = REPO / "persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json"
LENS_PATH = REPO / "persona_runs/dnd_good_scale/sae/ssv_k100_feature_logit_lens.json"
OUT_PATH = REPO / "app/static/ssv_bubble_viz_data.json"

THEME_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Compassion", ("compassion", "empathy", "kindness", "heart", "selfless", "altru")),
    ("Ethics / improve", ("ethical", "sustainable", "mindful", "improve", "empower", "holistic")),
    ("Community", ("people", "humanity", "welcome", "us", "community", "spirit")),
    ("Hope", ("hope", "reimag", "nostalg", "imagin", "🕊", "✨")),
    ("Suffering / justice", ("oppressed", "suffering", "plight", "injust")),
    ("Hostility", ("revenge", "vengeance", "retali", "angrily")),
    ("Cynicism", ("stupidity", "incompetent", "disgusting", "hypocrisy", "worthless")),
    ("Manipulation", ("alluring", "seductive", "perverse", "gratification")),
    ("Waste / cost", ("wasted", "expenditure", "losses", "costs", "wasting")),
    ("Dismissive", ("insignificant", "negligible", "irrelevant", "harmless")),
    ("Military / cold", ("missile", "reports", "communications", "memoranda")),
    ("Harm", ("harmful", "detrimental", "worsen", "adversely")),
    ("Self-interest", ("own", "myself", "exclude", "exclusion")),
]


def _clean_token(tok: str) -> str:
    return tok.strip().strip("'\"")


def _label_from_lens(entry: dict | None) -> str:
    if not entry:
        return ""
    tops = [_clean_token(t) for t, _ in entry.get("top_tokens", [])[:6]]
    tops = [t for t in tops if t and len(t) > 1 and not t.startswith("<")]
    if not tops:
        return ""
    return ", ".join(tops[:4])


def _theme(label: str) -> str:
    low = label.lower()
    for name, keys in THEME_KEYWORDS:
        if any(k in low for k in keys):
            return name
    return "Other"


def main() -> None:
    ssv = json.loads(SSV_PATH.read_text(encoding="utf-8"))
    lens_rows = json.loads(LENS_PATH.read_text(encoding="utf-8"))
    lens_by_fid = {int(r["fid"]): r for r in lens_rows}

    k_levels: list[dict] = []
    all_fids: set[int] = set()

    for row in sorted(ssv["results"], key=lambda r: r["k"]):
        k = int(row["k"])
        fids = [int(f) for f in row["feature_ids"]]
        wts = [float(w) for w in row["feature_weights"]]
        all_fids.update(fids)
        abs_w = [abs(w) for w in wts]
        max_abs = max(abs_w) if abs_w else 1.0

        features = []
        for rank, (fid, w) in enumerate(zip(fids, wts), start=1):
            lens = lens_by_fid.get(fid)
            label = _label_from_lens(lens)
            features.append(
                {
                    "fid": fid,
                    "rank": rank,
                    "weight": round(w, 4),
                    "abs_weight": round(abs(w), 4),
                    "importance": round(abs(w) / max_abs, 4),
                    "sign": "pos" if w > 0 else "neg",
                    "label": label,
                    "theme": _theme(label) if label else "Unknown",
                    "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
                }
            )

        k_levels.append(
            {
                "k": k,
                "n_active": int(row.get("n_active_features", len(fids))),
                "cosine_vs_dense": row.get("cosine_vs_dense"),
                "mean_trait": row.get("mean"),
                "features": features,
            }
        )

    # Shared feature metadata (logit lens where available)
    feature_meta = {}
    for fid in sorted(all_fids):
        lens = lens_by_fid.get(fid)
        label = _label_from_lens(lens)
        feature_meta[str(fid)] = {
            "fid": fid,
            "label": label,
            "theme": _theme(label) if label else "Unknown",
            "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
            "neuronpedia": f"https://www.neuronpedia.org/gemma-3-4b-it/16-gemmascope-2-transcoder-262k/{fid}",
        }

    out = {
        "meta": {
            "trait": ssv.get("trait", "good"),
            "layer": ssv.get("layer", 16),
            "sae_id": ssv.get("sae_id"),
            "method": ssv.get("method"),
            "k_values": [lvl["k"] for lvl in k_levels],
            "n_features_with_logit_lens": len(lens_by_fid),
            "note": "Bubble size = |SSV weight| normalized within K. Green = amplified for Good, red = suppressed.",
        },
        "k_levels": k_levels,
        "feature_meta": feature_meta,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(k_levels)} K levels, {len(all_fids)} unique features)")


if __name__ == "__main__":
    main()
