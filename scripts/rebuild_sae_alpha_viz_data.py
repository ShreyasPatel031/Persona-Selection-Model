#!/usr/bin/env python3
"""Rebuild sae_alpha_viz_data.json/js from sweep + logit lens labels."""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "persona_runs/dnd_good_scale/sae/alpha_sweep_analysis.json"
LENS = REPO / "app/static/logit_lens_l16_all.json"
INTERP = REPO / "app/static/feature_interpretations.json"
OUT_JSON = REPO / "app/static/sae_alpha_viz_data.json"
OUT_JS = REPO / "app/static/sae_alpha_viz_data.js"

JUNK_EXACT = {
    "PLDNN", "MalayMarks", "DenovoMis", "posTocco", "doneProcessAvg", "odikwa",
    "Geografia", "cercando", ".</", "\ufffd",
}


def is_readable_token(tok: str) -> bool:
    t = str(tok).strip()
    if not t or len(t) > 28:
        return False
    if t in JUNK_EXACT or t.startswith("<unused"):
        return False
    if re.match(r"^[\W\d_]+$", t):
        return False
    # Prefer ASCII words; allow short non-ascii if mostly letters
    if t.isascii():
        return bool(re.search(r"[a-zA-Z]", t))
    letters = sum(c.isalpha() for c in t)
    return letters >= max(2, len(t) // 2)


def pick_tokens(pairs: list, n: int = 20) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for tok, score in pairs:
        if is_readable_token(tok):
            out.append((tok, float(score)))
        if len(out) >= n:
            break
    if not out and pairs:
        out = [(str(pairs[i][0]).strip(), float(pairs[i][1])) for i in range(min(n, len(pairs)))]
    return out


def load_interpretations(path: Path = INTERP) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(doc.get("features"), dict):
        return doc["features"]
    return {k: v for k, v in doc.items() if k.isdigit()}


def describe_feature(fid: int, row: dict | None, interp: dict | None = None) -> dict:
    if not row:
        interpretation = (interp or {}).get("interpretation") if interp else None
        return {
            "short": interpretation or f"F{fid}",
            "interpretation": interpretation,
            "interpretation_source": (interp or {}).get("source") if interp else None,
            "tooltip": interpretation or f"F{fid}: no logit-lens data available.",
            "boost": [],
            "suppress": [],
        }
    boost = pick_tokens(row.get("top_boost") or [], 20)
    supp = pick_tokens(row.get("top_suppress") or [], 15)
    raw_boost = row.get("top_boost") or []
    raw_supp = row.get("top_suppress") or []
    interpretation = (interp or {}).get("interpretation") if interp else None
    interpretation_source = (interp or {}).get("source") if interp else None

    short_tokens = []
    seen = set()
    for t, _ in boost:
        if not is_readable_token(t):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        short_tokens.append(t)
        if len(short_tokens) >= 4:
            break
    token_short = ", ".join(short_tokens) if short_tokens else f"F{fid}"
    short = interpretation or token_short

    evidence: list[str] = ["Logit-lens evidence — MORE likely to generate:"]
    if boost:
        evidence.extend(f"  • \"{t}\"  (logit +{s:.2f})" for t, s in boost)
    elif raw_boost:
        evidence.append("  (no clean English tokens in top-k; raw logit-lens boost:)")
        evidence.extend(f"  • \"{str(p[0]).strip()}\"  (logit +{float(p[1]):.2f})" for p in raw_boost[:15])
    else:
        evidence.append("  (no boost tokens)")
    evidence.append("")
    evidence.append("LESS likely to generate:")
    if supp:
        evidence.extend(f"  • \"{t}\"  (logit −{s:.2f})" for t, s in supp)
    elif raw_supp:
        evidence.append("  (no clean English tokens in top-k; raw logit-lens suppress:)")
        evidence.extend(f"  • \"{str(p[0]).strip()}\"  (logit −{float(p[1]):.2f})" for p in raw_supp[:15])
    else:
        evidence.append("  (no suppress tokens)")

    if interpretation:
        tooltip = "\n".join([
            interpretation,
            f"Source: {interpretation_source or 'gemini'} · F{fid} · layer-16 SAE",
        ])
    else:
        tooltip = f"F{fid} · no Gemini interpretation · logit lens only"

    return {
        "short": short,
        "interpretation": interpretation,
        "interpretation_source": interpretation_source,
        "tooltip": tooltip,
        "evidence": "\n".join(evidence),
        "boost": boost,
        "suppress": supp,
    }


def main() -> int:
    lens_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LENS
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    lens = json.loads(lens_path.read_text(encoding="utf-8")) if lens_path.is_file() else {}
    interpretations = load_interpretations()

    alpha_keys = [f"{a:g}" for a in sweep["analysis"]["alphas"]]
    all_fids: set[int] = set()
    by_alpha: dict[str, list] = {}

    for ak in alpha_keys:
        acc: dict[int, list[float]] = defaultdict(list)
        for pq in sweep["per_question"]:
            for row in pq["top_features_by_alpha"].get(ak, []):
                acc[row["feature_id"]].append(row["activation"])
        rows = []
        for fid, vals in acc.items():
            all_fids.add(fid)
            desc = describe_feature(fid, lens.get(str(fid)), interpretations.get(str(fid)))
            rows.append({
                "feature_id": fid,
                "mean_activation": sum(vals) / len(vals),
                "mean_abs": sum(abs(v) for v in vals) / len(vals),
                "n_questions": len(vals),
                "label": desc["short"],
                "tooltip": desc["tooltip"],
            })
        rows.sort(key=lambda r: r["mean_abs"], reverse=True)
        by_alpha[ak] = rows

    labels = {
        str(fid): describe_feature(fid, lens.get(str(fid)), interpretations.get(str(fid)))
        for fid in all_fids
    }

    rank_matrix = {
        str(fid): {
            ak: next((i + 1 for i, r in enumerate(by_alpha[ak]) if r["feature_id"] == fid), None)
            for ak in alpha_keys
        }
        for fid in all_fids
    }

    tracked = sorted(all_fids)
    traj = {}
    for fid in tracked:
        desc = labels[str(fid)]
        series = []
        for ak in alpha_keys:
            vals = []
            for pq in sweep["per_question"]:
                for row in pq["top_features_by_alpha"].get(ak, []):
                    if row["feature_id"] == fid:
                        vals.append(row["activation"])
            rank = rank_matrix[str(fid)][ak]
            series.append({
                "alpha": float(ak),
                "mean_activation": sum(vals) / len(vals) if vals else None,
                "mean_abs": sum(abs(v) for v in vals) / len(vals) if vals else None,
                "rank": rank,
                "in_top20": rank is not None and rank <= 20,
            })
        traj[str(fid)] = {
            "feature_id": fid,
            "label": desc["short"],
            "tooltip": desc["tooltip"],
            "series": series,
        }

    rank_changes = {}
    for i in range(1, len(alpha_keys)):
        prev, cur = alpha_keys[i - 1], alpha_keys[i]
        prev_top = {r["feature_id"] for r in by_alpha[prev][:20]}
        cur_top = {r["feature_id"] for r in by_alpha[cur][:20]}
        rank_changes[f"{prev}->{cur}"] = {
            "new_in_top20": sorted(cur_top - prev_top),
            "dropped_from_top20": sorted(prev_top - cur_top),
        }

    viz = {
        "meta": {
            "run_id": sweep["run_id"],
            "layer": sweep["layer"],
            "sae_id": sweep["sae_id"],
            "n_questions": sweep["n_questions"],
            "alphas": [float(a) for a in alpha_keys],
            "label_method": "Gemini autointerp over logit lens + token evidence",
            "n_features_labeled": len(labels),
            "data_version": int(time.time()),
        },
        "labels": labels,
        "mean_delta_l2_by_alpha": sweep["analysis"]["mean_delta_l2_by_alpha"],
        "by_alpha_top": {ak: rows[:20] for ak, rows in by_alpha.items()},
        "rank_matrix": rank_matrix,
        "rank_changes": rank_changes,
        "trajectories": traj,
        "questions": [
            {"question": pq["question"], "replies": pq["replies"]}
            for pq in sweep["per_question"]
        ],
        "emergence": {
            "early": sweep["analysis"]["early_emerging_features"][:15],
            "late": sweep["analysis"]["late_emerging_features"][:10],
        },
    }

    OUT_JSON.write_text(json.dumps(viz, indent=2) + "\n", encoding="utf-8")
    OUT_JS.write_text(
        "window.__SAE_VIZ_DATA__ = " + json.dumps(viz, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_JSON} ({len(labels)} features labeled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
