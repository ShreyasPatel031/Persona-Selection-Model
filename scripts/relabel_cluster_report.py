#!/usr/bin/env python3
"""Re-label cluster report clusters using logit-lens themes (no re-steering)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ssv_lens_themes import cluster_theme_from_lens
from trait_sae_config import resolve_trait


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", default="good")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--lens", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = cfg["layer"]
    report_path = args.report or cfg["sae_dir"] / "ssv_cluster_report.json"
    lens_path = args.lens or cfg["sae_dir"] / f"ssv_feature_logit_lens_262k_l{layer}.json"
    out_path = args.out or report_path

    report = json.loads(report_path.read_text(encoding="utf-8"))
    lens_rows = json.loads(lens_path.read_text(encoding="utf-8"))
    lens_by_fid = {int(r["fid"]): r for r in lens_rows}

    for cluster in report.get("clusters", []):
        fids = cluster.get("feature_ids", [])
        old = cluster.get("label", "")
        cluster["label"] = cluster_theme_from_lens(lens_by_fid, fids, args.trait)
        cluster["label_source"] = "logit_lens"
        cluster["label_prev"] = old

    for row in report.get("causal", []):
        cid = row.get("cluster_id")
        if cid is None:
            continue
        for cluster in report["clusters"]:
            if cluster["cluster_id"] == cid:
                row["cluster_label"] = cluster["label"]
                break

    for row in report.get("forward_selection", []):
        cid = row.get("added_cluster")
        if cid is None:
            continue
        for cluster in report["clusters"]:
            if cluster["cluster_id"] == cid:
                row["cluster_label"] = cluster["label"]
                break

    report.setdefault("meta", {})["cluster_label_source"] = "logit_lens"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    for c in report["clusters"]:
        print(f"  cluster {c['cluster_id']}: {c.get('label_prev')} -> {c['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
