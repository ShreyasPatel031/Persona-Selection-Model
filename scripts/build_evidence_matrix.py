#!/usr/bin/env python3
"""Join convergence, necessity, sufficiency, and logit-lens into one evidence table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.trait_sae_config import resolve_trait


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def feature_labels(logit_lens: dict | list | None) -> dict[int, str]:
    if not logit_lens:
        return {}
    rows = logit_lens
    if isinstance(logit_lens, dict):
        rows = logit_lens.get("features") or logit_lens.get("feature_rows") or []
    out: dict[int, str] = {}
    for row in rows:
        fid = row.get("feature_id") or row.get("fid")
        if fid is None:
            continue
        label = row.get("label") or row.get("interpretation") or ""
        if not label and row.get("top_tokens"):
            toks = row["top_tokens"]
            if toks and isinstance(toks[0], (list, tuple)):
                label = ", ".join(str(t[0]) for t in toks[:5])
        if label:
            out[int(fid)] = str(label)[:120]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--sae-dir", default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    tag = f"l{layer}"
    sae = Path(args.sae_dir or cfg["sae_dir"])
    conv = load_json(sae / f"feature_convergence_{tag}_k{args.k}.json")
    if conv is None:
        conv = load_json(sae / f"feature_convergence_{tag}_k20.json")

    necessity = load_json(sae / f"necessity_default_good_{tag}.json")
    sufficiency = load_json(sae / f"sufficiency_baseline_matrix_{tag}.json")
    chen = load_json(sae / f"chen_m32_top50_20q_{tag}.json")
    logit = load_json(sae / f"ssv_feature_logit_lens_262k_{tag}.json")
    labels = feature_labels(logit)

    rows: list[dict] = []

    if conv:
        core = (conv.get("method_invariant_core") or {}).get("2") or []
        for c in core:
            fid = int(c["feature_id"])
            rows.append(
                {
                    "feature_id": fid,
                    "role": "method_invariant_core",
                    "votes": c["votes"],
                    "sources": c.get("sources"),
                    "logit_lens_label": labels.get(fid),
                }
            )

    if necessity:
        baseline = next(
            (s["mean_trait"] for s in necessity.get("sets") or [] if s["set_name"] == "baseline_none"),
            None,
        )
        for s in necessity.get("sets") or []:
            if s["set_name"] == "baseline_none":
                continue
            rows.append(
                {
                    "set_name": s["set_name"],
                    "test": "necessity_default_good",
                    "n_features": s["n_features"],
                    "mean_trait": s.get("mean_trait"),
                    "delta_from_baseline": s.get("delta_from_baseline"),
                    "baseline_pos_prompt": baseline,
                    "frac_incoherent": s.get("frac_incoherent"),
                }
            )

    if sufficiency:
        for c in sufficiency.get("conditions") or []:
            rows.append(
                {
                    "condition": c["label"],
                    "test": "sufficiency_baseline",
                    "mean_trait": c.get("mean_trait"),
                    "sys_prompt": c.get("sys_prompt"),
                }
            )

    chen_summary = None
    if chen:
        feats = chen.get("features") or []
        passing = [f for f in feats if (f.get("best_mean") or 0) >= (chen.get("t_pass") or 50)]
        chen_summary = {
            "n_features_tested": len(feats),
            "n_pass_t50": len(passing),
            "top5": sorted(
                feats,
                key=lambda f: -(f.get("best_mean") or 0),
            )[:5],
            "formal_proof": chen.get("formal_proof"),
        }

    payload = {
        "method": "good_feature_evidence_matrix",
        "trait": cfg["trait"],
        "layer": layer,
        "convergence": {
            "pairwise_jaccard": conv.get("pairwise_jaccard") if conv else None,
            "core_votes2": conv.get("method_invariant_core", {}).get("2") if conv else None,
        },
        "necessity_baseline_pos": baseline if necessity else None,
        "sufficiency_conditions": sufficiency.get("conditions") if sufficiency else None,
        "chen_top50_summary": chen_summary,
        "rows": rows,
        "claim_template": (
            "Good trait is not recoverable from any single top-cos SAE feature (Chen M.3.2). "
            "Independent selectors (GradSAE, OMP, SSV, F-stat) show low overlap (non-identifiability). "
            "Method-invariant core + necessity under pos prompt + sufficiency residual-add vs DiffMean "
            "support one defensible sparse handle, explicitly non-unique."
        ),
        "missing_artifacts": [
            str(p.name)
            for p, d in [
                (sae / f"necessity_default_good_{tag}.json", necessity),
                (sae / f"sufficiency_baseline_matrix_{tag}.json", sufficiency),
                (sae / f"chen_m32_top50_20q_{tag}.json", chen),
            ]
            if d is None
        ],
    }

    out = Path(args.out or sae / "good_feature_evidence_matrix.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    if payload["missing_artifacts"]:
        print(f"Missing: {payload['missing_artifacts']}")
    if chen_summary:
        print(f"Chen top50: {chen_summary['n_features_tested']} tested, {chen_summary['n_pass_t50']} pass T50")


if __name__ == "__main__":
    main()
