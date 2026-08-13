#!/usr/bin/env python3
"""Side-by-side comparison of SSV vs OMP d-sweep results for one trait."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.trait_sae_config import resolve_trait


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument(
        "--residual",
        action="store_true",
        help="Compare ssv_dsweep_residual_l{layer}.json instead of ssv_dsweep_l{layer}.json",
    )
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    sae_dir = cfg["sae_dir"]

    ssv_name = f"ssv_dsweep_residual_l{layer}.json" if args.residual else f"ssv_dsweep_l{layer}.json"
    ssv_path = sae_dir / ssv_name
    omp_path = sae_dir / f"ssv_omp_dsweep_l{layer}.json"
    if not ssv_path.is_file():
        raise SystemExit(f"Missing {ssv_path}")
    if not omp_path.is_file():
        raise SystemExit(f"Missing {omp_path}")

    ssv = json.loads(ssv_path.read_text())
    omp = json.loads(omp_path.read_text())
    ssv_by_d = {r["d"]: r for r in ssv.get("results") or []}
    omp_by_d = {r["d"]: r for r in omp.get("results") or []}

    print(f"\nSSV vs OMP d-sweep — trait={args.trait} layer={layer}\n")
    header = f"{'d':>4} | {'OMP mean':>8} | {'OMP features (top-3)':<28} | {'SSV mean':>8} | {'SSV features (top-3)':<28} | overlap"
    print(header)
    print("-" * len(header))

    for d in sorted(set(ssv_by_d) | set(omp_by_d)):
        o = omp_by_d.get(d, {})
        s = ssv_by_d.get(d, {})
        o_fids = o.get("feature_ids") or []
        s_fids = s.get("feature_ids") or []
        overlap = len(set(o_fids) & set(s_fids))
        o_top = ",".join(str(x) for x in o_fids[:3])
        s_top = ",".join(str(x) for x in s_fids[:3])
        print(
            f"{d:>4} | {str(o.get('mean')):>8} | {o_top:<28} | "
            f"{str(s.get('mean')):>8} | {s_top:<28} | {overlap}/{d}"
        )

    print("\nMetadata:")
    print(f"  OMP top-5: {(omp.get('omp_features') or [])[:5]}")
    print(f"  SSV F-stat top-5: {ssv.get('fstat_top5')}")
    print(f"  SSV opt top-5: {(ssv.get('ssv_features') or [])[:5]}")


if __name__ == "__main__":
    main()
