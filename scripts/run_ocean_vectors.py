#!/usr/bin/env python3
"""Get OCEAN vectors and decide whether they work, for one trait or all five.

Runs, per trait: prompt ladder -> ladder directions -> validated bipolar sweep,
then prints the three things that decide whether a vector is real.

    monotonicity   Spearman rho between |magnitude| and the inventory score,
                   over unlocked rungs only.
    correlation    the prompting baseline rho(level, score) for the same trait
                   and instrument, as the reference the vector is compared to,
                   plus rho between magnitude and free-text trait markers.
    behaviour      free-text replies at every rung with a coherence verdict, so
                   a score obtained past the coherence ceiling is not mistaken
                   for a working one.

A trait passes only if the direction beats matched-norm random controls, keeps at
least three unlocked rungs, and moves the score monotonically.

    python3 scripts/run_ocean_vectors.py --run-id ocean_v1 --trait conscientiousness
    python3 scripts/run_ocean_vectors.py --run-id ocean_v1 --all-traits
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.persona.inventory_ipip import TRAITS  # noqa: E402

DEFAULT_ITEMS = REPO_ROOT / "data" / "ipip_neo_120.csv"


def run_trait(
    trait: str,
    run_dir: Path,
    *,
    model_id: str | None,
    items_csv: Path | None,
    magnitudes: list[float] | None,
    auto_calibrate: bool,
    n_rungs: int,
    steer_toward: str,
    direction: str,
    variants: int,
    n_random_controls: int,
    n_probes: int,
    max_new_tokens: int,
    skip_ladder: bool,
) -> dict:
    from app.persona.intensity_ladder import (
        build_ladder_vectors,
        run_prompt_ladder,
        run_validated_sweep,
    )
    from app.persona.ocean_probes import PROBE_QUESTIONS

    stage_dir = run_dir / "ladder"
    stage_dir.mkdir(parents=True, exist_ok=True)
    ladder_json = stage_dir / f"prompt_ladder_{trait}.json"
    centroids = stage_dir / f"centroids_{trait}.pt"
    vec_pt = stage_dir / f"ladder_vectors_{trait}.pt"
    geom_json = stage_dir / f"ladder_geometry_{trait}.json"
    sweep_json = stage_dir / f"validated_sweep_{trait}_{direction}.json"

    if not (skip_ladder and ladder_json.is_file() and centroids.is_file()):
        logging.info("[%s] stage 1: prompt ladder", trait)
        run_prompt_ladder(
            ladder_json,
            centroids,
            trait=trait,
            model_id=model_id,
            variants=variants,
            items_csv=items_csv,
        )
    if not (skip_ladder and vec_pt.is_file()):
        logging.info("[%s] stage 2: ladder directions", trait)
        build_ladder_vectors(centroids, vec_pt, geom_json)

    logging.info("[%s] stage 3: validated sweep", trait)
    run_validated_sweep(
        vec_pt,
        sweep_json,
        trait=trait,
        which=direction,
        magnitudes=magnitudes or None,
        auto_calibrate=auto_calibrate,
        n_rungs=n_rungs,
        steer_toward=steer_toward,
        n_random_controls=n_random_controls,
        model_id=model_id,
        items_csv=items_csv,
        probe_questions=PROBE_QUESTIONS[: max(1, n_probes)],
        max_new_tokens=max_new_tokens,
    )

    ladder = json.loads(ladder_json.read_text())
    sweep = json.loads(sweep_json.read_text())
    curve = sweep["trait_curve"]
    verdict = sweep["verdict"]
    return {
        "trait": trait,
        "prompting_rho": ladder["spearman_level_vs_target_score"],
        "prompting_range": ladder["target_score_range"],
        "prompting_usable": f"{ladder['n_usable_administrations']}/{ladder['n_administrations']}",
        "steered_toward": verdict["steered_toward"],
        "steering_rho": curve["spearman_absalpha_vs_target_ev"],
        "steering_usable": f"{curve['n_usable_rungs']}/{curve['n_rungs']}",
        "best_delta": (curve["best_usable"] or {}).get("delta_vs_baseline"),
        "best_magnitude": (curve["best_usable"] or {}).get("magnitude"),
        "control_delta": verdict["max_control_abs_delta"],
        "beats_controls": verdict["beats_random_controls"],
        "marker_rho": curve["marker_spearman"],
        "calibrated_ceiling": (sweep.get("magnitude_calibration") or {}).get("ceiling_magnitude"),
        "layer": sweep["layer"],
        "layer_choice": sweep.get("layer_choice"),
        "coherence_ceiling": curve["coherence_ceiling_magnitude"],
        "control_ceilings": verdict["control_coherence_ceilings"],
        "works": verdict["works"],
        "report": str(sweep_json),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--trait", default="conscientiousness", choices=list(TRAITS))
    p.add_argument("--all-traits", action="store_true", help="Run all five domains.")
    p.add_argument("--model-id", default="", help="HF model id (default: GEMMA_MODEL_ID).")
    p.add_argument(
        "--items-csv",
        default=str(DEFAULT_ITEMS),
        help="Inventory CSV; pass empty to use the built-in IPIP-50.",
    )
    p.add_argument(
        "--magnitudes",
        default="",
        help="|alpha| grid in units of the layer's mean activation norm. "
        "Empty (default) calibrates the grid to the measured coherence ceiling.",
    )
    p.add_argument(
        "--no-auto-calibrate",
        action="store_true",
        help="Use --magnitudes verbatim instead of calibrating to the coherence ceiling.",
    )
    p.add_argument("--rungs", type=int, default=6, help="Rungs in the calibrated grid.")
    p.add_argument("--steer-toward", default="auto", choices=("auto", "high", "low"))
    p.add_argument("--direction", default="pc1", choices=("pc1", "endpoint", "ordinal"))
    p.add_argument("--variants", type=int, default=3, help="Marker rotations per ladder level.")
    p.add_argument("--random-controls", type=int, default=2)
    p.add_argument("--probes", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--skip-ladder-if-present",
        action="store_true",
        help="Reuse existing ladder artefacts and only re-run the sweep.",
    )
    p.add_argument("--out", default="", help="Summary JSON path.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.config import PERSONA_RUNS_DIR

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    items_csv = Path(args.items_csv) if args.items_csv else None
    magnitudes = [float(x.strip()) for x in args.magnitudes.split(",") if x.strip()]
    traits = list(TRAITS) if args.all_traits else [args.trait]

    rows: list[dict] = []
    for trait in traits:
        try:
            rows.append(
                run_trait(
                    trait,
                    run_dir,
                    model_id=args.model_id or None,
                    items_csv=items_csv,
                    magnitudes=magnitudes,
                    auto_calibrate=not args.no_auto_calibrate,
                    n_rungs=args.rungs,
                    steer_toward=args.steer_toward,
                    direction=args.direction,
                    variants=args.variants,
                    n_random_controls=args.random_controls,
                    n_probes=args.probes,
                    max_new_tokens=args.max_new_tokens,
                    skip_ladder=args.skip_ladder_if_present,
                )
            )
        except Exception as exc:  # keep going so one trait cannot sink the run
            logging.exception("[%s] failed: %s", trait, exc)
            rows.append({"trait": trait, "error": str(exc), "works": False})

    print("\n" + "=" * 100)
    print("DOES THE VECTOR WORK?  (usable = rungs that were not option-locked)")
    print("=" * 100)
    header = (
        f"{'trait':17} {'toward':6} {'steer rho':>9} {'usable':>7} {'best delta':>10} "
        f"{'@mag':>7} {'ctrl delta':>10} {'marker rho':>10} {'ceiling':>8} {'works':>6}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        if r.get("error"):
            print(f"{r['trait']:17} ERROR: {r['error'][:70]}")
            continue
        print(
            f"{r['trait']:17} {str(r['steered_toward']):6} {str(r['steering_rho']):>9} "
            f"{r['steering_usable']:>7} {str(r['best_delta']):>10} {str(r['best_magnitude']):>7} "
            f"{str(r['control_delta']):>10} {str(r['marker_rho']):>10} "
            f"{str(r['coherence_ceiling']):>8} {str(r['works']):>6}"
        )
    print("-" * 100)
    print("prompting baseline for reference (same instrument):")
    for r in rows:
        if r.get("error"):
            continue
        print(
            f"  {r['trait']:17} rho={str(r['prompting_rho']):>7} range={str(r['prompting_range']):>16} "
            f"usable={r['prompting_usable']}"
        )

    working = [r["trait"] for r in rows if r.get("works")]
    print(f"\nvectors that pass all three checks: {working or 'none'}")

    out = Path(args.out) if args.out else run_dir / "ocean_vector_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": args.run_id, "traits": rows}, indent=2) + "\n")
    print(f"summary: {out}")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
