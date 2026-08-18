#!/usr/bin/env python3
"""Get OCEAN vectors and decide whether they work, for one trait or all five.

Runs, per trait: prompt ladder -> ladder directions -> validated sweep at each
pole, then prints the three things that decide whether a direction is real.

    monotonicity   Spearman rho between |magnitude| and the inventory score,
                   over unlocked rungs only.
    correlation    the prompting baseline rho(level, score) for the same trait
                   and instrument, as the reference the vector is compared to,
                   plus rho between magnitude and free-text trait markers.
    behaviour      free-text replies at every rung with a coherence verdict and
                   a refusal check, so a score obtained past the coherence
                   ceiling, or by the model dropping out of first person, is not
                   mistaken for a working one.

Both poles are swept by default. A baseline sitting near the scale midpoint makes
"which pole has headroom" close to a coin flip, and testing one pole can miss a
direction that works cleanly in the other.

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
    poles: list[str],
    direction: str,
    variants: int,
    n_random_controls: int,
    n_probes: int,
    max_new_tokens: int,
    baseline: str,
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

    per_pole: dict[str, dict] = {}
    for pole in poles:
        pole_json = stage_dir / f"validated_sweep_{trait}_{direction}_{pole}.json"
        logging.info("[%s] stage 3: validated sweep toward %s", trait, pole)
        run_validated_sweep(
            vec_pt,
            pole_json,
            trait=trait,
            which=direction,
            magnitudes=magnitudes or None,
            auto_calibrate=auto_calibrate,
            n_rungs=n_rungs,
            steer_toward=pole,
            n_random_controls=n_random_controls,
            model_id=model_id,
            items_csv=items_csv,
            probe_questions=PROBE_QUESTIONS[: max(1, n_probes)],
            max_new_tokens=max_new_tokens,
            baseline=baseline,
        )
        sw = json.loads(pole_json.read_text())
        curve = sw["trait_curve"]
        best = curve["best_usable"] or {}
        per_pole[pole] = {
            "report": str(pole_json),
            "baseline_ev": sw["baseline"]["target_ev"],
            "rho": curve["spearman_absalpha_vs_target_ev"],
            "usable": f"{curve['n_usable_rungs']}/{curve['n_rungs']}",
            "best_delta": best.get("delta_vs_baseline"),
            "best_magnitude": best.get("magnitude"),
            "control_delta": sw["verdict"]["max_control_abs_delta"],
            "margin": sw["verdict"]["control_margin_ratio"],
            "refused_at_best": sw["verdict"].get("refused_at_best_rung"),
            "marker_rho": curve["marker_spearman"],
            "ceiling": curve["coherence_ceiling_magnitude"],
            "works": sw["verdict"]["works"],
        }

    winners = [p for p, v in per_pole.items() if v["works"]]
    best_pole = (
        max(winners, key=lambda p: abs(per_pole[p]["best_delta"] or 0.0)) if winners else None
    )
    ladder = json.loads(ladder_json.read_text())
    return {
        "trait": trait,
        "prompting_rho": ladder["spearman_level_vs_target_score"],
        "prompting_range": ladder["target_score_range"],
        "prompting_usable": f"{ladder['n_usable_administrations']}/{ladder['n_administrations']}",
        "poles": per_pole,
        "passing_poles": winners,
        "best_pole": best_pole,
        "works": bool(winners),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "Empty (default) calibrates to the trait's measured latent span "
        "(0.25×…2× of prompt level 1→9 on PC1), clipped by the coherence ceiling.",
    )
    p.add_argument(
        "--no-auto-calibrate",
        action="store_true",
        help="Use --magnitudes verbatim instead of span-calibrating.",
    )
    p.add_argument(
        "--rungs",
        type=int,
        default=5,
        help="Rungs in the span-calibrated grid (default: 5 → 0.25/0.5/1/1.5/2× span).",
    )
    p.add_argument(
        "--steer-toward",
        default="both",
        choices=("both", "high", "low"),
        help="Which pole(s) to sweep (default: both).",
    )
    p.add_argument(
        "--baseline",
        default="persona_free",
        choices=("persona_free", "neutral_level5"),
        help="Unsteered baseline prompt. neutral_level5 instructs neutrality and "
        "pins a forced-choice inventory to the neutral option; kept only for comparison.",
    )
    p.add_argument("--direction", default="pc1", choices=("pc1", "endpoint", "ordinal"))
    p.add_argument("--variants", type=int, default=3, help="Marker rotations per ladder level.")
    p.add_argument("--random-controls", type=int, default=2)
    p.add_argument("--probes", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--skip-ladder-if-present",
        action="store_true",
        help="Reuse existing ladder artefacts and only re-run the sweeps.",
    )
    p.add_argument("--out", default="", help="Summary JSON path.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.config import PERSONA_RUNS_DIR

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    items_csv = Path(args.items_csv) if args.items_csv else None
    magnitudes = [float(x.strip()) for x in args.magnitudes.split(",") if x.strip()]
    traits = list(TRAITS) if args.all_traits else [args.trait]
    poles = ["high", "low"] if args.steer_toward == "both" else [args.steer_toward]

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
                    poles=poles,
                    direction=args.direction,
                    variants=args.variants,
                    n_random_controls=args.random_controls,
                    n_probes=args.probes,
                    max_new_tokens=args.max_new_tokens,
                    baseline=args.baseline,
                    skip_ladder=args.skip_ladder_if_present,
                )
            )
        except Exception as exc:  # keep going so one trait cannot sink the run
            logging.exception("[%s] failed: %s", trait, exc)
            rows.append({"trait": trait, "error": str(exc), "works": False})

    print("\n" + "=" * 112)
    print("DOES THE VECTOR WORK?  (usable = rungs that were not option-locked)")
    print("=" * 112)
    print(
        f"{'trait':18} {'pole':5} {'base':>6} {'rho':>6} {'usable':>7} {'delta':>8} "
        f"{'@mag':>9} {'ctrl':>7} {'margin':>7} {'refused':>7} {'works':>6}"
    )
    print("-" * 112)
    for r in rows:
        if r.get("error"):
            print(f"{r['trait']:18} ERROR: {r['error'][:80]}")
            continue
        for pole, v in r["poles"].items():
            print(
                f"{r['trait']:18} {pole:5} {str(v['baseline_ev']):>6} {str(v['rho']):>6} "
                f"{v['usable']:>7} {str(v['best_delta']):>8} {str(v['best_magnitude']):>9} "
                f"{str(v['control_delta']):>7} {str(v['margin']):>7} "
                f"{str(v['refused_at_best']):>7} {str(v['works']):>6}"
            )
    print("-" * 112)
    print("prompting baseline for reference (same instrument):")
    for r in rows:
        if r.get("error"):
            continue
        print(
            f"  {r['trait']:18} rho={str(r['prompting_rho']):>7} "
            f"range={str(r['prompting_range']):>16} usable={r['prompting_usable']}"
        )

    working = [(r["trait"], r["best_pole"]) for r in rows if r.get("works")]
    print(f"\nvectors that pass all checks: {working or 'none'}")

    out = Path(args.out) if args.out else run_dir / "ocean_vector_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"run_id": args.run_id, "baseline": args.baseline, "traits": rows}, indent=2)
        + "\n"
    )
    print(f"summary: {out}")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
