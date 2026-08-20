#!/usr/bin/env python3
"""Re-derive ladder vectors from prompts that actually establish their levels.

A ladder direction is fitted across level centroids, so it inherits whatever the
level prompts really did to the model. Where a level prompt fails to move the
target domain, the centroids at that end of the ladder differ from the other end
by something other than the trait, and the fitted direction is that something.
On Gemma-3-4B the low-openness levels are the failure case: the model does not
adopt them, so the openness ladder is a plausible hedging axis rather than an
openness axis — consistent with an openness vector that moves agreeableness and
conscientiousness on the inventory while leaving openness flat.

This script therefore refuses to guess. It reads the calibration written by
``scripts/calibrate_prior_prompts.py``, picks the framing that best establishes
both poles of each trait, and rebuilds only the traits whose priors clear the
gate. Traits with no usable framing are reported as such and left alone: a vector
derived from prompts that do not work is worse than no vector, because it looks
usable.

    PYTHONPATH=. python3 scripts/rebuild_ladder_vectors.py \\
        --calibration results/prior_prompt_calibration/summary.json \\
        --out-dir vectors_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("rebuild_ladder_vectors")


def _choose_style(rows: list[dict], trait: str, styles: list[str]) -> tuple[str | None, dict]:
    """Pick the framing that establishes both poles, preferring the widest gap.

    Falls back to the framing with the widest usable prompt gap when neither
    clears the full gate, so the caller can decide whether to accept it — but the
    fallback is reported, never silently promoted to a pass.
    """
    detail: dict = {}
    passing: list[tuple[float, str]] = []
    fallback: list[tuple[float, str]] = []
    for style in styles:
        poles = [r for r in rows if r["trait"] == trait and r["style"] == style]
        if len(poles) < 2:
            continue
        gap = min(abs(float(r["prompt_gap"])) for r in poles)
        detail[style] = {
            "min_abs_prompt_gap": round(gap, 4),
            "prior_evs": {r["pole"]: r["prior_ev"] for r in poles},
            "midpoint_fractions": {
                r["pole"]: r["prior_target_midpoint_fraction"] for r in poles
            },
            "both_poles_established": all(r["prior_established"] for r in poles),
            "midpoint_ok_both_poles": all(r["midpoint_ok"] for r in poles),
        }
        if detail[style]["both_poles_established"]:
            passing.append((gap, style))
        elif detail[style]["midpoint_ok_both_poles"]:
            fallback.append((gap, style))
    if passing:
        return max(passing)[1], detail
    if fallback:
        return None, detail | {"widest_unestablished": max(fallback)[1]}
    return None, detail


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibration", default="results/prior_prompt_calibration/summary.json")
    p.add_argument("--out-dir", default="vectors_v2")
    p.add_argument("--report", default="results/ladder_rebuild/summary.json")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--n-markers", type=int, default=6)
    p.add_argument(
        "--include-unestablished",
        action="store_true",
        help=(
            "Also rebuild traits whose best framing fails the gate, using the "
            "widest-gap framing that at least keeps the target domain off the "
            "neutral option. Recorded as unestablished in the report."
        ),
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.intensity_ladder import build_ladder_vectors, run_prompt_ladder

    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    rows = cal["table"]
    traits = sorted({r["trait"] for r in rows})
    styles = sorted({r["style"] for r in rows})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan: list[dict] = []
    for trait in traits:
        style, detail = _choose_style(rows, trait, styles)
        chosen = style
        established = style is not None
        if style is None and args.include_unestablished:
            chosen = detail.get("widest_unestablished")
        plan.append(
            {
                "trait": trait,
                "chosen_style": chosen,
                "priors_established": established,
                "per_style": detail,
            }
        )
        logger.info(
            "%-18s style=%-10s established=%s",
            trait,
            chosen or "NONE",
            established,
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ladder_rebuild",
        "model_id": args.model_id,
        "calibration": str(Path(args.calibration).resolve()),
        "n_markers": args.n_markers,
        "variants": args.variants,
        "out_dir": str(out_dir.resolve()),
        "plan": plan,
        "built": [],
        "skipped": [],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    flush()

    for entry in plan:
        trait = entry["trait"]
        style = entry["chosen_style"]
        if style is None:
            logger.warning(
                "%s: no framing establishes both priors; not rebuilding. A vector "
                "fitted to these levels would encode whatever the prompts did "
                "instead of the trait.",
                trait,
            )
            report["skipped"].append({"trait": trait, "reason": "no framing establishes priors"})
            flush()
            continue

        ladder_json = out_dir / f"prompt_ladder_{trait}.json"
        centroids = out_dir / f"centroids_{trait}.pt"
        vectors = out_dir / f"ladder_vectors_{trait}.pt"
        geometry = out_dir / f"ladder_geometry_{trait}.json"

        logger.info("rebuilding %s with style=%s n_markers=%s", trait, style, args.n_markers)
        run_prompt_ladder(
            ladder_json,
            centroids,
            trait=trait,
            model_id=args.model_id,
            variants=args.variants,
            n_markers=args.n_markers,
            items_csv=Path(args.items_csv),
            style=style,
        )
        build_ladder_vectors(centroids, vectors, geometry)

        ladder = json.loads(ladder_json.read_text(encoding="utf-8"))
        geo = json.loads(geometry.read_text(encoding="utf-8"))
        report["built"].append(
            {
                "trait": trait,
                "style": style,
                "priors_established": entry["priors_established"],
                "vectors_pt": str(vectors.resolve()),
                "prompting_rho_level_vs_score": ladder.get(
                    "spearman_level_vs_target_score"
                ),
                "level_mean_target_score": ladder.get("level_mean_target_score"),
                "target_score_range": ladder.get("target_score_range"),
                "n_locked_administrations": len(ladder.get("locked_administrations") or []),
                "off_target_spearman": ladder.get("off_target_spearman"),
                "best_layer": geo.get("geometry", {}).get("best_layer"),
            }
        )
        flush()
        logger.info(
            "%s rebuilt: prompting rho=%s range=%s locked=%s",
            trait,
            ladder.get("spearman_level_vs_target_score"),
            ladder.get("target_score_range"),
            len(ladder.get("locked_administrations") or []),
        )

    print("\n" + "=" * 96)
    print("LADDER REBUILD")
    print("=" * 96)
    for b in report["built"]:
        print(
            f"  {b['trait']:<18}style={b['style']:<10}"
            f"rho={b['prompting_rho_level_vs_score']}  range={b['target_score_range']}  "
            f"locked={b['n_locked_administrations']}  best_layer={b['best_layer']}"
        )
    for s in report["skipped"]:
        print(f"  {s['trait']:<18}SKIPPED: {s['reason']}")
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
