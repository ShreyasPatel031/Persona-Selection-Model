#!/usr/bin/env python3
"""Re-score the opposite-prior sweeps against a dose-matched random control.

The ``works`` flag in ``scripts/opposite_prior_ipip.py`` compares the trait
direction's best movement anywhere on the dose grid against the random control's
largest movement anywhere on the dose grid. Those are usually different doses, so
the comparison is trait-at-its-best against noise-at-its-worst, and it fails a
pole whose signal is clean at moderate dose but swamped by degradation at the
ceiling. Conscientiousness-up is the clear example: at α=790 the trait direction
has moved the inventory +1.49 EV while the matched random direction has moved
+0.26, but at the top rung both have moved about +1.75, and the ceiling rung is
what the flag reports.

A random direction is a control for *what a perturbation of this size does*, so
it has to be read at the same size. This script reports, per pole, the largest
trait movement at a dose where the matched control is still quiet, the margin at
that same dose, and the ordering over that band only.

Nothing is re-run: this is reanalysis of the committed sweep JSON. The band is
defined by the control's behaviour, never by the trait's, so the choice of dose
cannot be tuned to flatter the trait direction.

    python3 scripts/dose_matched_control.py results/opposite_prior_ipip/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# A quarter of a Likert point: below this the matched random direction has not
# meaningfully disturbed the readout.
CONTROL_QUIET = 0.25
PASS_RHO = 0.8
PASS_MARGIN = 2.0
MIN_BAND_RUNGS = 4


def _score_pole(row: dict) -> dict:
    from app.persona.intensity_ladder import monotone_fraction, spearman_rho

    grid = [0.0] + [abs(float(m)) for m in row["magnitude_grid"]]
    trait = [float(x) for x in row["ev_curve"]]
    controls = [[float(x) for x in c["ev_curve"]] for c in row["control_curves"]]
    usable = [bool(x) for x in row["usable_flags"]]
    pole = row["pole"]
    n = min(len(grid), len(trait), *(len(c) for c in controls))

    base_t = trait[0]
    rows = []
    for i in range(n):
        ctrl_delta = max(abs(c[i] - c[0]) for c in controls)
        rows.append(
            {
                "magnitude": round(grid[i], 2),
                "trait_delta": round(trait[i] - base_t, 4),
                "max_control_abs_delta": round(ctrl_delta, 4),
                "control_quiet": ctrl_delta <= CONTROL_QUIET,
                "usable": usable[i] if i < len(usable) else False,
            }
        )

    band = [r for r in rows if r["control_quiet"] and r["usable"]]
    signed = (lambda d: d) if pole == "up" else (lambda d: -d)
    best = max(band, key=lambda r: signed(r["trait_delta"])) if band else None

    xs = [r["magnitude"] for r in band]
    ys = [signed(r["trait_delta"]) for r in band]
    rho = spearman_rho(xs, ys) if len(band) >= 2 else None

    if best is None:
        margin = None
    elif best["max_control_abs_delta"] <= 0.0:
        margin = None if abs(best["trait_delta"]) == 0.0 else float("inf")
    else:
        margin = abs(best["trait_delta"]) / best["max_control_abs_delta"]

    sign_ok = best is not None and signed(best["trait_delta"]) > 0
    gap = row.get("prompt_gap")
    passes = bool(
        sign_ok
        and rho is not None
        and rho >= PASS_RHO
        and len(band) >= MIN_BAND_RUNGS
        and margin is not None
        and margin >= PASS_MARGIN
    )
    return {
        "trait": row["trait"],
        "pole": pole,
        "layer": row.get("layer"),
        "prompt_style": row.get("prompt_style"),
        "n_markers": row.get("n_markers"),
        "prior_ev": row.get("prior_ev"),
        "reference_ev": row.get("reference_ev"),
        "prompt_gap": gap,
        "n_band_rungs": len(band),
        "band_max_magnitude": max(xs) if xs else None,
        "best_dose": None if best is None else best["magnitude"],
        "trait_delta_at_best_dose": None if best is None else best["trait_delta"],
        "control_delta_at_best_dose": None if best is None else best["max_control_abs_delta"],
        "dose_matched_margin": None if margin is None else round(margin, 2),
        "rho_over_band": None if rho is None else round(float(rho), 4),
        "monotone_fraction_over_band": (
            None if len(ys) < 2 else round(float(monotone_fraction(ys) or 0.0), 4)
        ),
        "pct_of_prompt_gap": (
            None
            if best is None or not gap
            else round(100.0 * best["trait_delta"] / float(gap), 1)
        ),
        "sign_correct": sign_ok,
        "passes_dose_matched": passes,
        "reported_works": row.get("works"),
        "per_rung": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sweeps", nargs="+", help="opposite-prior summary JSON files")
    p.add_argument("--out", default="results/dose_matched_control.json")
    args = p.parse_args(argv)

    out: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "dose_matched_control",
        "criterion": {
            "control_quiet_ev": CONTROL_QUIET,
            "rho": PASS_RHO,
            "margin": PASS_MARGIN,
            "min_band_rungs": MIN_BAND_RUNGS,
        },
        "sweeps": [],
    }

    for path in args.sweeps:
        src = Path(path)
        blob = json.loads(src.read_text(encoding="utf-8"))
        table = blob.get("table") or []
        if not table or "control_curves" not in table[0]:
            continue
        scored = [_score_pole(r) for r in table]
        out["sweeps"].append(
            {
                "file": src.name,
                "layer_source": blob.get("layer_source"),
                "layers": blob.get("layers"),
                "n_poles": len(scored),
                "n_passing_dose_matched": sum(1 for s in scored if s["passes_dose_matched"]),
                "n_passing_as_reported": sum(1 for s in scored if s["reported_works"]),
                "table": scored,
            }
        )

        print("\n" + "=" * 112)
        print(f"{src.name}   dose-matched control (band = rungs where |control Δ| ≤ {CONTROL_QUIET})")
        print("=" * 112)
        head = (
            f"{'trait':<18}{'pole':<6}{'L':>3}{'band':>5}{'dose':>8}"
            f"{'traitΔ':>9}{'ctrlΔ':>8}{'margin':>8}{'ρ':>7}{'%gap':>7}  verdict"
        )
        print(head)
        for s in scored:
            margin = s["dose_matched_margin"]
            print(
                f"{s['trait']:<18}{s['pole']:<6}{s['layer'] or 0:>3}{s['n_band_rungs']:>5}"
                f"{(s['best_dose'] or 0):>8.0f}{(s['trait_delta_at_best_dose'] or 0):>+9.3f}"
                f"{(s['control_delta_at_best_dose'] or 0):>8.3f}"
                f"{('inf' if margin == float('inf') else f'{margin or 0:.2f}'):>8}"
                f"{(s['rho_over_band'] or 0):>+7.2f}{(s['pct_of_prompt_gap'] or 0):>7.1f}"
                f"  {'PASS' if s['passes_dose_matched'] else 'fail'}"
                f"{'' if s['passes_dose_matched'] == bool(s['reported_works']) else '  (differs from reported)'}"
            )
        print(
            f"passing dose-matched: {out['sweeps'][-1]['n_passing_dose_matched']}/{len(scored)}"
            f"   as reported: {out['sweeps'][-1]['n_passing_as_reported']}/{len(scored)}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
