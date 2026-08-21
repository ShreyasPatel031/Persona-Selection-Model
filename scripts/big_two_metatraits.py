#!/usr/bin/env python3
"""Big Two as continuous metatrait scores, not as counted sign matches.

``scripts/big_two_covariance.py`` asks a binary question per partner trait: did it
move the way Digman's metatraits predict? That throws away magnitude and ordering,
and with two predicted partners per pole the whole test rests on 16 coin flips.

The metatraits are composites, so score them as composites at every dose:

    alpha (Stability)   = mean(C, A, reverse(N))
    beta  (Plasticity)  = mean(E, O)

with ``reverse(x) = 6 - x`` on the 1-5 scale. Steering one trait pushes one
metatrait and should leave the other alone, which turns the Big Two check into two
things a sign count cannot express: a dose-response correlation on the intended
metatrait, and a discriminant claim about the unintended one.

Two caveats stated rather than hidden. The target trait is a term in its own
composite, so alpha moving when conscientiousness is steered is partly definitional
-- the informative quantity is the *partner-only* composite, which drops the target
from the average, and it is reported alongside. And the sweep JSON stores all five
trait scores only for the trait direction, not for the random controls, so there is
no matched-control metatrait curve here.

    PYTHONPATH=. python3 scripts/big_two_metatraits.py \\
        results/opposite_prior_ipip/summary_v3_calibrated_prompts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCALE_MAX = 5
ALPHA = {"conscientiousness": +1, "agreeableness": +1, "neuroticism": -1}
BETA = {"extraversion": +1, "openness": +1}


def _composite(scores: dict, loadings: dict, drop: str | None = None) -> float | None:
    terms = [
        scores[t] if sign > 0 else float(SCALE_MAX + 1) - scores[t]
        for t, sign in loadings.items()
        if t != drop and t in scores
    ]
    return sum(terms) / len(terms) if terms else None


def _metatrait_of(trait: str) -> tuple[str, int]:
    if trait in ALPHA:
        return "alpha", ALPHA[trait]
    return "beta", BETA[trait]


def analyse(row: dict, band_max: float | None) -> dict:
    from app.persona.intensity_ladder import monotone_fraction, spearman_rho

    trait = row["trait"]
    pole = row["pole"]
    target_sign = 1 if pole == "up" else -1
    meta, loading = _metatrait_of(trait)
    other = "beta" if meta == "alpha" else "alpha"
    loadings = {"alpha": ALPHA, "beta": BETA}

    # Direction the metatrait is pushed when the target trait is pushed.
    meta_direction = target_sign * loading

    rungs = []
    for r in row["trait_rows"]:
        if not r.get("usable"):
            continue
        mag = abs(float(r["magnitude"]))
        if band_max is not None and mag > band_max + 1e-9:
            continue
        ev = r["ev_scores"]
        rungs.append(
            {
                "magnitude": round(mag, 2),
                "target": round(float(ev[trait]), 4),
                "alpha": round(_composite(ev, ALPHA), 4),
                "beta": round(_composite(ev, BETA), 4),
                "intended_partner_only": (
                    None
                    if (v := _composite(ev, loadings[meta], drop=trait)) is None
                    else round(v, 4)
                ),
                "unintended": round(_composite(ev, loadings[other]), 4),
            }
        )

    if len(rungs) < 3:
        return {"trait": trait, "pole": pole, "n_rungs": len(rungs), "usable": False}

    xs = [r["magnitude"] for r in rungs]

    def signed_rho(key: str, sign: int) -> float | None:
        rho = spearman_rho(xs, [sign * r[key] for r in rungs])
        return None if rho is None else round(float(rho), 4)

    base, top = rungs[0], rungs[-1]
    intended_key = meta
    d_intended = top[intended_key] - base[intended_key]
    d_partner = (
        None
        if top["intended_partner_only"] is None
        else top["intended_partner_only"] - base["intended_partner_only"]
    )
    d_unintended = top["unintended"] - base["unintended"]

    return {
        "trait": trait,
        "pole": pole,
        "usable": True,
        "n_rungs": len(rungs),
        "band_max_magnitude": band_max,
        "intended_metatrait": meta,
        "metatrait_direction": int(meta_direction),
        "unintended_metatrait": other,
        "rho_dose_vs_intended": signed_rho(intended_key, meta_direction),
        "rho_dose_vs_partner_only": (
            None
            if base["intended_partner_only"] is None
            else round(
                float(
                    spearman_rho(
                        xs, [meta_direction * r["intended_partner_only"] for r in rungs]
                    )
                    or 0.0
                ),
                4,
            )
        ),
        "rho_dose_vs_unintended": signed_rho("unintended", 1),
        "delta_intended": round(d_intended, 4),
        "delta_partner_only": None if d_partner is None else round(d_partner, 4),
        "delta_unintended": round(d_unintended, 4),
        "intended_sign_correct": bool(meta_direction * d_intended > 0),
        "partner_only_sign_correct": (
            None if d_partner is None else bool(meta_direction * d_partner > 0)
        ),
        "discriminant_ratio": (
            None
            if d_partner in (None, 0)
            else round(abs(d_unintended) / abs(d_partner), 3)
            if d_partner
            else None
        ),
        "monotone_fraction_intended": round(
            float(monotone_fraction([meta_direction * r[intended_key] for r in rungs]) or 0.0), 4
        ),
        "rungs": rungs,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sweep", help="opposite-prior summary JSON")
    p.add_argument("--out", default="results/big_two_metatraits.json")
    p.add_argument(
        "--full-grid",
        action="store_true",
        help="Use every usable rung instead of the dose-matched quiet band.",
    )
    args = p.parse_args(argv)

    from scripts.dose_matched_control import _score_pole

    blob = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    out_rows = []
    for row in blob["table"]:
        band = None if args.full_grid else _score_pole(row)["band_max_magnitude"]
        out_rows.append(analyse(row, band))

    usable = [r for r in out_rows if r["usable"]]
    n_partner_ok = sum(1 for r in usable if r["partner_only_sign_correct"])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "big_two_metatraits",
        "source": args.sweep,
        "band": "full grid" if args.full_grid else "dose-matched quiet band",
        "definition": {
            "alpha": "mean(C, A, 6-N)",
            "beta": "mean(E, O)",
            "partner_only": "same composite with the steered trait dropped",
        },
        "n_poles": len(out_rows),
        "n_usable": len(usable),
        "n_partner_only_sign_correct": n_partner_ok,
        "table": out_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 106)
    print(f"BIG TWO METATRAITS  ({report['band']})   alpha = mean(C, A, 6-N)   beta = mean(E, O)")
    print("=" * 106)
    print(
        f"{'trait':<18}{'pole':<6}{'meta':<7}{'dir':>4}{'ρ int':>8}{'ρ partner':>11}"
        f"{'ρ unint':>9}{'Δpartner':>10}{'Δunint':>9}{'disc':>7}"
    )
    for r in out_rows:
        if not r["usable"]:
            print(f"{r['trait']:<18}{r['pole']:<6}(only {r['n_rungs']} usable rungs)")
            continue
        disc = r["discriminant_ratio"]
        print(
            f"{r['trait']:<18}{r['pole']:<6}{r['intended_metatrait']:<7}"
            f"{r['metatrait_direction']:>+4}{(r['rho_dose_vs_intended'] or 0):>+8.2f}"
            f"{(r['rho_dose_vs_partner_only'] or 0):>+11.2f}"
            f"{(r['rho_dose_vs_unintended'] or 0):>+9.2f}"
            f"{(r['delta_partner_only'] or 0):>+10.3f}{(r['delta_unintended'] or 0):>+9.3f}"
            f"{('-' if disc is None else f'{disc:.2f}'):>7}"
        )
    print(
        f"\npartner-only composite moved the predicted way on "
        f"{n_partner_ok}/{len(usable)} usable poles"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
