#!/usr/bin/env python3
"""E7 - Big Two (Digman alpha/beta) covariance on our committed inventory sweeps.

Blas et al. Sec 5.3 report that 46.15% of cross-trait sign patterns match the
Digman metatraits and that "no LLM satisfied all Big Two correlations" - computed on
their SJT/classifier scores, not on inventory scores. This script runs the inventory
version on our sweeps.

Metatrait structure:
    alpha / Stability   C+  A+  N-
    beta  / Plasticity  E+  O+

For a sweep steering one trait toward one pole, every *other* trait either has a
predicted co-movement sign (same metatrait) or no prediction (other metatrait).
Two statistics are reported:

``sign_match_rate``
    Of the partner traits with a prediction, how many actually moved that way. This
    is the analogue of their 46.15%.

``shared_drift``
    Mean signed movement across all five traits. The reason this must be reported
    next to the match rate: if a large perturbation degrades the respondent, every
    trait slides toward the same option and the sign pattern can look like alpha
    structure while carrying no trait information. A match rate is only interpretable
    when shared drift is small relative to the on-target movement.

Only rungs passing the shipped option-lock screen are used, and both readouts are
computed so the result does not depend on the readout choice.

No GPU: pure reanalysis of committed JSON.

    PYTHONPATH=. python3 scripts/big_two_covariance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.persona.intensity_ladder import spearman_rho

SWEEP_DIRS = (Path("results/gemma_final"), Path("results/e1_inspan"))
OUT = Path("results/big_two_covariance.json")

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
# metatrait -> {trait: loading sign}
ALPHA = {"conscientiousness": +1, "agreeableness": +1, "neuroticism": -1}
BETA = {"extraversion": +1, "openness": +1}


def metatrait_of(trait: str) -> tuple[str, int] | None:
    if trait in ALPHA:
        return "alpha", ALPHA[trait]
    if trait in BETA:
        return "beta", BETA[trait]
    return None


def shipped_screen(lock: dict) -> bool:
    return lock["top_option_fraction"] < 0.90 and lock["option_entropy"] >= 0.30


def analyze_sweep(path: Path, readout: str) -> dict | None:
    key = "argmax_scores" if readout == "argmax" else "ev_scores"
    rep = json.loads(path.read_text())
    rows = [r for r in rep["trait_curve"]["rows"] if shipped_screen(r["lock"])]
    return _analyze_rows(
        rep["trait"],
        rep["verdict"]["steered_toward"],
        rows,
        key,
        "in-span" if path.parent.name == "e1_inspan" else "ceiling",
    )


def analyze_opposite_prior(row: dict, readout: str) -> dict | None:
    """Big Two test on an opposite-prior pole, inside the dose-matched band.

    Restricting to the band where the matched random direction is still quiet is
    not optional here. Past that dose every trait slides toward the same option,
    which can imitate alpha structure while carrying no trait information — the
    failure mode ``shared_drift`` exists to catch. Reading the sign pattern only
    where the control has not moved removes the confound rather than reporting it.
    """
    from scripts.dose_matched_control import _score_pole

    key = "argmax_scores" if readout == "argmax" else "ev_scores"
    if key == "argmax_scores":
        return None  # opposite-prior sweeps record the EV readout only
    band_max = _score_pole(row)["band_max_magnitude"]
    if band_max is None:
        return None
    rows = [
        r
        for r in row["trait_rows"]
        if shipped_screen(r["lock"]) and abs(float(r["magnitude"])) <= band_max + 1e-9
    ]
    return _analyze_rows(
        row["trait"],
        "high" if row["pole"] == "up" else "low",
        rows,
        "ev_scores",
        f"dose-matched band (<= {band_max:g})",
    )


def _analyze_rows(
    trait: str,
    toward: str,
    rows: list[dict],
    key: str,
    grid_label: str,
) -> dict | None:
    if len(rows) < 3:
        return None
    base = next((r for r in rows if abs(float(r["magnitude"])) < 1e-9), None)
    if base is None:
        return None

    # Intended direction of the target trait.
    target_sign = 1 if toward == "high" else -1
    # Rung whose target moved furthest in the intended direction.
    def target_val(r: dict) -> float:
        return float(r[key][trait])

    best = max(rows, key=lambda r: target_sign * (target_val(r) - target_val(base)))
    on_target = target_val(best) - target_val(base)
    if target_sign * on_target <= 0:
        return None  # target did not move the intended way; sign test is meaningless

    deltas = {t: float(best[key][t]) - float(base[key][t]) for t in TRAITS}
    shared_drift = sum(deltas.values()) / len(deltas)

    mt = metatrait_of(trait)
    if mt is None:
        return None
    meta, target_load = mt
    # Direction the metatrait itself was pushed.
    meta_direction = target_sign * target_load

    partners: dict[str, dict] = {}
    matches = 0
    predicted = 0
    for other in TRAITS:
        if other == trait:
            continue
        omt = metatrait_of(other)
        if omt is None or omt[0] != meta:
            partners[other] = {"predicted_sign": None, "delta": round(deltas[other], 4)}
            continue
        pred = meta_direction * omt[1]
        actual = deltas[other]
        ok = (actual > 0) if pred > 0 else (actual < 0)
        predicted += 1
        matches += 1 if ok else 0
        partners[other] = {
            "predicted_sign": int(pred),
            "delta": round(actual, 4),
            "match": bool(ok),
        }

    # Cross-trait covariance along the dose path, not just at the best rung.
    doses = [abs(float(r["magnitude"])) for r in rows]
    cross_rho = {}
    for other in TRAITS:
        if other == trait:
            continue
        ys = [float(r[key][other]) for r in rows]
        rho = spearman_rho(doses, ys)
        cross_rho[other] = None if rho is None else round(rho, 3)

    return {
        "pole": f"{trait[0].upper()}-{'up' if target_sign > 0 else 'down'}",
        "trait": trait,
        "toward": toward,
        "metatrait": meta,
        "metatrait_direction": int(meta_direction),
        "grid": grid_label,
        "n_usable_rungs": len(rows),
        "best_magnitude": best["magnitude"],
        "on_target_delta": round(on_target, 4),
        "shared_drift": round(shared_drift, 4),
        "drift_vs_signal": (
            None if on_target == 0 else round(abs(shared_drift) / abs(on_target), 3)
        ),
        "partners": partners,
        "n_predicted": predicted,
        "n_matched": matches,
        "rho_dose_vs_offtarget": cross_rho,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--opposite-prior",
        default="",
        help=(
            "Run on an opposite-prior summary instead of the validated_sweep_* "
            "artifacts, restricted to the dose-matched band."
        ),
    )
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    report: dict = {"note": __doc__, "readouts": {}}
    if args.opposite_prior:
        report["source"] = args.opposite_prior

    for readout in ("argmax", "ev"):
        seen: dict[str, dict] = {}
        if args.opposite_prior:
            blob = json.loads(Path(args.opposite_prior).read_text(encoding="utf-8"))
            for row in blob["table"]:
                r = analyze_opposite_prior(row, readout)
                if r is not None:
                    seen[r["pole"]] = r
        else:
            for d in SWEEP_DIRS:
                for p in sorted(d.glob("validated_sweep_*.json")):
                    r = analyze_sweep(p, readout)
                    if r is None:
                        continue
                    # in-span run wins where a pole has both
                    if r["pole"] in seen and seen[r["pole"]]["grid"] == "in-span":
                        continue
                    seen[r["pole"]] = r
        rows = list(seen.values())
        tot_pred = sum(r["n_predicted"] for r in rows)
        tot_match = sum(r["n_matched"] for r in rows)
        clean = [r for r in rows if r["drift_vs_signal"] is not None and r["drift_vs_signal"] < 0.5]
        clean_pred = sum(r["n_predicted"] for r in clean)
        clean_match = sum(r["n_matched"] for r in clean)
        report["readouts"][readout] = {
            "n_sweeps_testable": len(rows),
            "n_predicted_pairs": tot_pred,
            "n_matched_pairs": tot_match,
            "sign_match_rate": None if tot_pred == 0 else round(tot_match / tot_pred, 4),
            "n_sweeps_low_drift": len(clean),
            "sign_match_rate_low_drift_only": (
                None if clean_pred == 0 else round(clean_match / clean_pred, 4)
            ),
            "their_reported_rate_on_sjts": 0.4615,
            "sweeps": rows,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for readout in ("argmax", "ev"):
        R = report["readouts"][readout]
        print(f"\n=== Big Two sign-pattern match, {readout} readout ===")
        print(
            f"  testable sweeps {R['n_sweeps_testable']}  "
            f"predicted pairs {R['n_predicted_pairs']}  matched {R['n_matched_pairs']}  "
            f"rate {R['sign_match_rate']}   (theirs on SJTs: 0.4615)"
        )
        print(
            f"  low-drift subset: {R['n_sweeps_low_drift']} sweeps, "
            f"rate {R['sign_match_rate_low_drift_only']}"
        )
        hdr = f"  {'pole':<8}{'meta':<7}{'onΔ':>8}{'drift':>8}{'d/s':>6}{'matched':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in R["sweeps"]:
            print(
                f"  {r['pole']:<8}{r['metatrait']:<7}{r['on_target_delta']:>8}"
                f"{r['shared_drift']:>8}{str(r['drift_vs_signal']):>6}"
                f"{r['n_matched']}/{r['n_predicted']:<8}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
