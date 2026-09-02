#!/usr/bin/env python3
"""Big Two (Digman alpha/beta) covariance on the final-cycle MPI-120 sweeps.

Blas et al. (arXiv:2604.14463, Sec 5.3) report that **46.15%** of cross-trait sign
patterns match the Digman metatraits and that "no LLM satisfied all Big Two
correlations" — computed on their SJT / classifier-scored text, because they had
dropped the inventory as patternless. This runs their test on inventory scores.

``scripts/big_two_covariance.py`` ran the same logic on the older sweeps
(homemade form, mixed ceiling/in-span grids). This script re-runs it on the
final-cycle artifacts: the published MPI-120, all 120 items scored at every rung,
opposite-prior baselines, one layer, EV readout.

Metatrait structure (Digman 1997; DeYoung 2002):

    alpha / Stability    C+  A+  N-
    beta  / Plasticity   E+  O+

Steering one trait toward one pole implies a direction for the *metatrait*, hence
a predicted sign for each same-metatrait partner. Cross-metatrait traits get no
prediction and are reported as unconstrained.

``sign_match_rate``
    Of partner traits with a prediction, the fraction that moved that way. Direct
    analogue of their 46.15%. Chance is 50%.

``shared_drift``
    Mean *signed* movement across all five domains, reported next to every match
    rate. If a large injection degrades the respondent, all five domains slide the
    same way and the sign pattern can imitate alpha structure while carrying no
    trait information. A match rate is only interpretable when drift is small
    relative to on-target movement, so the low-drift subset (|drift|/|on-target|
    < 0.5) is reported separately and is the number to quote.

No GPU: pure reanalysis of committed JSON.

    PYTHONPATH=. python3 scripts/big_two_final_cycle.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

from app.persona.intensity_ladder import spearman_rho

SWEEPS = Path("results/final_cycle/phase4_sweeps_specificity.json")
OUT = Path("results/final_cycle/big_two_covariance.json")
THEIR_RATE = 0.4615

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
# metatrait -> {trait: loading sign}. Kept in step with scripts/big_two_covariance.py,
# which runs the same test on the older sweeps; a sibling import is not usable here
# because an installed site-packages `scripts` package shadows this directory.
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


def binom_p_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value against a fair coin."""
    if n == 0:
        return 1.0
    probs = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    return round(min(1.0, sum(q for q in probs if q <= probs[k] + 1e-12)), 5)


def analyze(rec: dict) -> dict | None:
    trait, pole = rec["trait"], rec["pole"]
    mt = metatrait_of(trait)
    if mt is None:
        return None
    meta, target_load = mt

    rungs = [r for r in rec["rungs"] if r["usable"] and shipped_screen(r["lock"])]
    if len(rungs) < 3:
        return None
    base = next((r for r in rungs if abs(float(r["magnitude"])) < 1e-9), None)
    if base is None:
        return None

    target_sign = 1 if pole == "up" else -1
    best = max(rungs, key=lambda r: target_sign * (r["ev_scores"][trait] - base["ev_scores"][trait]))
    on_target = best["ev_scores"][trait] - base["ev_scores"][trait]
    if target_sign * on_target <= 0:
        return None  # target never moved the intended way; the sign test is vacuous

    deltas = {t: best["ev_scores"][t] - base["ev_scores"][t] for t in TRAITS}
    drift = sum(deltas.values()) / len(deltas)
    meta_direction = target_sign * target_load

    partners: dict[str, dict] = {}
    predicted = matched = 0
    for other in TRAITS:
        if other == trait:
            continue
        omt = metatrait_of(other)
        if omt is None or omt[0] != meta:
            partners[other] = {"predicted_sign": None, "delta": round(deltas[other], 4)}
            continue
        pred = meta_direction * omt[1]
        ok = deltas[other] > 0 if pred > 0 else deltas[other] < 0
        predicted += 1
        matched += 1 if ok else 0
        partners[other] = {
            "predicted_sign": int(pred),
            "delta": round(deltas[other], 4),
            "match": bool(ok),
        }

    # Drift-discordant pairs: the metatrait demands this partner move *opposite* to
    # the direction all five domains are sliding. A global degradation slide cannot
    # produce these, so they are the confound-proof subset of the sign test.
    drift_sign = 1 if drift > 0 else -1
    discordant = {
        name: p
        for name, p in partners.items()
        if p["predicted_sign"] is not None and p["predicted_sign"] == -drift_sign
    }

    doses = [abs(float(r["magnitude"])) for r in rungs]
    cross_rho = {}
    for other in TRAITS:
        if other == trait:
            continue
        rho = spearman_rho(doses, [r["ev_scores"][other] for r in rungs])
        cross_rho[other] = None if rho is None else round(rho, 3)

    return {
        "pole": f"{trait[0].upper()}-{pole}",
        "trait": trait,
        "toward": "high" if target_sign > 0 else "low",
        "metatrait": meta,
        "metatrait_direction": int(meta_direction),
        "layer": rec["layer"],
        "n_usable_rungs": len(rungs),
        "best_magnitude": best["magnitude"],
        "on_target_delta": round(on_target, 4),
        "shared_drift": round(drift, 4),
        "drift_vs_signal": round(abs(drift) / abs(on_target), 3),
        "partners": partners,
        "n_predicted": predicted,
        "n_matched": matched,
        "n_discordant": len(discordant),
        "n_discordant_matched": sum(1 for p in discordant.values() if p["match"]),
        "discordant_partners": sorted(discordant),
        "rho_dose_vs_offtarget": cross_rho,
    }


def main() -> None:
    table = json.loads(SWEEPS.read_text())["table"]
    rows = [r for r in (analyze(rec) for rec in table) if r is not None]
    rows.sort(key=lambda r: r["pole"])

    def tally(rs: list[dict]) -> tuple[int, int]:
        return sum(r["n_matched"] for r in rs), sum(r["n_predicted"] for r in rs)

    m_all, p_all = tally(rows)
    clean = [r for r in rows if r["drift_vs_signal"] < 0.5]
    m_lo, p_lo = tally(clean)
    d_all = sum(r["n_discordant"] for r in rows)
    d_hit = sum(r["n_discordant_matched"] for r in rows)

    report = {
        "note": __doc__,
        "source": str(SWEEPS),
        "instrument": "MPI-120 (published Johnson IPIP-NEO-120, second person)",
        "readout": "ev",
        "baseline": "opposite-prior",
        "metatraits": {"alpha_stability": "C+ A+ N-", "beta_plasticity": "E+ O+"},
        "their_reported_rate_on_sjts": THEIR_RATE,
        "n_sweeps_testable": len(rows),
        "n_predicted_pairs": p_all,
        "n_matched_pairs": m_all,
        "sign_match_rate": None if p_all == 0 else round(m_all / p_all, 4),
        "binomial_p_vs_chance": binom_p_two_sided(m_all, p_all),
        "n_sweeps_low_drift": len(clean),
        "n_predicted_pairs_low_drift": p_lo,
        "sign_match_rate_low_drift_only": None if p_lo == 0 else round(m_lo / p_lo, 4),
        "binomial_p_vs_chance_low_drift": binom_p_two_sided(m_lo, p_lo),
        "n_discordant_pairs": d_all,
        "n_discordant_matched": d_hit,
        "discordant_match_rate": None if d_all == 0 else round(d_hit / d_all, 4),
        "sweeps": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== Big Two (Digman alpha/beta) sign match — MPI-120, EV readout ===")
    print(
        f"  all sweeps:  {m_all}/{p_all} pairs = {report['sign_match_rate']}"
        f"   p={report['binomial_p_vs_chance']}   (theirs on SJTs: {THEIR_RATE})"
    )
    print(
        f"  low drift:   {m_lo}/{p_lo} pairs = {report['sign_match_rate_low_drift_only']}"
        f"   p={report['binomial_p_vs_chance_low_drift']}   ({len(clean)} sweeps)"
    )
    print(
        f"  drift-proof: {d_hit}/{d_all} pairs where the predicted sign opposes the"
        " global slide"
    )
    hdr = f"  {'pole':<9}{'meta':<7}{'onΔ':>8}{'drift':>8}{'d/s':>7}{'matched':>9}{'drift-proof':>13}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        dp = f"{r['n_discordant_matched']}/{r['n_discordant']}" if r["n_discordant"] else "—"
        print(
            f"  {r['pole']:<9}{r['metatrait']:<7}{r['on_target_delta']:>+8.2f}"
            f"{r['shared_drift']:>+8.2f}{r['drift_vs_signal']:>7.2f}"
            f"{r['n_matched']:>6}/{r['n_predicted']}{dp:>13}"
        )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
