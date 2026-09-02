#!/usr/bin/env python3
"""Re-score the opposite-prior sweeps against an on-target/off-target specificity control.

``scripts/dose_matched_control.py`` compares the trait direction against a
matched-norm *random* direction. That answers "is this particular direction
special", which is not the claim being made. The claim is "steering trait T moves
T", and the control that bears on it is the other four domains of the same
inventory administration: if pushing extraversion moves openness by +1.61 while
moving extraversion by +1.90, the readout is not carrying trait-specific
information no matter how quiet a random direction happens to be.

The two controls also cost different things. The random control needs its own
forward pass per rung per seed; the off-target control is free, because every
administration already scores all 120 items and the summary JSON keeps all five
domain EVs per rung in ``trait_rows[i]["ev_scores"]``.

Three statistics per rung:

``specificity_ratio``
    ``|on-target Δ| / mean(|off-target Δ|)``. 1.0 means the target moved no more
    than the average bystander domain.

``shared_drift``
    Mean *signed* Δ across all five domains. Detects the case where every domain
    slides the same way, which is degradation wearing a trait-shaped costume.

``past_midpoint``
    Signed distance of the target EV beyond 3.0 in the intended direction. This
    matters here because ``data/ipip_neo_120.csv`` is keying-balanced 12/12 in
    every domain, so *any* item-independent response distribution scores exactly
    3.0 (see the option-lock note in ``app/persona/inventory_ipip.py``). The
    opposite-prior design starts the target at an extreme, so a collapse toward
    the midpoint always looks like movement in the intended direction and is
    worth ``|3.0 - prior_ev|`` EV points for free. A negative ``past_midpoint``
    means the trait never left the interval that pure collapse would explain.

Two dose bands are reported. ``spec_band`` is rungs where the off-target movement
is small relative to the on-target movement; the ordering statistic in the verdict
is computed over it. It is defined using the trait's own numbers and so can flatter
the trait, which is the objection the dose-matched script raises against picking a
dose post hoc, and it is the price of the relative definition. ``offquiet_band``
is rungs where the mean off-target movement is below a fixed absolute threshold,
defined without reference to the target; it is reported as a diagnostic and is a
harsher bar than the reference prompt itself clears, because a real personality
prompt moves the bystander domains too (Big Two co-movement).

The shipped option-lock screen (``usable``) is computed on all 120 items, so a
total collapse of the 24 target items can pass it while the other 96 stay varied.
``target_locked`` re-applies the same thresholds to the target domain only.

Specificity alone is not sufficient and this script is not a replacement for the
random control: a ratio can be high at a dose where the whole readout is being
destroyed, if the bystander domains happen to move less than the target.
``max_control_abs_delta`` from the matched random direction is therefore carried
through to the chosen dose so the two controls can be read together.

No GPU, no re-running: pure reanalysis of committed sweep JSON.

    python3 scripts/specificity_control.py results/opposite_prior_ipip/summary_v3_calibrated_prompts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
MIDPOINT = 3.0

# Off-target movement below this is treated as "the bystander domains have not
# meaningfully moved", the specificity analogue of CONTROL_QUIET.
OFF_QUIET = 0.25
# A rung is in spec_band when the target outmoves the average bystander.
SPEC_BAND_MIN = 1.0

# Verdict thresholds. Rationale in the module docstring and in the --help text.
PASS_SPECIFICITY = 2.0  # same 2x bar the random control uses, applied off-target
PASS_ABS_DELTA = 0.5  # EV points; smaller than this is inside random-direction range
PASS_DRIFT_RATIO = 0.5  # |shared drift| this far above the on-target move is collapse
PASS_RHO = 0.8
MIN_BAND_RUNGS = 4

# Shipped option-lock thresholds, re-applied to the 24 target items.
MAX_TOP_OPTION_FRACTION = 0.90
MIN_OPTION_ENTROPY = 0.30

EPS = 1e-9


def _score_pole(row: dict) -> dict:
    from app.persona.intensity_ladder import monotone_fraction, spearman_rho

    trait = row["trait"]
    pole = row["pole"]
    sign = 1 if pole == "up" else -1
    rungs = row["trait_rows"]
    base = {t: float(rungs[0]["ev_scores"][t]) for t in TRAITS}
    others = [t for t in TRAITS if t != trait]

    controls = [[float(x) for x in c["ev_curve"]] for c in row["control_curves"]]

    per_rung: list[dict] = []
    for i, rr in enumerate(rungs):
        ev = {t: float(rr["ev_scores"][t]) for t in TRAITS}
        on = ev[trait] - base[trait]
        off = [ev[t] - base[t] for t in others]
        off_mean_abs = sum(abs(x) for x in off) / len(off)
        off_max_abs = max(abs(x) for x in off)
        ratio = None if off_mean_abs <= EPS else abs(on) / off_mean_abs
        tl = rr.get("target_lock") or {}
        top = tl.get("top_option_fraction")
        ent = tl.get("option_entropy")
        per_rung.append(
            {
                "magnitude": round(abs(float(rr["magnitude"])), 2),
                "on_target_delta": round(on, 4),
                "off_target_mean_abs_delta": round(off_mean_abs, 4),
                "off_target_max_abs_delta": round(off_max_abs, 4),
                "off_target_argmax": others[max(range(len(off)), key=lambda k: abs(off[k]))],
                "specificity_ratio": None if ratio is None else round(ratio, 3),
                "shared_drift": round((on + sum(off)) / len(TRAITS), 4),
                "target_ev": round(ev[trait], 4),
                "past_midpoint": round(sign * (ev[trait] - MIDPOINT), 4),
                "per_domain_delta": {t: round(ev[t] - base[t], 4) for t in TRAITS},
                "max_control_abs_delta": (
                    None
                    if not controls
                    else round(max(abs(c[i] - c[0]) for c in controls if i < len(c)), 4)
                ),
                "usable": bool(rr.get("usable")),
                "target_top_option_fraction": top,
                "target_option_entropy": ent,
                "target_locked": bool(
                    top is None
                    or ent is None
                    or top >= MAX_TOP_OPTION_FRACTION
                    or ent < MIN_OPTION_ENTROPY
                ),
                "sign_correct": sign * on > 0,
            }
        )

    live = [r for r in per_rung if r["magnitude"] > 0.0 and r["usable"]]
    spec_band = [
        r
        for r in live
        if r["specificity_ratio"] is not None and r["specificity_ratio"] >= SPEC_BAND_MIN
    ]
    offquiet_band = [r for r in live if r["off_target_mean_abs_delta"] <= OFF_QUIET]

    candidates = [r for r in live if r["sign_correct"] and r["specificity_ratio"] is not None]
    best = max(candidates, key=lambda r: r["specificity_ratio"]) if candidates else None

    def _rho(band: list[dict]) -> float | None:
        if len(band) < 2:
            return None
        rho = spearman_rho([r["magnitude"] for r in band], [sign * r["on_target_delta"] for r in band])
        return None if rho is None else round(float(rho), 4)

    def _mono(band: list[dict]) -> float | None:
        if len(band) < 2:
            return None
        mf = monotone_fraction([sign * r["on_target_delta"] for r in band])
        return None if mf is None else round(float(mf), 4)

    rho_spec = _rho(spec_band)
    rho_offquiet = _rho(offquiet_band)

    gap = row.get("prompt_gap")
    ref = row.get("reference_row") or {}
    ref_spec = None
    if ref.get("ev_scores"):
        ref_on = float(ref["ev_scores"][trait]) - base[trait]
        ref_off = [float(ref["ev_scores"][t]) - base[t] for t in others]
        ref_off_mean = sum(abs(x) for x in ref_off) / len(ref_off)
        ref_spec = None if ref_off_mean <= EPS else round(abs(ref_on) / ref_off_mean, 3)

    drift_ratio = (
        None
        if best is None or abs(best["on_target_delta"]) <= EPS
        else abs(best["shared_drift"]) / abs(best["on_target_delta"])
    )
    passes = bool(
        best is not None
        and best["specificity_ratio"] >= PASS_SPECIFICITY
        and abs(best["on_target_delta"]) >= PASS_ABS_DELTA
        and not best["target_locked"]
        and drift_ratio is not None
        and drift_ratio <= PASS_DRIFT_RATIO
        and rho_spec is not None
        and rho_spec >= PASS_RHO
        and len(spec_band) >= MIN_BAND_RUNGS
    )
    blockers = []
    if best is None:
        blockers.append("no sign-correct usable rung")
    else:
        if best["specificity_ratio"] < PASS_SPECIFICITY:
            blockers.append(f"specificity {best['specificity_ratio']:.2f}")
        if abs(best["on_target_delta"]) < PASS_ABS_DELTA:
            blockers.append(f"|Δ| {abs(best['on_target_delta']):.3f}")
        if best["target_locked"]:
            blockers.append("target readout locked")
        if drift_ratio is not None and drift_ratio > PASS_DRIFT_RATIO:
            blockers.append(f"drift/signal {drift_ratio:.2f}")
    if rho_spec is None or rho_spec < PASS_RHO:
        blockers.append(f"rho {rho_spec}")
    if len(spec_band) < MIN_BAND_RUNGS:
        blockers.append(f"band {len(spec_band)}")

    return {
        "pole": f"{trait[0].upper()}-{pole}",
        "trait": trait,
        "pole_direction": pole,
        "layer": row.get("layer"),
        "span": row.get("span"),
        "prior_ev": row.get("prior_ev"),
        "reference_ev": row.get("reference_ev"),
        "prompt_gap": gap,
        "midpoint_collapse_credit": round(MIDPOINT - float(row["prior_ev"]), 4),
        "midpoint_collapse_pct_of_gap": (
            None if not gap else round(100.0 * (MIDPOINT - float(row["prior_ev"])) / float(gap), 1)
        ),
        "reference_specificity_ratio": ref_spec,
        "n_spec_band_rungs": len(spec_band),
        "n_offquiet_band_rungs": len(offquiet_band),
        "offquiet_band_max_magnitude": (
            max(r["magnitude"] for r in offquiet_band) if offquiet_band else None
        ),
        "best_dose": None if best is None else best["magnitude"],
        "specificity_at_best": None if best is None else best["specificity_ratio"],
        "on_target_delta_at_best": None if best is None else best["on_target_delta"],
        "off_target_mean_abs_at_best": None if best is None else best["off_target_mean_abs_delta"],
        "off_target_max_abs_at_best": None if best is None else best["off_target_max_abs_delta"],
        "off_target_argmax_at_best": None if best is None else best["off_target_argmax"],
        "shared_drift_at_best": None if best is None else best["shared_drift"],
        "drift_vs_signal_at_best": (
            None
            if best is None or abs(best["on_target_delta"]) <= EPS
            else round(abs(best["shared_drift"]) / abs(best["on_target_delta"]), 3)
        ),
        "past_midpoint_at_best": None if best is None else best["past_midpoint"],
        "clears_midpoint_at_best": None if best is None else best["past_midpoint"] > 0.0,
        "target_locked_at_best": None if best is None else best["target_locked"],
        "target_option_entropy_at_best": None if best is None else best["target_option_entropy"],
        "max_control_abs_delta_at_best": None if best is None else best["max_control_abs_delta"],
        "pct_of_prompt_gap_at_best": (
            None
            if best is None or not gap
            else round(100.0 * best["on_target_delta"] / float(gap), 1)
        ),
        "rho_over_spec_band": rho_spec,
        "rho_over_offquiet_band": rho_offquiet,
        "monotone_fraction_over_offquiet_band": _mono(offquiet_band),
        "passes_specificity": passes,
        "blockers": blockers,
        "reported_works": row.get("works"),
        "per_rung": per_rung,
    }


def _print_comparison(dm_path: Path, sweep_name: str, scored: list[dict]) -> None:
    """Random-control verdict beside the specificity verdict, per pole."""
    blob = json.loads(dm_path.read_text(encoding="utf-8"))
    dm: dict[str, dict] = {}
    for sw in blob.get("sweeps", []):
        if sweep_name and sw.get("file") not in (None, sweep_name):
            continue
        for r in sw.get("table", []):
            dm[f"{r['trait'][0].upper()}-{r['pole']}"] = r
    if not dm:
        print(f"\n(no matching sweep in {dm_path}; skipping comparison)")
        return

    print("\nverdict comparison: matched random direction vs off-target specificity")
    head = (
        f"{'pole':<8}| {'dose':>7}{'traitΔ':>8}{'margin':>7}{'ρ':>6}{'bnd':>4} {'random':<7}"
        f"| {'dose':>7}{'onΔ':>8}{'spec':>7}{'ρ':>6}{'bnd':>4} {'specif':<7}| change"
    )
    print(head)
    print("-" * len(head))
    n_both = n_flip = 0
    for s in scored:
        r = dm.get(s["pole"])
        if r is None:
            continue
        a = bool(r.get("passes_dose_matched"))
        b = bool(s["passes_specificity"])
        n_both += 1 if (a and b) else 0
        n_flip += 1 if a != b else 0
        margin = r.get("dose_matched_margin")
        change = "same" if a == b else ("specificity rescues" if b else "specificity fails it")
        print(
            f"{s['pole']:<8}| {(r.get('best_dose') or 0):>7.0f}"
            f"{(r.get('trait_delta_at_best_dose') or 0):>+8.3f}"
            f"{(margin if isinstance(margin, (int, float)) else 0):>7.2f}"
            f"{(r.get('rho_over_band') or 0):>+6.2f}{r.get('n_band_rungs', 0):>4} "
            f"{('PASS' if a else 'fail'):<7}"
            f"| {(s['best_dose'] or 0):>7.0f}{(s['on_target_delta_at_best'] or 0):>+8.3f}"
            f"{(s['specificity_at_best'] or 0):>7.2f}"
            f"{(s['rho_over_spec_band'] if s['rho_over_spec_band'] is not None else 0):>+6.2f}"
            f"{s['n_spec_band_rungs']:>4} {('PASS' if b else 'fail'):<7}| {change}"
        )
    print(f"passes both criteria: {n_both}/{len(scored)}   verdict changes: {n_flip}/{len(scored)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sweeps", nargs="+", help="opposite-prior summary JSON files")
    p.add_argument("--out", default="results/specificity_control.json")
    p.add_argument(
        "--dose-matched",
        default=None,
        help="results/dose_matched_control.json; prints the verdict comparison when given",
    )
    args = p.parse_args(argv)

    out: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "specificity_control",
        "note": __doc__,
        "criterion": {
            "off_quiet_ev": OFF_QUIET,
            "spec_band_min": SPEC_BAND_MIN,
            "specificity": PASS_SPECIFICITY,
            "min_abs_on_target_delta": PASS_ABS_DELTA,
            "max_drift_over_signal": PASS_DRIFT_RATIO,
            "rho": PASS_RHO,
            "rho_band": "spec_band",
            "min_band_rungs": MIN_BAND_RUNGS,
            "target_option_lock": {
                "max_top_option_fraction": MAX_TOP_OPTION_FRACTION,
                "min_option_entropy": MIN_OPTION_ENTROPY,
            },
        },
        "sweeps": [],
    }

    for path in args.sweeps:
        src = Path(path)
        blob = json.loads(src.read_text(encoding="utf-8"))
        table = blob.get("table") or []
        if not table or "trait_rows" not in table[0]:
            continue
        scored = [_score_pole(r) for r in table]
        out["sweeps"].append(
            {
                "file": src.name,
                "layers": blob.get("layers"),
                "n_poles": len(scored),
                "n_passing_specificity": sum(1 for s in scored if s["passes_specificity"]),
                "n_passing_as_reported": sum(1 for s in scored if s["reported_works"]),
                "table": scored,
            }
        )

        print("\n" + "=" * 126)
        print(f"{src.name}   on-target vs off-target specificity")
        print(
            "best dose = max specificity ratio among sign-correct usable rungs; "
            "rho over spec_band"
        )
        print("=" * 126)
        head = (
            f"{'pole':<8}{'dose':>8}{'onΔ':>8}{'offμ':>7}{'offmax':>8}{'worst':>6}"
            f"{'spec':>7}{'refsp':>6}{'drift':>7}{'d/s':>6}{'pastmid':>8}{'tgtent':>7}"
            f"{'rndΔ':>7}{'ρ':>7}{'band':>5}{'oq':>4}{'%gap':>7}  verdict"
        )
        print(head)
        print("-" * len(head))
        for s in scored:
            print(
                f"{s['pole']:<8}{(s['best_dose'] or 0):>8.0f}"
                f"{(s['on_target_delta_at_best'] or 0):>+8.3f}"
                f"{(s['off_target_mean_abs_at_best'] or 0):>7.3f}"
                f"{(s['off_target_max_abs_at_best'] or 0):>8.3f}"
                f"{(s['off_target_argmax_at_best'] or '-')[:5]:>6}"
                f"{(s['specificity_at_best'] or 0):>7.2f}"
                f"{(s['reference_specificity_ratio'] or 0):>6.2f}"
                f"{(s['shared_drift_at_best'] or 0):>+7.3f}"
                f"{(s['drift_vs_signal_at_best'] or 0):>6.2f}"
                f"{(s['past_midpoint_at_best'] or 0):>+8.3f}"
                f"{(s['target_option_entropy_at_best'] or 0):>7.3f}"
                f"{(s['max_control_abs_delta_at_best'] or 0):>7.3f}"
                f"{(s['rho_over_spec_band'] if s['rho_over_spec_band'] is not None else 0):>+7.2f}"
                f"{s['n_spec_band_rungs']:>5}{s['n_offquiet_band_rungs']:>4}"
                f"{(s['pct_of_prompt_gap_at_best'] or 0):>7.1f}"
                f"  {'PASS' if s['passes_specificity'] else 'fail'}"
                f"{'' if s['passes_specificity'] else '  ' + '; '.join(s['blockers'])}"
            )
        n = out["sweeps"][-1]["n_passing_specificity"]
        print(f"passing specificity: {n}/{len(scored)}")

        print("\nmidpoint-collapse budget (inventory keying is 12/12 per domain, so an")
        print("item-independent readout scores exactly 3.0 and the opposite-prior design")
        print("hands every pole |3.0 - prior| EV points in the 'right' direction for free)")
        mh = (
            f"{'pole':<8}{'prior':>7}{'ref':>7}{'gap':>7}{'collapse':>10}{'%gap':>6}"
            f"{'onΔ@best':>10}{'pastmid':>9}  beyond-collapse"
        )
        print(mh)
        print("-" * len(mh))
        for s in scored:
            print(
                f"{s['pole']:<8}{s['prior_ev']:>7.2f}{s['reference_ev']:>7.2f}"
                f"{s['prompt_gap']:>+7.2f}{s['midpoint_collapse_credit']:>+10.2f}"
                f"{(s['midpoint_collapse_pct_of_gap'] or 0):>6.0f}"
                f"{(s['on_target_delta_at_best'] or 0):>+10.3f}"
                f"{(s['past_midpoint_at_best'] or 0):>+9.3f}"
                f"  {'yes' if s['clears_midpoint_at_best'] else 'no'}"
            )
        n_clear = sum(1 for s in scored if s["clears_midpoint_at_best"])
        print(f"poles whose best-specificity dose clears the midpoint: {n_clear}/{len(scored)}")

        if args.dose_matched:
            _print_comparison(Path(args.dose_matched), src.name, scored)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
