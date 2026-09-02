"""The inventory result under the STANDARD readout, per pole-direction.

Blas et al., Serapio-Garcia et al. and Jiang et al. all score an item by its
committed answer: argmax over the Likert option tokens. That is also what a human
respondent does. This script reports what our sweeps look like under that readout
alone, so the claim does not rest on our expected-value scoring.

Three quantities are kept separate, because conflating them is how the earlier
write-up went wrong:

- ``rho``    - does the score move in rank order with dose (ordering reliability)
- ``delta``  - how many Likert points the score actually moves (effect size)
- ``ctrl``   - how far the score moves under a matched-norm *random* direction on
               the same grid, measured the same way (max-min span), which is the
               only noise estimate available given that no rung is repeated

A pole only counts as supported if the ordering is sign-correct AND the trait
movement clearly exceeds the random-direction movement.

Where a pole has both a ceiling-dosed and an in-span run, the in-span run wins.

Run: PYTHONPATH=. python3 scripts/argmax_dose_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.persona.intensity_ladder import spearman_rho

SWEEP_DIRS = (Path("results/gemma_final"), Path("results/e1_inspan"))
OUT = Path("results/argmax_dose_response.json")

MIN_RUNGS = 3
CLEAR_MARGIN = 2.0
POLE_ORDER = (
    "E-up", "E-down", "A-up", "A-down", "C-up",
    "C-down", "N-up", "N-down", "O-up", "O-down",
)


def shipped_screen(lock: dict) -> bool:
    return lock["top_option_fraction"] < 0.90 and lock["option_entropy"] >= 0.30


def span(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def curve_stats(rows: list[dict], key: str, sign: int) -> dict:
    keep = [r for r in rows if r.get(key) is not None and shipped_screen(r["lock"])]
    if len(keep) < MIN_RUNGS:
        return {"n_rungs": len(keep), "rho": None, "delta": None, "span": None}
    xs = [abs(r["alpha"]) for r in keep]
    ys = [float(r[key]) for r in keep]
    base = next((float(r[key]) for r in keep if abs(r["magnitude"]) < 1e-9), ys[0])
    extreme = min(ys) if sign < 0 else max(ys)
    rho = spearman_rho(xs, ys)
    return {
        "n_rungs": len(keep),
        "rho": round(rho, 4) if rho is not None else None,
        "delta": round(extreme - base, 4),
        "span": round(max(ys) - min(ys), 4),
    }


def main() -> None:
    poles: dict[str, dict] = {}
    for directory in SWEEP_DIRS:
        for path in sorted(directory.glob("validated_sweep_*.json")):
            rep = json.loads(path.read_text())
            trait = rep["trait"]
            toward = rep["verdict"]["steered_toward"]
            sign = -1 if toward == "low" else 1
            pole = f"{trait[0].upper()}-{'up' if sign > 0 else 'down'}"
            rows = rep["trait_curve"]["rows"]

            trait_am = curve_stats(rows, "target_argmax", sign)
            ctrl_spans = [
                span([r for r in c["rows"] if shipped_screen(r["lock"])], "target_argmax")
                for c in (rep.get("control_curves") or [])
            ]
            ctrl_spans = [s for s in ctrl_spans if s is not None]
            ctrl = round(max(ctrl_spans), 4) if ctrl_spans else None

            sign_ok = (
                None if trait_am["rho"] is None
                else (trait_am["rho"] > 0 if sign > 0 else trait_am["rho"] < 0)
            )
            ratio = (
                round(trait_am["span"] / ctrl, 2)
                if trait_am["span"] is not None and ctrl not in (None, 0)
                else None
            )
            poles[pole] = {
                "pole": pole,
                "grid": "in-span" if directory.name == "e1_inspan" else "ceiling",
                "argmax": trait_am,
                "expected_value": curve_stats(rows, "target_ev", sign),
                "control_argmax_span": ctrl,
                "n_random_controls": len(rep.get("control_curves") or []),
                "sign_correct": sign_ok,
                "trait_over_control": ratio,
                "supported": bool(sign_ok and ratio is not None and ratio >= CLEAR_MARGIN),
            }

    hdr = (
        f"{'pole':<8}{'grid':<9}{'rho':>7}{'delta':>8}{'span':>7}"
        f"{'ctrl span':>10}{'ratio':>7}{'rungs':>7}{'verdict':>12}"
    )
    print("Inventory dose-response under the standard argmax readout")
    print("delta/span in Likert points on a 1-5 scale; ctrl = matched-norm random direction\n")
    print(hdr)
    print("-" * len(hdr))
    for name in POLE_ORDER:
        p = poles.get(name)
        if p is None:
            continue
        a = p["argmax"]
        if a["rho"] is None:
            print(f"{name:<8}{p['grid']:<9}{'not measurable (' + str(a['n_rungs']) + ' clean rungs)':>58}")
            continue
        verdict = "supported" if p["supported"] else ("wrong sign" if not p["sign_correct"] else "not vs ctrl")
        print(
            f"{name:<8}{p['grid']:<9}{a['rho']:>+7.2f}{a['delta']:>+8.3f}{a['span']:>7.3f}"
            f"{(p['control_argmax_span'] if p['control_argmax_span'] is not None else float('nan')):>10.3f}"
            f"{(p['trait_over_control'] if p['trait_over_control'] is not None else float('nan')):>7.2f}"
            f"{a['n_rungs']:>7}{verdict:>12}"
        )

    measurable = [p for p in poles.values() if p["argmax"]["rho"] is not None]
    print(
        f"\nargmax: {sum(1 for p in measurable if p['sign_correct'])}/{len(measurable)}"
        f" sign-correct of {len(POLE_ORDER)} poles;"
        f" {sum(1 for p in poles.values() if p['supported'])} also clear the {CLEAR_MARGIN}x"
        f" random-control margin"
    )
    print(f"random control draws per sweep: {sorted({p['n_random_controls'] for p in poles.values()})}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"note": __doc__.strip(), "table": [poles[n] for n in POLE_ORDER if n in poles]}, indent=2)
        + "\n"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
