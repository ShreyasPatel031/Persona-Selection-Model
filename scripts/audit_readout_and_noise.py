"""Audit how much of each inventory result survives the standard argmax readout.

The prompting papers (Serapio-Garcia et al. 2025; Jiang et al. MPI/P2) and Blas
et al. (2604.14463) all score an item by the *committed* answer: argmax over the
Likert option tokens. That is also what a human respondent does. Our headline
numbers use expected value over the option-token distribution instead, which is a
more sensitive but non-standard readout.

This script asks, per sweep:

- how far the argmax score moves vs how far the EV score moves;
- how many of the 24 items per trait actually change their committed answer
  between the baseline rung and the best rung (the behavioural effect a human
  scorer would see);
- how the effect compares against the matched-norm random controls at the same
  doses, under *both* readouts, so the control margin is not inherited from the
  readout choice;
- whether any rung is repeated, i.e. whether an error bar exists at all.

No GPU needed - pure reanalysis of committed JSON artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

SWEEPS = (Path("results/e1_inspan"), Path("results/gemma_final"))
OUT = Path("results/readout_audit.json")


def usable_rows(data: dict) -> list[dict]:
    return [r for r in (data.get("trait_curve") or {}).get("rows", []) if r.get("usable")]


def control_curves(data: dict) -> list[list[dict]]:
    curves = data.get("control_curves") or data.get("controls") or []
    out = []
    for c in curves:
        rows = [r for r in (c.get("rows") or []) if r.get("usable")]
        if rows:
            out.append(rows)
    return out


def span(rows: list[dict], key: str) -> float | None:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return round(max(vals) - min(vals), 4) if len(vals) >= 2 else None


def item_flips(data: dict, rows: list[dict], trait: str) -> dict | None:
    """How many committed answers differ between the baseline and best rung."""
    base = next((r for r in rows if abs(r.get("magnitude", 0.0)) < 1e-9), None)
    if base is None or not rows:
        return None
    best = max(rows, key=lambda r: abs(r.get("magnitude", 0.0)))
    b_items = {
        (i["trait"], i["text"]): i.get("value")
        for i in (base.get("responses") or [])
        if i.get("trait") == trait
    }
    x_items = {
        (i["trait"], i["text"]): i.get("value")
        for i in (best.get("responses") or [])
        if i.get("trait") == trait
    }
    shared = set(b_items) & set(x_items)
    if not shared:
        return None
    flipped = [k for k in shared if b_items[k] != x_items[k]]
    return {
        "n_items": len(shared),
        "n_flipped": len(flipped),
        "best_magnitude": round(abs(best.get("magnitude", 0.0)), 1),
    }


def main() -> None:
    out = []
    for directory in SWEEPS:
        for path in sorted(directory.glob("validated_sweep_*.json")):
            data = json.loads(path.read_text())
            trait = data.get("trait")
            rows = usable_rows(data)
            mags = [round(abs(r.get("magnitude", 0.0)), 1) for r in rows]
            rec = {
                "sweep": path.stem.replace("validated_sweep_", ""),
                "grid": "in-span" if directory.name == "e1_inspan" else "ceiling",
                "n_usable_rungs": len(rows),
                "argmax_span": span(rows, "target_argmax"),
                "ev_span": span(rows, "target_ev"),
                "n_distinct_magnitudes": len(set(mags)),
                "n_repeated_magnitudes": len(mags) - len(set(mags)),
                "control_curves": len(control_curves(data)),
                "control_argmax_span": max(
                    (span(c, "target_argmax") or 0.0) for c in control_curves(data)
                )
                if control_curves(data)
                else None,
                "control_ev_span": max(
                    (span(c, "target_ev") or 0.0) for c in control_curves(data)
                )
                if control_curves(data)
                else None,
                "item_flips": item_flips(data, rows, trait),
            }
            out.append(rec)

    hdr = (
        f"{'sweep':<28}{'grid':<9}{'argmax Δ':>9}{'EV Δ':>7}"
        f"{'ctrl argmaxΔ':>13}{'ctrl EVΔ':>9}{'items flipped':>15}{'repeats':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        flips = r["item_flips"]
        flip_s = f"{flips['n_flipped']}/{flips['n_items']}" if flips else "n/a"
        print(
            f"{r['sweep']:<28}{r['grid']:<9}"
            f"{(r['argmax_span'] if r['argmax_span'] is not None else float('nan')):>9.3f}"
            f"{(r['ev_span'] if r['ev_span'] is not None else float('nan')):>7.3f}"
            f"{(r['control_argmax_span'] if r['control_argmax_span'] is not None else float('nan')):>13.3f}"
            f"{(r['control_ev_span'] if r['control_ev_span'] is not None else float('nan')):>9.3f}"
            f"{flip_s:>15}{r['n_repeated_magnitudes']:>8}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"note": __doc__.strip(), "table": out}, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
