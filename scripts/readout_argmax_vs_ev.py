"""Compare argmax and expected-value inventory readouts on identical activations.

Blas et al. (arXiv:2604.14463) score inventory items by greedy-decoding one
constrained option token, i.e. an argmax over "A".."E", and report that steering
produced "no salient patterns" on the inventory. Our sweeps record both
``target_argmax`` and ``target_ev`` at every rung, so the two readouts can be
compared on the same forward passes rather than across experiments.

The result on N-up: argmax takes two distinct values across nine doses and
correlates at +0.37, while the expected value over the option-token softmax gives
+0.98. An argmax cannot resolve movement in the option distribution that has not
crossed a decision boundary.

No GPU needed - pure reanalysis of committed JSON artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from app.persona.intensity_ladder import spearman_rho

SWEEP_DIRS = (Path("results/e1_inspan"), Path("results/gemma_final"))
OUT = Path("results/readout_argmax_vs_ev.json")

MIN_RUNGS = 3


def rows_for(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    curve = data.get("trait_curve") or {}
    return [r for r in curve.get("rows", []) if r.get("usable")]


def summarise(path: Path) -> dict | None:
    rows = rows_for(path)
    mags: Sequence[float] = [abs(r["magnitude"]) for r in rows]
    argmax = [r.get("target_argmax") for r in rows]
    ev = [r.get("target_ev") for r in rows]
    if len(rows) < MIN_RUNGS or any(v is None for v in argmax + ev):
        return {
            "sweep": path.stem.replace("validated_sweep_", ""),
            "dir": path.parent.name,
            "n_usable": len(rows),
            "note": "too few usable rungs to correlate",
        }
    return {
        "sweep": path.stem.replace("validated_sweep_", ""),
        "dir": path.parent.name,
        "n_usable": len(rows),
        "rho_argmax": round(spearman_rho(mags, argmax) or float("nan"), 4),
        "rho_ev": round(spearman_rho(mags, ev) or float("nan"), 4),
        "argmax_range": round(max(argmax) - min(argmax), 4),
        "ev_range": round(max(ev) - min(ev), 4),
        "distinct_argmax_values": len(set(argmax)),
    }


def main() -> None:
    out: list[dict] = []
    header = (
        f"{'sweep':<30}{'dir':<14}{'rho argmax':>11}{'rho EV':>9}"
        f"{'distinct argmax':>17}{'rungs':>7}"
    )
    print(header)
    for directory in SWEEP_DIRS:
        for path in sorted(directory.glob("validated_sweep_*.json")):
            rec = summarise(path)
            if rec is None:
                continue
            out.append(rec)
            if "note" in rec:
                print(f"{rec['sweep']:<30}{rec['dir']:<14}  {rec['note']} ({rec['n_usable']})")
            else:
                print(
                    f"{rec['sweep']:<30}{rec['dir']:<14}"
                    f"{rec['rho_argmax']:>+11.3f}{rec['rho_ev']:>+9.3f}"
                    f"{rec['distinct_argmax_values']:>17}{rec['n_usable']:>7}"
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"note": __doc__.strip(), "table": out}, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
