"""Reanalyze gemma_final IPIP sweeps restricted to the behaviorally proven dose span.

The bipolar judge runs (results/bipolar, results/bipolar_afix) achieved 10/10
sign-correct monotone control using doses bounded by each trait's ladder span
(E 741, A 819, C 988, N 1787, O 1529). The IPIP validated sweeps instead ran
their grids up to coherence ceilings, 3-4x past those doses. This script
recomputes the IPIP dose-response Spearman using only the rungs inside the
behavioral span, i.e. what the correlation would have been had the inventory
sweeps used the same dosing the judge sweeps did.

No GPU needed - pure reanalysis of committed JSON artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# direction_span_magnitude at the proven steering layers (bipolar summaries).
BEHAVIORAL_SPAN = {
    "extraversion": 741.36,  # L15
    "agreeableness": 819.30,  # L15 (bipolar_afix)
    "conscientiousness": 988.09,  # L17
    "neuroticism": 1786.56,  # L20
    "openness": 1528.69,  # L19
}

SWEEPS_DIR = Path("results/gemma_final")
OUT = Path("results/gemma_final/inspan_reanalysis.json")


def spearman(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 3 or xa.std() < 1e-12 or ya.std() < 1e-12:
        return float("nan")
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    rows_out = []
    print(f"{'trait':<18}{'pole':<6}{'full rho':>9}{'in-span rho':>12}{'rungs':>7}  in-span EVs")
    for trait, span in BEHAVIORAL_SPAN.items():
        for pole in ("high", "low"):
            path = SWEEPS_DIR / f"validated_sweep_{trait}_pc1_{pole}.json"
            data = json.loads(path.read_text())
            rows = data["trait_curve"]["rows"]
            alphas = [r["alpha"] for r in rows]
            mags = [abs(r["magnitude"]) for r in rows]
            evs = [r["target_ev"] for r in rows]
            keep = [i for i, m in enumerate(mags) if m <= span * 1.05]
            full = spearman(alphas, evs)
            inspan = (
                spearman([alphas[i] for i in keep], [evs[i] for i in keep])
                if len(keep) >= 3
                else float("nan")
            )
            rec = {
                "trait": trait,
                "pole": pole,
                "behavioral_span": span,
                "full_rho": round(full, 4),
                "inspan_rho": round(inspan, 4),
                "n_inspan_rungs": len(keep),
                "inspan_magnitudes": [round(mags[i], 1) for i in keep],
                "inspan_evs": [round(evs[i], 4) for i in keep],
                "dropped_magnitudes": [round(m, 1) for m in mags if m > span * 1.05],
            }
            rows_out.append(rec)
            print(
                f"{trait:<18}{pole:<6}{full:>+9.2f}{inspan:>+12.2f}{len(keep):>7}"
                f"  {[round(evs[i], 2) for i in keep]}"
            )
    OUT.write_text(json.dumps({"note": __doc__.strip(), "table": rows_out}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
