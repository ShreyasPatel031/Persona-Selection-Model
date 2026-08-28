"""How different is our ladder PC1 from the two-arm mean-difference contrast?

Blas et al. and most CAA work build the steering vector as a *difference of two
centroids* (construct minus antithesis). We build it as PC1 across nine graded
prompt-ladder centroids. We have never run a head-to-head steering comparison of
the two, so the strongest available evidence is geometric: at the layers we
actually steer, how aligned is ``v_endpoint`` (level 9 minus level 1, i.e. the
two-arm contrast) with ``v_pc1``, and does the ladder order itself along each?

If the cosine is near 1 the two vectors are nearly the same and our "different
vector" difference is cosmetic. If it is low, the two are genuinely different
directions and the head-to-head sweep is worth running.

Run: PYTHONPATH=. python3 scripts/endpoint_vs_pc1_geometry.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Layer actually used per trait in the sweeps we report.
STEERED_LAYER = {
    "extraversion": 15,
    "agreeableness": 15,
    "conscientiousness": 15,
    "neuroticism": 20,
    "openness": 15,
}

GEOM_DIR = Path("results/gemma_ocean_v2")
OUT = Path("results/endpoint_vs_pc1_geometry.json")


def main() -> None:
    out = []
    hdr = (
        f"{'trait':<19}{'layer':>6}{'cos(endpoint,pc1)':>19}{'pc1 var share':>15}"
        f"{'rho lvl~pc1':>13}{'rho lvl~endpt':>15}{'mono pc1':>10}"
    )
    print("Two-arm contrast vs ladder PC1, at the layer we actually steer\n")
    print(hdr)
    print("-" * len(hdr))
    for trait, layer in STEERED_LAYER.items():
        path = GEOM_DIR / f"ladder_geometry_{trait}.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text())
        rows = rep["geometry"]["per_layer"]
        agree = {a["layer"]: a for a in rep.get("direction_agreement", [])}
        if layer >= len(rows):
            continue
        r = rows[layer]
        rec = {
            "trait": trait,
            "layer": layer,
            "cos_endpoint_pc1": r.get("cos_endpoint_pc1"),
            "pc1_variance_ratio": r.get("pc1_variance_ratio"),
            "spearman_level_vs_pc1": r.get("spearman_level_vs_pc1_projection"),
            "spearman_level_vs_endpoint": r.get("spearman_level_vs_endpoint_projection"),
            "monotone_fraction_pc1": r.get("monotone_fraction_pc1_projection"),
            "cos_pc1_ordinal": (agree.get(layer) or {}).get("cos_pc1_ordinal"),
        }
        out.append(rec)

        def f(v: float | None) -> str:
            return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

        print(
            f"{trait:<19}{layer:>6}{f(rec['cos_endpoint_pc1']):>19}"
            f"{f(rec['pc1_variance_ratio']):>15}{f(rec['spearman_level_vs_pc1']):>13}"
            f"{f(rec['spearman_level_vs_endpoint']):>15}"
            f"{f(rec['monotone_fraction_pc1']):>10}"
        )

    cosines = [r["cos_endpoint_pc1"] for r in out if isinstance(r["cos_endpoint_pc1"], (int, float))]
    if cosines:
        print(
            f"\ncos(endpoint, pc1) range {min(cosines):.3f}-{max(cosines):.3f}, "
            f"mean {sum(cosines)/len(cosines):.3f}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"note": __doc__.strip(), "table": out}, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
