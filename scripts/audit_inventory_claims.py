"""Stress-test every inventory dose-response claim before anyone cites it.

Three questions a reviewer will ask, answered from the committed artifacts:

1. **Does the result depend on our non-standard readout?** The prompting papers
   and Blas et al. score an item by its committed answer (argmax over the option
   tokens), which is also what a human respondent does. We report expected value
   over the option distribution. Both are recorded per rung, so the same
   activations can be scored both ways.

2. **Does the result depend on rungs where the respondent has partly collapsed?**
   The shipped screen only rejects a rung when one option covers >=90% of items or
   option entropy < 0.30 nats, so a rung answering with two of five options
   passes. Recompute under progressively stricter usability rules.

3. **Is the control comparison strong enough?** Report how many random control
   directions each sweep actually used and whether any rung was ever repeated,
   i.e. whether an error bar exists.

Run: PYTHONPATH=. python3 scripts/audit_inventory_claims.py
No GPU needed - pure reanalysis of committed JSON artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.persona.intensity_ladder import spearman_rho

SWEEP_DIRS = (Path("results/e1_inspan"), Path("results/gemma_final"))
OUT = Path("results/inventory_claim_audit.json")

MIN_RUNGS = 3

# Progressively stricter definitions of "this rung is a usable measurement".
USABILITY_RULES: dict[str, object] = {
    "as_shipped": lambda l: l["top_option_fraction"] < 0.90 and l["option_entropy"] >= 0.30,
    "top_lt_080": lambda l: l["top_option_fraction"] < 0.80,
    "top_lt_075_min4_options": lambda l: l["top_option_fraction"] < 0.75
    and l["distinct_options"] >= 4,
    "all5_options_top_lt_075": lambda l: l["distinct_options"] == 5
    and l["top_option_fraction"] < 0.75,
}


def steer_sign(report: dict) -> int:
    return -1 if report["verdict"]["steered_toward"] == "low" else 1


def rho_under(rows: list[dict], ok, key: str) -> tuple[float | None, int]:
    keep = [r for r in rows if r.get(key) is not None and ok(r["lock"])]
    if len(keep) < MIN_RUNGS:
        return None, len(keep)
    xs = [abs(r["alpha"]) for r in keep]
    ys = [float(r[key]) for r in keep]
    return spearman_rho(xs, ys), len(keep)


def main() -> None:
    records: list[dict] = []
    for directory in SWEEP_DIRS:
        for path in sorted(directory.glob("validated_sweep_*.json")):
            rep = json.loads(path.read_text())
            rows = rep["trait_curve"]["rows"]
            sign = steer_sign(rep)
            mags = [round(abs(r["magnitude"]), 3) for r in rows]

            rec: dict = {
                "sweep": path.stem.replace("validated_sweep_", ""),
                "grid": "in-span" if directory.name == "e1_inspan" else "ceiling",
                "steered_toward": rep["verdict"]["steered_toward"],
                "n_random_controls": len(rep.get("control_curves") or []),
                "n_repeated_doses": len(mags) - len(set(mags)),
                "readout": {},
                "usability": {},
            }

            for key, label in (("target_ev", "expected_value"), ("target_argmax", "argmax")):
                rho, n = rho_under(rows, USABILITY_RULES["as_shipped"], key)
                rec["readout"][label] = {
                    "rho": round(rho, 4) if rho is not None else None,
                    "n_rungs": n,
                    "sign_correct": None
                    if rho is None
                    else bool(rho > 0) if sign > 0 else bool(rho < 0),
                }

            for name, ok in USABILITY_RULES.items():
                rho, n = rho_under(rows, ok, "target_ev")
                rec["usability"][name] = {
                    "rho": round(rho, 4) if rho is not None else None,
                    "n_rungs": n,
                    "sign_correct": None
                    if rho is None
                    else bool(rho > 0) if sign > 0 else bool(rho < 0),
                }
            records.append(rec)

    def tally(getter) -> str:
        measurable = [r for r in records if getter(r)["rho"] is not None]
        ok = [r for r in measurable if getter(r)["sign_correct"]]
        return f"{len(ok)}/{len(measurable)} sign-correct ({len(measurable)}/{len(records)} measurable)"

    print("Readout sensitivity (shipped usability screen):")
    print(f"  expected value : {tally(lambda r: r['readout']['expected_value'])}")
    print(f"  argmax         : {tally(lambda r: r['readout']['argmax'])}")

    print("\nUsability sensitivity (expected-value readout):")
    for name in USABILITY_RULES:
        print(f"  {name:<26}: {tally(lambda r, n=name: r['usability'][n])}")

    print("\nPer sweep, expected value under each usability rule:")
    hdr = f"{'sweep':<27}{'grid':<9}" + "".join(f"{n[:20]:>22}" for n in USABILITY_RULES)
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        line = f"{r['sweep']:<27}{r['grid']:<9}"
        for name in USABILITY_RULES:
            cell = r["usability"][name]
            n = cell["n_rungs"]
            if cell["rho"] is None:
                text = f"only {n} rungs"
            else:
                mark = "" if cell["sign_correct"] else "  WRONG SIGN"
                text = f"{cell['rho']:+.2f} n={n}{mark}"
            line += f"{text:>22}"
        print(line)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "note": __doc__.strip(),
                "usability_rules": list(USABILITY_RULES),
                "table": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
