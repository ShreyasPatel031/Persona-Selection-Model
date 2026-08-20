#!/usr/bin/env python3
"""Final ablation: on THEIR setup, does swapping only the vector recover the inventory?

Everything is held at Blas et al.'s choices except the one variable under test:

    model       Llama-3.1-8B-Instruct   (theirs; ungated mirror)
    instrument  MPI-120                 (theirs)
    readout     argmax over option tokens (theirs, and what a human respondent does)
    injection   full sequence            (held constant so scope is not a co-variable)

Three arms decompose *estimator* from *corpus*:

    theirs_meandiff_statement   two-arm mean-difference,  their 500 construct vs
                                500 antithesis statements        <- their pipeline
    ours_endpoint               two-arm mean-difference,  our nine-level ladder
                                activations recorded on the inventory
    ours_pc1                    PC1 over nine levels,     same ladder activations

    ours_pc1 vs ours_endpoint  ->  does the estimator matter?
    ours_endpoint vs theirs    ->  does the corpus matter?

Both poles are run for every arm, because the decisive question is not whether a
score moves but whether **flipping the vector flips the direction of movement**.
A monotone dose-score correlation is not sufficient: a degrading respondent
collapsing onto one option produces a clean correlation with no trait content, so
single-option dominance is tracked against dose as well (see E0).

What this can and cannot establish
----------------------------------
It CAN show whether switching the vector-derivation restores direction-controlled,
dose-ordered, control-beating movement on their own model and instrument.

It CANNOT show that "everything works out" or that anyone gets perfect correlation.
Our own Gemma inventory results are partial: C-up clears its control by only 1.9x,
C-down's large delta rests on a single non-monotone rung, and E-up fails its
control outright. Two random controls and one administration per rung do not
support a precision claim. Read the output as a direction-of-effect result.

    python3 scripts/final_ablation_their_setup.py --out-dir results/final_ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("final_ablation")

TRAIT = "conscientiousness"
THEIR_VECTORS = Path("/tmp/psych-steer/replication/vectors/Llama-3.1-8B-Instruct")
MIN_RUNGS = 3
CLEAR_MARGIN = 2.0


def shipped_screen(lock: dict) -> bool:
    return lock["top_option_fraction"] < 0.90 and lock["option_entropy"] >= 0.30


def load_their_vector(layer: int, variant: str = "meandiff", mode: str = "statement") -> torch.Tensor | None:
    base = THEIR_VECTORS / TRAIT / variant / mode
    p = base / f"layer_{layer}.pt"
    if not p.is_file():
        cands = [c for c in sorted(base.glob(f"layer_{layer}_*.pt")) if not c.name.endswith("_raw.pt")]
        if not cands:
            return None
        p = cands[0]
    return torch.load(p, map_location="cpu").detach().float().reshape(-1)


def analyze(path: Path, pole: str) -> dict:
    """Argmax dose-response plus the two tests that separate signal from collapse."""
    from app.persona.intensity_ladder import monotone_fraction, spearman_rho

    d = json.loads(path.read_text())
    rows = d["trait_curve"]["rows"]
    keep = [r for r in rows if r.get("target_argmax") is not None and shipped_screen(r["lock"])]
    base = next((float(r["target_argmax"]) for r in rows if abs(float(r["magnitude"])) < 1e-9), None)
    if len(keep) < MIN_RUNGS or base is None:
        return {"n_usable_rungs": len(keep), "insufficient": True}

    doses = [abs(float(r["magnitude"])) for r in keep]
    scores = [float(r["target_argmax"]) for r in keep]
    tops = [float(r["lock"]["top_option_fraction"]) for r in keep]
    extreme = max(scores) if pole == "high" else min(scores)
    delta = extreme - base
    rho = spearman_rho(doses, scores)

    ctrl = []
    for c in d.get("control_curves") or []:
        crows = [r for r in c["rows"] if r.get("target_argmax") is not None and shipped_screen(r["lock"])]
        if len(crows) >= 2:
            v = [float(r["target_argmax"]) for r in crows]
            ctrl.append(max(v) - min(v))
    ctrl_span = max(ctrl) if ctrl else None
    span = max(scores) - min(scores)
    ratio = (span / ctrl_span) if ctrl_span not in (None, 0) else None
    sign_ok = rho is not None and ((rho > 0) if pole == "high" else (rho < 0))

    return {
        "n_usable_rungs": len(keep),
        "baseline_argmax": round(base, 4),
        "delta": round(delta, 4),
        "span": round(span, 4),
        "rho_dose_vs_score": None if rho is None else round(rho, 4),
        "sign_correct": bool(sign_ok),
        "monotone_fraction": (
            None if monotone_fraction(scores) is None else round(monotone_fraction(scores), 3)
        ),
        "max_control_span": None if ctrl_span is None else round(ctrl_span, 4),
        "span_over_control": None if ratio is None else round(ratio, 2),
        "beats_control": bool(ratio is not None and ratio >= CLEAR_MARGIN),
        "rho_dose_vs_top_option_fraction": (
            None
            if spearman_rho(doses, tops) is None
            else round(spearman_rho(doses, tops), 4)
        ),
        "collapse_suspected": bool(
            spearman_rho(doses, tops) is not None and spearman_rho(doses, tops) > 0.7
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/final_ablation")
    p.add_argument("--ladder-dir", default="results/e1_vector", help="Holds ladder_vectors_<trait>.pt from the Llama ladder")
    p.add_argument("--model-id", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--items-csv", default="/tmp/mpi_120.csv", help="MPI-120 (their instrument)")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--random-controls", type=int, default=2)
    p.add_argument("--scope", default="full")
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layer = int(args.layer)

    from app.persona.intensity_ladder import direction_span_magnitude, run_validated_sweep

    blob = torch.load(Path(args.ladder_dir) / f"ladder_vectors_{TRAIT}.pt", map_location="cpu")
    centroids: torch.Tensor = blob["level_centroids"]

    arms: list[tuple[str, torch.Tensor]] = [
        ("ours_pc1", blob["v_pc1"][layer]),
        ("ours_endpoint", blob["v_endpoint"][layer]),
    ]
    theirs = load_their_vector(layer)
    if theirs is not None:
        arms.append(("theirs_meandiff_statement", theirs))
    else:
        logger.warning("their vector missing at layer %s", layer)

    # One dose grid in residual units for every arm, keyed to our PC1's ladder span.
    # Their unit vector's own ladder span is ~0.01 residual units (near-orthogonal to
    # the ladder), so own-span dosing would inject essentially nothing and manufacture
    # a null. Matched L2 is the fair comparison.
    dose_span = direction_span_magnitude(centroids, layer, blob["v_pc1"][layer])
    mags = [dose_span * m for m in (0.25, 0.5, 1.0, 1.5, 2.0)]
    logger.info("dose grid (residual units, matched L2): %s", [round(m) for m in mags])

    rows: list[dict] = []
    for name, direction in arms:
        own_span = direction_span_magnitude(centroids, layer, direction)
        for pole in ("high", "low"):
            out_json = out / f"sweep_{TRAIT}_{name}_{pole}.json"
            if not args.evaluate_only:
                arm_blob = dict(blob)
                stack = blob["v_pc1"].clone()
                stack[layer] = direction.to(stack.dtype)
                arm_blob["v_pc1"] = stack
                arm_pt = out / f"_arm_{name}.pt"
                torch.save(arm_blob, arm_pt)
                logger.info("=== %s %s (own ladder span %.4f) ===", name, pole, own_span)
                try:
                    run_validated_sweep(
                        arm_pt,
                        out_json,
                        trait=TRAIT,
                        which="pc1",
                        layer_idx=layer,
                        magnitudes=mags,
                        auto_calibrate=False,
                        steer_toward=pole,
                        n_random_controls=args.random_controls,
                        alpha_units="raw",
                        model_id=args.model_id,
                        items_csv=Path(args.items_csv),
                        probe_questions=[],
                        baseline="persona_free",
                        injection_scope=args.scope,
                    )
                except Exception as exc:
                    logger.exception("%s %s failed", name, pole)
                    rows.append({"arm": name, "pole": pole, "error": str(exc)})
                    continue
            if not out_json.is_file():
                rows.append({"arm": name, "pole": pole, "error": "no report"})
                continue
            rows.append(
                {
                    "arm": name,
                    "pole": pole,
                    "own_ladder_span": round(own_span, 4),
                    "report": str(out_json),
                    **analyze(out_json, pole),
                }
            )

    # Bipolar sign control per arm: does flipping the vector flip the movement?
    bipolar: dict[str, dict] = {}
    for name, _ in arms:
        hi = next((r for r in rows if r["arm"] == name and r["pole"] == "high" and "delta" in r), None)
        lo = next((r for r in rows if r["arm"] == name and r["pole"] == "low" and "delta" in r), None)
        if hi and lo:
            opposed = hi["delta"] > 0 and lo["delta"] < 0
            bipolar[name] = {
                "up_delta": hi["delta"],
                "down_delta": lo["delta"],
                "signs_opposed": opposed,
                "up_beats_control": hi["beats_control"],
                "down_beats_control": lo["beats_control"],
                "collapse_suspected": hi["collapse_suspected"] or lo["collapse_suspected"],
                "direction_controlled": bool(
                    opposed and not (hi["collapse_suspected"] or lo["collapse_suspected"])
                ),
            }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "final_ablation_their_setup",
        "held_fixed": {
            "model": args.model_id,
            "instrument": "MPI-120",
            "readout": "argmax over option tokens",
            "injection_scope": args.scope,
            "layer": layer,
            "dose_grid_residual_units": [round(m, 2) for m in mags],
            "n_random_controls": args.random_controls,
        },
        "varied": "vector only (estimator x corpus)",
        "limits": (
            "Two random controls and one administration per rung. Supports a "
            "direction-of-effect conclusion, not a precision or 'perfect correlation' "
            "claim."
        ),
        "bipolar_sign_control": bipolar,
        "table": rows,
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 104)
    print("FINAL ABLATION - their model, their instrument (MPI-120), their readout; vector varied")
    print("=" * 104)
    hdr = (
        f"{'arm':<28}{'pole':<6}{'rho':>8}{'delta':>8}{'span':>7}"
        f"{'ctrl':>7}{'ratio':>7}{'mono':>6}{'ρ(dose,top)':>12}{'ok':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r or r.get("insufficient"):
            print(f"{r['arm']:<28}{r['pole']:<6} {r.get('error') or 'insufficient usable rungs'}")
            continue
        print(
            f"{r['arm']:<28}{r['pole']:<6}{str(r['rho_dose_vs_score']):>8}{str(r['delta']):>8}"
            f"{str(r['span']):>7}{str(r['max_control_span']):>7}{str(r['span_over_control']):>7}"
            f"{str(r['monotone_fraction']):>6}{str(r['rho_dose_vs_top_option_fraction']):>12}"
            f"{str(r['beats_control']):>6}"
        )
    print("\nBipolar sign control (flipping the vector should flip the movement)")
    for name, b in bipolar.items():
        print(
            f"  {name:<28} up={b['up_delta']:<8} down={b['down_delta']:<8} "
            f"opposed={str(b['signs_opposed']):<6} collapse={str(b['collapse_suspected']):<6} "
            f"direction_controlled={b['direction_controlled']}"
        )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
