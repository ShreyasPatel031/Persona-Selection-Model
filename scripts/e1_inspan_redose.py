#!/usr/bin/env python3
"""E1 — in-span re-dosed IPIP sweeps for the four inventory wrinkles.

Pre-registered in docs/IPIP_WRINKLE_PLAN.md. Uses the same PC1 vectors and the
layers that produced 10/10 judged behaviour, but doses only inside the
behavioural trait span (not the coherence ceiling that poisoned E-up / N-up).

Poles and absolute-magnitude grids (residual units = direction_span × fraction):

    N-up   L20   0.05–0.55 × span   (monotone rise lived here in gemma_final)
    E-up   L15   0.40–1.00 × span   (skip the 0–0.3× dead zone)
    E-down L15   0.40–1.00 × span
    A-down L15   0.15–1.30 × span   (A-low effect sat at |mag|≈1088 > L15 span)

Pass gate (pre-registered): sign correct, Spearman ρ ≥ +0.8 over ≥4 usable
rungs (ρ of signed magnitude vs target EV; for low poles the signed dose is
negative so a falling EV yields positive ρ when correlated with |α| toward the
intended pole — we report both the sweep's built-in ρ and a signed-dose ρ),
and beats the matched-norm random control by ≥2× on |ΔEV|.

    # On a GPU host with vectors already on disk:
    python3 scripts/e1_inspan_redose.py \\
        --vectors-dir /content/ladder \\
        --out-dir results/e1_inspan \\
        --model-id unsloth/gemma-3-4b-it

No GPU in the cloud-agent VM — run this on Colab L4 (see docs/E1_COLAB.md).
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

logger = logging.getLogger("e1_inspan")

# Behavioural spans / layers from results/bipolar (+ bipolar_afix for A@L15).
# Recomputed from the vectors at startup; these are the expected values.
POLES = [
    {
        "trait": "neuroticism",
        "pole": "high",
        "layer": 20,
        "frac_lo": 0.05,
        "frac_hi": 0.55,
        "n_rungs": 8,
        "note": "N-up: window where gemma_final rose 3.30→3.39 before span-edge collapse",
    },
    {
        "trait": "extraversion",
        "pole": "high",
        "layer": 15,
        "frac_lo": 0.40,
        "frac_hi": 1.00,
        "n_rungs": 8,
        "note": "E-up: skip 0–0.3× dead zone; dense 0.4–1.0×",
    },
    {
        "trait": "extraversion",
        "pole": "low",
        "layer": 15,
        "frac_lo": 0.40,
        "frac_hi": 1.00,
        "n_rungs": 8,
        "note": "E-down: same window, negative sign",
    },
    {
        "trait": "agreeableness",
        "pole": "low",
        "layer": 15,
        "frac_lo": 0.15,
        "frac_hi": 1.30,
        "n_rungs": 8,
        "note": "A-down: widen past L15 span; effect previously at |mag|≈1088",
    },
]

PASS_RHO = 0.8
PASS_MARGIN = 2.0
MIN_USABLE = 4


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n < 1:
        return []
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    order_x = sorted(range(len(xs)), key=lambda i: xs[i])
    order_y = sorted(range(len(ys)), key=lambda i: ys[i])
    rx = [0.0] * len(xs)
    ry = [0.0] * len(ys)
    for pos, i in enumerate(order_x):
        rx[i] = float(pos + 1)
    for pos, i in enumerate(order_y):
        ry[i] = float(pos + 1)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    return num / den if den else None


def evaluate_sweep(sw: dict, pole: str) -> dict:
    """Pre-registered gate on a validated_sweep JSON."""
    curve = sw["trait_curve"]
    rows = [r for r in curve["rows"] if r.get("usable") and r.get("target_ev") is not None]
    # signed dose: +|mag| for high, −|mag| for low → positive ρ means EV moved
    # with the intended pole
    sign = 1.0 if pole == "high" else -1.0
    xs = [sign * abs(float(r["magnitude"])) for r in rows]
    ys = [float(r["target_ev"]) for r in rows]
    rho_signed = _spearman(xs, ys)
    # built-in sweep ρ is abs(alpha) vs EV (sign-blind); keep for continuity
    rho_abs = curve.get("spearman_absalpha_vs_target_ev")

    base = next((r for r in curve["rows"] if abs(r.get("alpha") or 0) < 1e-12), None)
    base_ev = float(base["target_ev"]) if base and base.get("target_ev") is not None else None
    # best usable in the intended direction
    if rows and base_ev is not None:
        if pole == "high":
            best = max(rows, key=lambda r: float(r["target_ev"]))
        else:
            best = min(rows, key=lambda r: float(r["target_ev"]))
        delta = float(best["target_ev"]) - base_ev
    else:
        best, delta = None, None

    sign_correct = (
        (delta is not None)
        and ((pole == "high" and delta > 0) or (pole == "low" and delta < 0))
    )
    # control margin on |Δ|
    ctrl_deltas = []
    for c in sw.get("control_curves") or []:
        crows = [r for r in c["rows"] if r.get("usable") and r.get("target_ev") is not None]
        if not crows or base_ev is None:
            continue
        if pole == "high":
            cbest = max(crows, key=lambda r: float(r["target_ev"]))
        else:
            cbest = min(crows, key=lambda r: float(r["target_ev"]))
        ctrl_deltas.append(abs(float(cbest["target_ev"]) - base_ev))
    max_ctrl = max(ctrl_deltas) if ctrl_deltas else None
    trait_abs = abs(delta) if delta is not None else None
    if trait_abs is not None and max_ctrl == 0:
        margin, beats = float("inf"), True
    elif trait_abs is not None and max_ctrl not in (None, 0):
        margin = trait_abs / max_ctrl
        beats = margin >= PASS_MARGIN
    else:
        margin, beats = None, False

    rho_ok = rho_signed is not None and rho_signed >= PASS_RHO
    usable_ok = len(rows) >= MIN_USABLE
    works = bool(sign_correct and rho_ok and usable_ok and beats)

    return {
        "n_usable": len(rows),
        "rho_signed_dose_vs_ev": None if rho_signed is None else round(rho_signed, 4),
        "rho_abs_alpha_vs_ev": rho_abs,
        "baseline_ev": base_ev,
        "best_ev": None if best is None else best["target_ev"],
        "best_magnitude": None if best is None else best["magnitude"],
        "delta": None if delta is None else round(delta, 4),
        "sign_correct": sign_correct,
        "max_control_abs_delta": None if max_ctrl is None else round(max_ctrl, 4),
        "margin": None if margin is None else (None if margin == float("inf") else round(margin, 3)),
        "beats_controls": beats,
        "works": works,
        "gate": {
            "pass_rho": PASS_RHO,
            "pass_margin": PASS_MARGIN,
            "min_usable": MIN_USABLE,
        },
    }


def run_one(
    spec: dict,
    *,
    vectors_dir: Path,
    out_dir: Path,
    model_id: str,
    items_csv: Path,
    n_controls: int,
    n_probes: int,
    max_new_tokens: int,
) -> dict:
    from app.persona.intensity_ladder import (
        direction_span_magnitude,
        run_validated_sweep,
    )
    from app.persona.ocean_probes import PROBE_QUESTIONS

    trait, pole, layer = spec["trait"], spec["pole"], spec["layer"]
    vec_pt = vectors_dir / f"ladder_vectors_{trait}.pt"
    if not vec_pt.is_file():
        raise FileNotFoundError(vec_pt)

    blob = torch.load(vec_pt, map_location="cpu")
    stack = blob["v_pc1"]
    cents = blob["level_centroids"]
    span = direction_span_magnitude(cents, layer, stack[layer])
    mags = _linspace(spec["frac_lo"] * span, spec["frac_hi"] * span, spec["n_rungs"])
    # run_validated_sweep expects unsigned alphas; with alpha_units=raw these
    # ARE the residual magnitudes. Sign is applied internally from steer_toward.
    logger.info(
        "%s-%s L%s span=%.1f grid(abs)=%s",
        trait,
        pole,
        layer,
        span,
        [round(m, 1) for m in mags],
    )

    out_json = out_dir / f"validated_sweep_{trait}_pc1_{pole}.json"
    run_validated_sweep(
        vec_pt,
        out_json,
        trait=trait,
        which="pc1",
        layer_idx=layer,
        magnitudes=mags,
        auto_calibrate=False,
        steer_toward=pole,
        n_random_controls=n_controls,
        alpha_units="raw",
        model_id=model_id,
        items_csv=items_csv,
        probe_questions=PROBE_QUESTIONS[: max(1, n_probes)],
        max_new_tokens=max_new_tokens,
        baseline="persona_free",
    )
    sw = json.loads(out_json.read_text())
    gate = evaluate_sweep(sw, pole)
    return {
        "trait": trait,
        "pole": pole,
        "layer": layer,
        "span": round(span, 2),
        "frac_lo": spec["frac_lo"],
        "frac_hi": spec["frac_hi"],
        "magnitude_grid_abs": [round(m, 2) for m in mags],
        "note": spec["note"],
        "report": str(out_json),
        **gate,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectors-dir", required=True, help="Dir with ladder_vectors_{trait}.pt")
    p.add_argument("--out-dir", default="results/e1_inspan")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--random-controls", type=int, default=1)
    p.add_argument("--probes", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--only",
        default="",
        help="Comma list of trait:pole to run (default: all four). "
        "Example: neuroticism:high,extraversion:high",
    )
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-score existing JSONs in --out-dir; no GPU.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = Path(args.vectors_dir)
    items_csv = Path(args.items_csv)

    wanted = None
    if args.only.strip():
        wanted = {tuple(x.strip().split(":")) for x in args.only.split(",") if x.strip()}

    rows: list[dict] = []
    for spec in POLES:
        key = (spec["trait"], spec["pole"])
        if wanted is not None and key not in wanted:
            continue
        if args.evaluate_only:
            path = out_dir / f"validated_sweep_{spec['trait']}_pc1_{spec['pole']}.json"
            if not path.is_file():
                rows.append({**{k: spec[k] for k in ("trait", "pole", "layer")}, "error": f"missing {path}"})
                continue
            sw = json.loads(path.read_text())
            gate = evaluate_sweep(sw, spec["pole"])
            rows.append({"trait": spec["trait"], "pole": spec["pole"], "layer": spec["layer"], "report": str(path), **gate})
            continue
        try:
            rows.append(
                run_one(
                    spec,
                    vectors_dir=vectors_dir,
                    out_dir=out_dir,
                    model_id=args.model_id,
                    items_csv=items_csv,
                    n_controls=args.random_controls,
                    n_probes=args.probes,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        except Exception as exc:
            logger.exception("%s-%s failed: %s", spec["trait"], spec["pole"], exc)
            rows.append(
                {
                    "trait": spec["trait"],
                    "pole": spec["pole"],
                    "layer": spec["layer"],
                    "error": str(exc),
                    "works": False,
                }
            )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "e1_inspan_redose",
        "pass_gate": {"rho_signed": PASS_RHO, "margin": PASS_MARGIN, "min_usable": MIN_USABLE},
        "n_passing": sum(1 for r in rows if r.get("works")),
        "n_total": len(rows),
        "table": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 100)
    print("E1 IN-SPAN REDOSE  (pass: sign ok, ρ_signed≥0.8, ≥4 usable, margin≥2×)")
    print("=" * 100)
    print(
        f"{'trait':16} {'pole':5} {'L':>3} {'ρ_signed':>9} {'ΔEV':>7} {'ctrl':>7} "
        f"{'margin':>7} {'usable':>7} {'works':>6}"
    )
    print("-" * 100)
    for r in rows:
        if r.get("error"):
            print(f"{r['trait']:16} {r['pole']:5} ERROR: {r['error'][:70]}")
            continue
        margin = r.get("margin")
        margin_s = "inf" if margin is None and r.get("beats_controls") else (
            f"{margin:.2f}" if isinstance(margin, (int, float)) else "—"
        )
        print(
            f"{r['trait']:16} {r['pole']:5} {r['layer']:>3} "
            f"{(r.get('rho_signed_dose_vs_ev') or float('nan')):>+9.3f} "
            f"{(r.get('delta') or 0):>+7.3f} "
            f"{(r.get('max_control_abs_delta') if r.get('max_control_abs_delta') is not None else float('nan')):>7.3f} "
            f"{margin_s:>7} "
            f"{r.get('n_usable', 0):>7} "
            f"{str(r.get('works')):>6}"
        )
    print("-" * 100)
    print(f"passing: {summary['n_passing']}/{summary['n_total']}   wrote {summary_path}")

    # Branch hint for the plan
    by = {(r["trait"], r["pole"]): r for r in rows if "works" in r}
    n_up = by.get(("neuroticism", "high"), {})
    e_up = by.get(("extraversion", "high"), {})
    if summary["n_passing"] == summary["n_total"] and summary["n_total"] > 0:
        print("\nBRANCH: all four passed → skip to E8 (unified final run).")
    elif n_up.get("works") and not e_up.get("works"):
        print("\nBRANCH: N-up ok, E-up failed → E2 (v_probe) then E3 (layer scan).")
    elif not n_up.get("works"):
        print("\nBRANCH: N-up failed even in-span → E4 (guardrail hypothesis).")
    else:
        print("\nBRANCH: mixed — see docs/IPIP_WRINKLE_PLAN.md.")
    return 0 if summary["n_passing"] == summary["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
