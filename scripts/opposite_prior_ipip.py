#!/usr/bin/env python3
"""Opposite-prior IPIP sweep: the inventory version of the bipolar judge test.

Why this exists. Every inventory sweep so far started from ``persona_free``
("answer as yourself"), which lands near the scale midpoint (~3.0 on 1-5). From
there the reachable range is at most ~2 points in either direction, and in
practice the model never approaches the ends, so the measured effects were
0.08-0.33 EV even when the dose-response correlation was near-perfect. The
correlation was real; the effect size was capped by the starting point, not by
the vector.

The judge protocol (scripts/bipolar_judge.py) does not have this problem because
it starts each pole from the prompt prior it has room to move away from:

    up    baseline = low-pole ladder prompt  (level 2), steer +v, reference level 9
    down  baseline = high-pole ladder prompt (level 8), steer -v, reference level 1

This script applies that design to the 120-item IPIP-NEO. It also records the
*prompt gap* (unsteered baseline level vs unsteered reference level) so the
steered movement can be reported as a fraction of what prompting achieves on the
same instrument — the same normalization the judge runs use.

No LLM judge anywhere: items are scored by expected value over the Likert option
token probabilities, and rungs whose option distribution collapses are screened
out by the lock test rather than silently averaged in.

    python3 scripts/opposite_prior_ipip.py \\
        --vectors-dir /content/ladder \\
        --out results/opposite_prior_ipip/summary.json
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

logger = logging.getLogger("opposite_prior_ipip")

# Layers that steer *judged free-text* behaviour bipolarly (results/bipolar).
# These were selected against an LLM-judge readout, not against the inventory.
# Agreeableness is L20 here to match results/bipolar; runs before 2026-08-20 used
# L15 for it, which matched neither table.
JUDGE_PROVEN_LAYERS = {
    "extraversion": 15,
    "agreeableness": 20,
    "conscientiousness": 17,
    "neuroticism": 20,
    "openness": 19,
}

# Layers at which an *inventory* dose-response was actually validated
# (results/gemma_final, results/e1_inspan validated_sweep_* artifacts).
# O and C differ from the judge choice, so a null at the judge layer confounds
# "the opposite prior does not help" with "wrong layer for this readout".
INVENTORY_LAYERS = {
    "extraversion": 15,
    "agreeableness": 15,
    "conscientiousness": 15,
    "neuroticism": 20,
    "openness": 15,
}

LAYER_SOURCES = {"judge": JUDGE_PROVEN_LAYERS, "inventory": INVENTORY_LAYERS}

# pole -> (baseline ladder level, reference ladder level, sign)
POLES = {
    "up": (2, 9, +1),
    "down": (8, 1, -1),
}

PASS_RHO = 0.8
PASS_MARGIN = 2.0
MIN_USABLE = 4


def _grid(span: float, lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [span * lo]
    step = (hi - lo) / (n - 1)
    return [span * (lo + i * step) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vectors-dir", required=True)
    p.add_argument("--out", default="results/opposite_prior_ipip/summary.json")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--traits", default="extraversion,agreeableness,conscientiousness,neuroticism,openness")
    p.add_argument("--poles", default="up,down")
    p.add_argument("--direction", default="pc1", choices=("pc1", "probe", "endpoint", "ordinal"))
    p.add_argument("--rungs", type=int, default=8)
    p.add_argument("--frac-lo", type=float, default=0.15)
    p.add_argument("--frac-hi", type=float, default=1.30)
    p.add_argument("--random-controls", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--layer-source",
        default="judge",
        choices=tuple(LAYER_SOURCES),
        help="Which validated-layer table to steer at (see module constants).",
    )
    p.add_argument(
        "--layers", default="", help="Per-trait override, e.g. openness:15,conscientiousness:15"
    )
    p.add_argument("--n-markers", type=int, default=3)
    p.add_argument(
        "--prompt-style",
        default="self",
        help=(
            "Persona framing for the prior and reference prompts, or a per-trait "
            "map like openness:character,default:self. The prior prompts decide "
            "what the sweep is measuring, so they must match the framing the "
            "vectors were derived from; see results/prior_prompt_calibration."
        ),
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.intensity_ladder import (
        _load_model,
        _option_token_ids,
        _Steering,
        administer_inventory,
        direction_span_magnitude,
        items_from_csv,
        monotone_fraction,
        option_lock,
        random_control_directions,
        response_validity,
        score_traits_ev,
        spearman_rho,
    )
    from app.persona.intensity_prompts import ladder_system_prompt

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    poles = [x.strip() for x in args.poles.split(",") if x.strip()]
    key = {"pc1": "v_pc1", "probe": "v_probe", "endpoint": "v_endpoint", "ordinal": "v_ordinal"}[
        args.direction
    ]

    layers = dict(LAYER_SOURCES[args.layer_source])
    for part in (x for x in args.layers.split(",") if x.strip()):
        name, _, value = part.partition(":")
        layers[name.strip()] = int(value)

    styles: dict[str, str] = {}
    default_style = "self"
    for part in (x for x in args.prompt_style.split(",") if x.strip()):
        name, sep, value = part.partition(":")
        if not sep:
            default_style = name.strip()
        elif name.strip() == "default":
            default_style = value.strip()
        else:
            styles[name.strip()] = value.strip()

    def style_for(trait: str) -> str:
        return styles.get(trait, default_style)

    items = items_from_csv(Path(args.items_csv))
    model, tokenizer, dev = _load_model(args.model_id, None)
    option_ids = _option_token_ids(tokenizer)

    def administer(system: str, direction: torch.Tensor | None, magnitude: float) -> dict:
        if direction is None or magnitude == 0.0:
            responses, _ = administer_inventory(
                model, tokenizer, dev, system, items, option_ids=option_ids
            )
        else:
            with _Steering(model, administer.layer, direction, magnitude):
                responses, _ = administer_inventory(
                    model, tokenizer, dev, system, items, option_ids=option_ids
                )
        lock = option_lock(responses)
        ev = score_traits_ev(responses)
        # Inventory-wide histograms cannot say whether the *target* domain was
        # pinned to the midpoint or whether the prompt bled onto other domains,
        # and those two have opposite interpretations for a flat EV curve.
        lock_by_trait = {
            t: option_lock([r for r in responses if str(r["trait"]) == t])
            for t in sorted({str(r["trait"]) for r in responses})
        }
        return {
            "magnitude": round(float(magnitude), 2),
            "ev_scores": {k: round(v, 4) for k, v in ev.items()},
            "target_ev": round(float(ev[administer.trait]), 4),
            "response_validity": round(response_validity(responses), 4),
            "lock": lock,
            "target_lock": lock_by_trait.get(administer.trait),
            "lock_by_trait": lock_by_trait,
            "usable": not lock["locked"],
        }

    rows_out: list[dict] = []
    for trait in traits:
        vec_pt = Path(args.vectors_dir) / f"ladder_vectors_{trait}.pt"
        blob = torch.load(vec_pt, map_location="cpu")
        layer = layers[trait]
        stack = blob[key]
        cents = blob["level_centroids"]
        v = stack[layer].float()
        v = v / v.norm().clamp_min(1e-9)
        span = direction_span_magnitude(cents, layer, stack[layer])
        controls = random_control_directions(
            int(v.shape[0]), args.random_controls, seed=args.seed, like=v
        )
        mags = _grid(span, args.frac_lo, args.frac_hi, args.rungs)

        administer.layer = layer
        administer.trait = trait

        for pole in poles:
            base_level, ref_level, sign = POLES[pole]
            base_system = ladder_system_prompt(
                trait, base_level, n_markers=args.n_markers, style=style_for(trait)
            )
            ref_system = ladder_system_prompt(
                trait, ref_level, n_markers=args.n_markers, style=style_for(trait)
            )

            logger.info(
                "%s-%s L%s span=%.1f prior=level%s ref=level%s grid=%s",
                trait, pole, layer, span, base_level, ref_level,
                [round(m) for m in mags],
            )

            baseline = administer(base_system, None, 0.0)
            reference = administer(ref_system, None, 0.0)
            prompt_gap = float(reference["target_ev"]) - float(baseline["target_ev"])
            logger.info(
                "  prior EV=%.3f  reference EV=%.3f  prompt gap=%+.3f",
                baseline["target_ev"], reference["target_ev"], prompt_gap,
            )

            def curve(direction: torch.Tensor, label: str) -> list[dict]:
                out = [baseline if label == "trait" else administer(base_system, None, 0.0)]
                for m in mags:
                    row = administer(base_system, direction, sign * m)
                    out.append(row)
                    logger.info(
                        "  %s mag=%+8.1f ev=%.3f usable=%s",
                        label, sign * m, row["target_ev"], row["usable"],
                    )
                return out

            trait_rows = curve(v, "trait")
            control_curves = [curve(cv, f"random{i}") for i, cv in enumerate(controls)]

            def score(rows: list[dict]) -> dict:
                usable = [r for r in rows if r["usable"] and r["target_ev"] is not None]
                xs = [abs(float(r["magnitude"])) for r in usable]
                ys = [float(r["target_ev"]) for r in usable]
                # for "down", EV should fall as |mag| grows -> negate for signed rho
                ys_signed = ys if pole == "up" else [-y for y in ys]
                rho = spearman_rho(xs, ys_signed)
                base_ev = float(rows[0]["target_ev"])
                if usable:
                    best = (max if pole == "up" else min)(usable, key=lambda r: float(r["target_ev"]))
                    delta = float(best["target_ev"]) - base_ev
                else:
                    best, delta = None, None
                return {
                    "n_usable": len(usable),
                    "rho_signed": None if rho is None else round(float(rho), 4),
                    "monotone_fraction": (
                        None if not ys_signed else round(float(monotone_fraction(ys_signed) or 0.0), 4)
                    ),
                    "baseline_ev": round(base_ev, 4),
                    "best_ev": None if best is None else best["target_ev"],
                    "best_magnitude": None if best is None else best["magnitude"],
                    "delta": None if delta is None else round(delta, 4),
                    "ev_curve": [round(float(r["target_ev"]), 4) for r in rows],
                    "usable_flags": [bool(r["usable"]) for r in rows],
                }

            t_score = score(trait_rows)
            c_scores = [score(c) for c in control_curves]
            max_ctrl = max((abs(c["delta"] or 0.0) for c in c_scores), default=None)
            t_abs = abs(t_score["delta"] or 0.0)
            if max_ctrl in (None, 0.0):
                margin, beats = None, t_abs > 0
            else:
                margin = round(t_abs / max_ctrl, 3)
                beats = margin >= PASS_MARGIN

            sign_ok = (t_score["delta"] is not None) and (
                (pole == "up" and t_score["delta"] > 0) or (pole == "down" and t_score["delta"] < 0)
            )
            pct_gap = (
                round(100.0 * (t_score["delta"] / prompt_gap), 1)
                if prompt_gap not in (0.0, None) and t_score["delta"] is not None
                else None
            )
            works = bool(
                sign_ok
                and t_score["rho_signed"] is not None
                and t_score["rho_signed"] >= PASS_RHO
                and t_score["n_usable"] >= MIN_USABLE
                and beats
            )

            rec = {
                "trait": trait,
                "pole": pole,
                "layer": layer,
                "direction": args.direction,
                "prompt_style": style_for(trait),
                "n_markers": args.n_markers,
                "span": round(span, 2),
                "prior_level": base_level,
                "reference_level": ref_level,
                "prior_ev": baseline["target_ev"],
                "reference_ev": reference["target_ev"],
                "prompt_gap": round(prompt_gap, 4),
                "magnitude_grid": [round(m, 2) for m in mags],
                **t_score,
                "pct_of_prompt_gap": pct_gap,
                "sign_correct": sign_ok,
                "max_control_abs_delta": max_ctrl,
                "margin": margin,
                "beats_controls": beats,
                "works": works,
                "control_curves": c_scores,
                "trait_rows": trait_rows,
                "reference_row": reference,
            }
            rows_out.append(rec)

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "opposite_prior_ipip",
                        "model_id": args.model_id,
                        "instrument": Path(args.items_csv).name,
                        "n_items": len(items),
                        "layer_source": args.layer_source,
                        "layers": layers,
                        "pass_gate": {
                            "rho_signed": PASS_RHO,
                            "margin": PASS_MARGIN,
                            "min_usable": MIN_USABLE,
                        },
                        "table": rows_out,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    print("\n" + "=" * 108)
    print("OPPOSITE-PRIOR IPIP  (up: from level-2 prior; down: from level-8 prior)")
    print("=" * 108)
    print(
        f"{'trait':16} {'pole':5} {'L':>3} {'prior':>6} {'best':>6} {'ref':>6} "
        f"{'Δ':>7} {'gap':>7} {'%gap':>7} {'ρ':>7} {'ctrl':>6} {'works':>6}"
    )
    print("-" * 108)
    for r in rows_out:
        print(
            f"{r['trait']:16} {r['pole']:5} {r['layer']:>3} "
            f"{r['prior_ev']:>6.2f} "
            f"{(r['best_ev'] if r['best_ev'] is not None else float('nan')):>6.2f} "
            f"{r['reference_ev']:>6.2f} "
            f"{(r['delta'] if r['delta'] is not None else float('nan')):>+7.3f} "
            f"{r['prompt_gap']:>+7.3f} "
            f"{(r['pct_of_prompt_gap'] if r['pct_of_prompt_gap'] is not None else float('nan')):>6.1f}% "
            f"{(r['rho_signed'] if r['rho_signed'] is not None else float('nan')):>+7.3f} "
            f"{(r['max_control_abs_delta'] if r['max_control_abs_delta'] is not None else 0.0):>6.3f} "
            f"{str(r['works']):>6}"
        )
    print("-" * 108)
    print(f"passing: {sum(1 for r in rows_out if r['works'])}/{len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
