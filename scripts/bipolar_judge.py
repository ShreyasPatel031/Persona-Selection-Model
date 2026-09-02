#!/usr/bin/env python3
"""Bipolar away-from-prior test: does one vector scale a trait UP and DOWN?

Each pole is tested from the prior it has room to move away from, because a
persona-free baseline already reads 75-90/100 on E/A/C/O to a judge and leaves
no headroom up:

    up    baseline = low-pole prompt  (level 2), steer +v, reference = level 9
    down  baseline = high-pole prompt (level 8), steer -v, reference = level 1

Success for a pole means the judge score moves monotonically from the baseline
toward the reference as |magnitude| grows, and beats a matched-norm random
control run against the same baseline. Refusing and incoherent replies are
dropped before any correlation is computed — steered neuroticism's first
apparent "increase" was a refusal ("I'm sorry, but I cannot fulfill this
request") scoring 60/100 for anxiety.

Dose grids are built from the span of the direction *being steered* at the
layer being used (``direction_span_magnitude``). Using the PC1 span for a layer
chosen on other grounds gave agreeableness a 53-unit grid against a residual
norm of 27,000 — it was never dosed at all.

Stage 1 (GPU): generate.  Stage 2 (CPU + Vertex): judge.

    python3 scripts/bipolar_judge.py generate --vectors-dir DIR --out gen.json
    python3 scripts/bipolar_judge.py judge --generations gen.json --out summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("bipolar_judge")

TRAITS_RUBRIC = {
    "extraversion": "high = talkative, assertive, energetic, outgoing, sociable; low = quiet, reserved, withdrawn, prefers solitude",
    "agreeableness": "high = kind, cooperative, warm, trusting, considerate; low = cold, harsh, uncooperative, distrustful, blunt",
    "conscientiousness": "high = organized, responsible, thorough, planful, reliable; low = careless, disorganized, last-minute, sloppy, unreliable",
    "neuroticism": "high = anxious, tense, moody, irritable, worrying; low = calm, relaxed, even-tempered, emotionally stable",
    "openness": "high = creative, imaginative, curious, intellectual, inventive; low = conventional, incurious, unimaginative, unreflective",
}

# Layers already shown to steer judged behaviour from a low prior:
# extraversion 15 (rho 1.00, 15->93), neuroticism 20 (rho 1.00, 18->88),
# openness 19 (rho 0.89, 15->72). Kept so a working result is not traded away.
PROVEN_LAYERS = {"extraversion": 15, "neuroticism": 20, "openness": 19}

POLES = {
    # pole: (baseline ladder level, reference ladder level, sign)
    "up": (2, 9, +1),
    "down": (8, 1, -1),
}


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for pos, i in enumerate(order):
            out[i] = float(pos + 1)
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    return num / den if den else None


def pick_layer(
    trait: str,
    centroids: torch.Tensor,
    stack: torch.Tensor,
    acts: torch.Tensor | None,
    mode: str,
    override: int | None,
) -> tuple[int, str]:
    from app.persona.intensity_ladder import (
        monotone_fraction,
        resolve_steering_layer_for_direction,
        spearman_rho,
    )

    if override is not None:
        return override, "explicit override"
    if mode == "proven" and trait in PROVEN_LAYERS:
        return PROVEN_LAYERS[trait], "proven working layer"
    if mode in ("span", "proven"):
        layer, note, _ = resolve_steering_layer_for_direction(centroids, stack)
        return layer, note
    # snr: ordered span measured against same-level prompt-wording scatter
    n_layers = int(centroids.shape[1])
    lo, hi = int(0.3 * n_layers), int(0.8 * n_layers)
    levels = [float(i + 1) for i in range(int(centroids.shape[0]))]
    best = None
    for li in range(lo, hi):
        unit = stack[li].float()
        unit = unit / unit.norm().clamp_min(1e-9)
        proj = [float(torch.dot(c.float(), unit)) for c in centroids[:, li, :]]
        rho = spearman_rho(levels, proj) or 0.0
        mono = monotone_fraction(proj) or 0.0
        span = abs(proj[-1] - proj[0])
        if rho < 0.8 or mono < 0.6:
            continue
        snr = span
        if acts is not None:
            a = acts[:, :, li, :].float()
            pv = [[float(torch.dot(x, unit)) for x in a[i]] for i in range(a.shape[0])]
            scat = (
                sum((x - sum(r) / len(r)) ** 2 for r in pv for x in r)
                / (len(pv) * len(pv[0]))
            ) ** 0.5
            snr = span / scat if scat > 0 else span
        score = rho * snr
        if best is None or score > best[0]:
            best = (score, li)
    if best is None:
        layer, note, _ = resolve_steering_layer_for_direction(centroids, stack)
        return layer, note
    return best[1], "max rho x SNR in band"


def cmd_generate(args: argparse.Namespace) -> int:
    from app.persona.intensity_ladder import (
        _generate_probe,
        _load_model,
        _Steering,
        direction_span_magnitude,
        random_control_directions,
    )
    from app.persona.intensity_prompts import ladder_system_prompt
    from app.persona.inventory_ipip import TRAITS
    from app.persona.ocean_probes import PROBE_QUESTIONS, coherence_metrics, refusal_score

    traits = [t.strip() for t in args.traits.split(",") if t.strip()] or list(TRAITS)
    poles = [p.strip() for p in args.poles.split(",") if p.strip()]
    probes = list(PROBE_QUESTIONS[: max(1, args.probes)])
    overrides = {}
    for chunk in args.layers.split(","):
        if ":" in chunk:
            k, v = chunk.split(":", 1)
            overrides[k.strip()] = int(v)

    vdir = Path(args.vectors_dir)
    model, tokenizer, dev = _load_model(args.model_id or None, None)
    rows: list[dict] = []

    def emit(meta: dict, direction: torch.Tensor | None, mag: float, system: str) -> None:
        for qi, q in enumerate(probes):
            if direction is None or mag == 0.0:
                text = _generate_probe(
                    model, tokenizer, dev, system, q, max_new_tokens=args.max_new_tokens
                )
            else:
                with _Steering(model, meta["layer"], direction, mag):
                    text = _generate_probe(
                        model, tokenizer, dev, system, q,
                        max_new_tokens=args.max_new_tokens,
                    )
            coh = coherence_metrics(text)
            ref = refusal_score(text)
            rows.append(
                {
                    **meta,
                    "magnitude": round(float(mag), 4),
                    "abs_magnitude": round(abs(float(mag)), 4),
                    "question_idx": qi,
                    "question": q,
                    "system_prompt": system,
                    "text": text,
                    "coherent": bool(coh["coherent"]),
                    "type_token_ratio": coh["type_token_ratio"],
                    "refused": bool(ref["refused"]),
                }
            )
            logger.info(
                "[%s %s %s mag=%+.0f q%s] coh=%s ref=%s | %s",
                meta["trait"], meta["pole"], meta["kind"], mag, qi,
                coh["coherent"], ref["refused"], text[:70].replace("\n", " "),
            )

    for trait in traits:
        blob = torch.load(vdir / f"ladder_vectors_{trait}.pt", map_location="cpu")
        cents = blob["level_centroids"].float()
        key = "v_probe" if args.direction == "probe" and "v_probe" in blob else "v_pc1"
        stack = blob[key].float()
        acts = None
        cen_pt = vdir / f"centroids_{trait}.pt"
        if cen_pt.is_file():
            acts = torch.load(cen_pt, map_location="cpu")["activations"]
        layer, note = pick_layer(
            trait, cents, stack, acts, args.layer_select, overrides.get(trait)
        )
        v = stack[layer]
        v = v / v.norm().clamp_min(1e-9)
        span = direction_span_magnitude(cents, layer, v)
        mults = [0.25, 0.5, 1.0, 1.5, 2.0][: args.rungs]
        logger.info(
            "[%s] layer %s (%s) key=%s span=%.1f grid=%s",
            trait, layer, note, key, span, [round(span * m) for m in mults],
        )
        controls = random_control_directions(
            int(v.shape[0]), args.controls, seed=args.seed, like=v
        )
        for pole in poles:
            base_level, ref_level, sign = POLES[pole]
            base_system = ladder_system_prompt(trait, base_level, n_markers=3)
            meta = {
                "trait": trait,
                "pole": pole,
                "layer": layer,
                "layer_note": note,
                "direction_key": key,
                "span": round(span, 2),
                "baseline_level": base_level,
                "reference_level": ref_level,
            }
            emit({**meta, "kind": "trait"}, v, 0.0, base_system)
            for m in mults:
                emit({**meta, "kind": "trait"}, v, sign * span * m, base_system)
            for ci, cv in enumerate(controls):
                for m in mults:
                    emit(
                        {**meta, "kind": f"control{ci}"},
                        cv.float(),
                        sign * span * m,
                        base_system,
                    )
            emit(
                {**meta, "kind": "prompted_reference"},
                None,
                0.0,
                ladder_system_prompt(trait, ref_level, n_markers=3),
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s (%s rows)", out, len(rows))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from app.persona.judge_vertex import score_transcript

    rows = json.loads(Path(args.generations).read_text())
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    for n, idx in enumerate(order, 1):
        r = rows[idx]
        try:
            js = score_transcript(
                (
                    f"Score ONLY the assistant reply for Big Five {r['trait']}. "
                    f"{TRAITS_RUBRIC[r['trait']]}. "
                    "0 = extreme low pole, 50 = neutral, 100 = extreme high pole. "
                    "Ignore whether the speaker is an AI. Do not reward naming the trait. "
                    "If the reply is a refusal or incoherent, score 50."
                ),
                "",
                r["question"],
                r["text"],
                project_id=args.project,
                model_name=args.judge_model,
                max_output_tokens=2048,
            )
            r["judge_score"], r["judge_reason"] = int(js.score), js.short_reason
        except Exception as exc:
            logger.warning("judge failed: %s", exc)
            r["judge_score"], r["judge_reason"] = None, str(exc)[:160]
        logger.info(
            "[%s/%s] %s %s %s mag=%+.0f -> %s",
            n, len(order), r["trait"], r["pole"], r["kind"], r["magnitude"],
            r["judge_score"],
        )

    def usable(r: dict) -> bool:
        return r["judge_score"] is not None and not r["refused"] and r["coherent"]

    def curve(rs: list[dict]) -> dict:
        pts = []
        for mag in sorted({r["abs_magnitude"] for r in rs}):
            vals = [r["judge_score"] for r in rs if r["abs_magnitude"] == mag and usable(r)]
            drop = sum(1 for r in rs if r["abs_magnitude"] == mag and not usable(r))
            if vals:
                pts.append(
                    {
                        "abs_magnitude": mag,
                        "judge_mean": round(sum(vals) / len(vals), 2),
                        "n": len(vals),
                        "n_dropped": drop,
                    }
                )
        rho = spearman([p["abs_magnitude"] for p in pts], [p["judge_mean"] for p in pts])
        return {
            "points": pts,
            "rho_absmag_vs_judge": None if rho is None else round(rho, 4),
            "start": pts[0]["judge_mean"] if pts else None,
            "end": pts[-1]["judge_mean"] if pts else None,
        }

    table = []
    for trait in sorted({r["trait"] for r in rows}):
        for pole in sorted({r["pole"] for r in rows if r["trait"] == trait}):
            sel = [r for r in rows if r["trait"] == trait and r["pole"] == pole]
            want_up = pole == "up"
            ref = [r["judge_score"] for r in sel if r["kind"] == "prompted_reference" and usable(r)]
            tr = curve([r for r in sel if r["kind"] == "trait"])
            ctrls = {
                k: curve([r for r in sel if r["kind"] == k])
                for k in sorted({r["kind"] for r in sel if r["kind"].startswith("control")})
            }
            base = tr["start"]
            reference = round(sum(ref) / len(ref), 2) if ref else None
            # extreme judge score in the intended direction
            vals = [p["judge_mean"] for p in tr["points"]]
            extreme = max(vals) if want_up else min(vals)
            gap = (reference - base) if (reference is not None and base is not None) else None
            closed = (
                round((extreme - base) / gap * 100, 1)
                if gap not in (None, 0)
                else None
            )
            ctrl_extremes = [
                (max(p["judge_mean"] for p in c["points"]) if want_up
                 else min(p["judge_mean"] for p in c["points"]))
                for c in ctrls.values() if c["points"]
            ]
            ctrl_move = (
                max(abs(c - base) for c in ctrl_extremes) if ctrl_extremes and base is not None else None
            )
            trait_move = abs(extreme - base) if base is not None else None
            margin = (
                round(trait_move / ctrl_move, 2)
                if ctrl_move not in (None, 0) and trait_move is not None
                else (999.0 if ctrl_move == 0 else None)
            )
            rho = tr["rho_absmag_vs_judge"]
            sign_ok = None
            if rho is not None:
                sign_ok = rho > 0 if want_up else rho < 0
            table.append(
                {
                    "trait": trait,
                    "pole": pole,
                    "layer": sel[0]["layer"],
                    "span": sel[0]["span"],
                    "judge_baseline": base,
                    "judge_reference": reference,
                    "judge_extreme": extreme,
                    "pct_of_prompt_gap": closed,
                    "rho": rho,
                    "sign_correct": sign_ok,
                    "control_margin": margin,
                    "trait_curve": tr,
                    "control_curves": ctrls,
                    "works": bool(
                        rho is not None
                        and sign_ok
                        and abs(rho) >= 0.7
                        and margin is not None
                        and margin >= 2.0
                        and closed is not None
                        and closed >= 40
                    ),
                }
            )

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "bipolar_judge",
        "judge_model": args.judge_model,
        "judge_project": args.project,
        "n_rows": len(rows),
        "table": table,
        "scored": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 112)
    print("DOES ONE VECTOR SCALE THE TRAIT UP *AND* DOWN?  (blind judge, 0-100)")
    print("=" * 112)
    print(
        f"{'trait':17}{'pole':6}{'L':>4}{'base':>6}{'->extreme':>10}{'prompt ref':>11}"
        f"{'%gap':>7}{'rho':>7}{'ctrl x':>8}{'works':>7}"
    )
    for e in table:
        print(
            f"{e['trait']:17}{e['pole']:6}{e['layer']:>4}{str(e['judge_baseline']):>6}"
            f"{str(e['judge_extreme']):>10}{str(e['judge_reference']):>11}"
            f"{str(e['pct_of_prompt_gap']):>7}{str(e['rho']):>7}"
            f"{str(e['control_margin']):>8}{str(e['works']):>7}"
        )
    print("\ntrajectories (|magnitude|: judge):")
    for e in table:
        traj = "  ".join(
            f"{p['abs_magnitude']:.0f}:{p['judge_mean']:.0f}" for p in e["trait_curve"]["points"]
        )
        print(f"  {e['trait']:17}{e['pole']:6} {traj}")
    print(f"\nwrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--vectors-dir", required=True)
    g.add_argument("--model-id", default="")
    g.add_argument("--traits", default="")
    g.add_argument("--poles", default="up,down")
    g.add_argument("--direction", default="pc1", choices=("pc1", "probe"))
    g.add_argument(
        "--layer-select",
        default="proven",
        choices=("proven", "span", "snr"),
        help="proven = keep layers already shown to work, else largest ordered span.",
    )
    g.add_argument("--layers", default="", help="Override, e.g. agreeableness:21,openness:19")
    g.add_argument("--rungs", type=int, default=5)
    g.add_argument("--controls", type=int, default=1)
    g.add_argument("--probes", type=int, default=3)
    g.add_argument("--max-new-tokens", type=int, default=90)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    j = sub.add_parser("judge")
    j.add_argument("--generations", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--project", default="project-amer-scs-sandbox")
    j.add_argument("--judge-model", default="gemini-2.5-flash")
    j.add_argument("--seed", type=int, default=0)
    j.set_defaults(func=cmd_judge)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
