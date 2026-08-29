#!/usr/bin/env python3
"""Away-from-prior test: can a vector push a trait UP from a low baseline?

Every trait except neuroticism already reads high to a judge at baseline
(agreeableness 90, conscientiousness 86, extraversion 80, openness 75 out of
100), leaving 10-25 points of headroom up against 75-90 down. Any claim that
steering "mostly moves traits down" is therefore a ceiling artifact, not a
property of the vectors.

This sets a *negative prior* first — the low-pole ladder prompt — so there is
~85 points of room up, then steers toward the high pole and judges the free
text. Matched random controls run at the same baseline, because generic
perturbation is what a judge most easily mistakes for a trait (steered
neuroticism's apparent climb came from a refusal and from degenerate text).

Stage 1 (GPU): generate. Stage 2 (CPU + Vertex): judge.

    python3 scripts/negative_prior_judge.py generate --out gen.json
    python3 scripts/negative_prior_judge.py judge --generations gen.json --out summary.json
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

logger = logging.getLogger("negative_prior")

TRAITS_RUBRIC = {
    "extraversion": "high = talkative, assertive, energetic, outgoing, sociable; low = quiet, reserved, withdrawn, prefers solitude",
    "agreeableness": "high = kind, cooperative, warm, trusting, considerate; low = cold, harsh, uncooperative, distrustful, blunt",
    "conscientiousness": "high = organized, responsible, thorough, planful, reliable; low = careless, disorganized, last-minute, sloppy, unreliable",
    "neuroticism": "high = anxious, tense, moody, irritable, worrying; low = calm, relaxed, even-tempered, emotionally stable",
    "openness": "high = creative, imaginative, curious, intellectual, inventive; low = conventional, incurious, unimaginative, unreflective",
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


def cmd_generate(args: argparse.Namespace) -> int:
    from app.persona.intensity_ladder import (
        _generate_probe,
        _load_model,
        _Steering,
        latent_span_magnitude,
        random_control_directions,
        resolve_steering_layer,
    )
    from app.persona.intensity_prompts import ladder_system_prompt
    from app.persona.inventory_ipip import TRAITS
    from app.persona.ocean_probes import PROBE_QUESTIONS, coherence_metrics, refusal_score

    traits = [t.strip() for t in args.traits.split(",") if t.strip()] or list(TRAITS)
    probes = list(PROBE_QUESTIONS[: max(1, args.probes)])
    vdir = Path(args.vectors_dir)
    model, tokenizer, dev = _load_model(args.model_id or None, None)

    rows: list[dict] = []

    def emit(trait, kind, mag, direction, system, tag):
        for qi, q in enumerate(probes):
            if direction is None or mag == 0.0:
                text = _generate_probe(
                    model, tokenizer, dev, system, q, max_new_tokens=args.max_new_tokens
                )
            else:
                with _Steering(model, tag["layer"], direction, mag):
                    text = _generate_probe(
                        model, tokenizer, dev, system, q,
                        max_new_tokens=args.max_new_tokens,
                    )
            coh = coherence_metrics(text)
            ref = refusal_score(text)
            rows.append(
                {
                    "trait": trait,
                    "kind": kind,
                    "magnitude": round(float(mag), 4),
                    "baseline_level": tag["baseline_level"],
                    "layer": tag["layer"],
                    "question_idx": qi,
                    "question": q,
                    "system_prompt": system,
                    "text": text,
                    "coherent": coh["coherent"],
                    "type_token_ratio": coh["type_token_ratio"],
                    "refused": bool(ref["refused"]),
                }
            )
            logger.info(
                "[%s %s mag=%.0f q%s] coh=%s ref=%s | %s",
                trait, kind, mag, qi, coh["coherent"], ref["refused"],
                text[:80].replace("\n", " "),
            )

    for trait in traits:
        blob = torch.load(vdir / f"ladder_vectors_{trait}.pt", map_location="cpu")
        n_layers = int(blob["v_pc1"].shape[0])
        layer, note = resolve_steering_layer(blob["geometry"], n_layers)
        key = "v_probe" if args.direction == "probe" and "v_probe" in blob else "v_pc1"
        v = blob[key][layer].float()
        v = v / v.norm().clamp_min(1e-9)
        cents = blob["level_centroids"].float()
        scale = float(cents[:, layer, :].norm(dim=-1).mean())
        span = latent_span_magnitude(blob["geometry"], layer) or (0.03 * scale)
        grid = [0.0] + [span * m for m in (0.25, 0.5, 1.0, 1.5, 2.0)][: args.rungs]
        tag = {"layer": layer, "baseline_level": args.baseline_level}
        # Negative prior: the low-pole ladder prompt as the baseline persona.
        system = ladder_system_prompt(trait, args.baseline_level, n_markers=3)
        logger.info(
            "[%s] layer %s (%s) key=%s span=%.1f grid=%s baseline_level=%s",
            trait, layer, note, key, span, [round(g) for g in grid], args.baseline_level,
        )
        for mag in grid:
            emit(trait, "trait", mag, v, system, tag)
        controls = random_control_directions(
            int(v.shape[0]), args.controls, seed=args.seed, like=v
        )
        for ci, cv in enumerate(controls):
            for mag in grid:
                if mag == 0.0:
                    continue
                emit(trait, f"control{ci}", mag, cv.float(), system, tag)
        # Reference ceiling: what the high-pole prompt itself achieves.
        emit(
            trait,
            "prompted_high",
            0.0,
            None,
            ladder_system_prompt(trait, 9, n_markers=3),
            {"layer": layer, "baseline_level": 9},
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
                    "If the reply is a refusal or is incoherent, score 50."
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
            "[%s/%s] %s %s mag=%.0f -> %s",
            n, len(order), r["trait"], r["kind"], r["magnitude"], r["judge_score"],
        )

    def usable(r: dict) -> bool:
        return (
            r["judge_score"] is not None
            and not r["refused"]
            and bool(r["coherent"])
        )

    table = []
    for trait in sorted({r["trait"] for r in rows}):
        tr = [r for r in rows if r["trait"] == trait]
        prompted = [r["judge_score"] for r in tr if r["kind"] == "prompted_high" and usable(r)]
        base = [r["judge_score"] for r in tr if r["kind"] == "trait" and r["magnitude"] == 0 and usable(r)]
        entry = {
            "trait": trait,
            "baseline_level": tr[0]["baseline_level"],
            "judge_baseline": round(sum(base) / len(base), 2) if base else None,
            "judge_prompted_high": round(sum(prompted) / len(prompted), 2) if prompted else None,
            "curves": {},
        }
        for kind in sorted({r["kind"] for r in tr if r["kind"].startswith(("trait", "control"))}):
            pts = []
            for mag in sorted({r["magnitude"] for r in tr if r["kind"] == kind}):
                vals = [r["judge_score"] for r in tr if r["kind"] == kind and r["magnitude"] == mag and usable(r)]
                n_drop = sum(1 for r in tr if r["kind"] == kind and r["magnitude"] == mag and not usable(r))
                if vals:
                    pts.append({
                        "magnitude": mag,
                        "judge_mean": round(sum(vals) / len(vals), 2),
                        "n": len(vals),
                        "n_dropped": n_drop,
                    })
            rho = spearman([p["magnitude"] for p in pts], [p["judge_mean"] for p in pts])
            entry["curves"][kind] = {
                "points": pts,
                "rho": None if rho is None else round(rho, 4),
                "delta": (
                    round(pts[-1]["judge_mean"] - pts[0]["judge_mean"], 2) if len(pts) >= 2 else None
                ),
                "peak": max((p["judge_mean"] for p in pts), default=None),
            }
        table.append(entry)

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "negative_prior_judge",
        "judge_model": args.judge_model,
        "judge_project": args.project,
        "n_rows": len(rows),
        "table": table,
        "scored": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 104)
    print("FROM A LOW-PRIOR BASELINE, DOES THE VECTOR PUSH THE JUDGE UP?")
    print("=" * 104)
    for e in table:
        tv = e["curves"].get("trait", {})
        print(f"\n{e['trait']}  baseline(level {e['baseline_level']})={e['judge_baseline']}  "
              f"prompted_high={e['judge_prompted_high']}")
        for kind, c in e["curves"].items():
            traj = "  ".join(f"{p['magnitude']:.0f}:{p['judge_mean']:.0f}" for p in c["points"])
            print(f"   {kind:10} rho={str(c['rho']):>7} delta={str(c['delta']):>7} peak={str(c['peak']):>5}  {traj}")
    print(f"\nwrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="GPU stage")
    g.add_argument("--out", required=True)
    g.add_argument("--vectors-dir", required=True)
    g.add_argument("--model-id", default="")
    g.add_argument("--traits", default="")
    g.add_argument("--direction", default="pc1", choices=("pc1", "probe"))
    g.add_argument("--baseline-level", type=int, default=2, help="Low-pole ladder level.")
    g.add_argument("--rungs", type=int, default=5)
    g.add_argument("--controls", type=int, default=1)
    g.add_argument("--probes", type=int, default=3)
    g.add_argument("--max-new-tokens", type=int, default=90)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    j = sub.add_parser("judge", help="Vertex stage")
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
