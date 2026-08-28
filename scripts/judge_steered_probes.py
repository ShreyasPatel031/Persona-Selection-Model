#!/usr/bin/env python3
"""Blind Gemini judge on *steered* free text, not prompted text.

Loads probe replies already saved in validated-sweep JSON (trait vector and
matched random controls), shuffles them, scores each with Vertex Gemini on the
target trait, and reports Spearman(signed magnitude, judge score) per pole.

    PYTHONPATH=. python3 scripts/judge_steered_probes.py \
      --sweeps-dir results/gemma_final \
      --out results/judge_vectors/summary.json \
      --project project-amer-scs-sandbox
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("judge_vectors")


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


TRAITS_RUBRIC = {
    "extraversion": "high = talkative, assertive, energetic, outgoing, sociable; low = quiet, reserved, withdrawn, prefers solitude",
    "agreeableness": "high = kind, cooperative, warm, trusting, considerate; low = cold, harsh, uncooperative, distrustful, blunt",
    "conscientiousness": "high = organized, responsible, thorough, planful, reliable; low = careless, disorganized, last-minute, sloppy, unreliable",
    "neuroticism": "high = anxious, tense, moody, irritable, worrying; low = calm, relaxed, even-tempered, emotionally stable",
    "openness": "high = creative, imaginative, curious, intellectual, inventive; low = conventional, incurious, unimaginative, unreflective",
}


def collect_items(sweeps_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(sweeps_dir.glob("validated_sweep_*.json")):
        d = json.loads(path.read_text())
        trait = d["trait"]
        toward = d["verdict"]["steered_toward"]
        for kind, curve in [("trait", d["trait_curve"])] + [
            (c.get("label") or "control", c) for c in (d.get("control_curves") or [])
        ]:
            for rung_i, row in enumerate(curve.get("rows") or []):
                for pi, probe in enumerate(row.get("probes") or []):
                    text = (probe.get("text") or "").strip()
                    if not text:
                        continue
                    items.append(
                        {
                            "id": f"{path.stem}:{kind}:r{rung_i}:p{pi}",
                            "file": path.name,
                            "trait": trait,
                            "toward": toward,
                            "kind": "trait" if kind == "trait" else "control",
                            "magnitude": row["magnitude"],
                            "alpha": row.get("alpha"),
                            "inventory_ev": row.get("target_ev"),
                            "usable": row.get("usable"),
                            "coherent": (probe.get("coherence") or {}).get("coherent"),
                            "question": probe.get("question"),
                            "text": text,
                        }
                    )
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps-dir", default=str(REPO_ROOT / "results" / "gemma_final"))
    p.add_argument("--out", required=True)
    p.add_argument("--project", default="project-amer-scs-sandbox")
    p.add_argument("--judge-model", default="gemini-2.5-flash")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.judge_vertex import score_transcript

    items = collect_items(Path(args.sweeps_dir))
    logger.info("collected %s steered texts from %s", len(items), args.sweeps_dir)
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)

    scored = []
    for n, idx in enumerate(order, 1):
        it = items[idx]
        try:
            js = score_transcript(
                (
                    f"Score ONLY the assistant reply for Big Five {it['trait']}. "
                    f"{TRAITS_RUBRIC[it['trait']]}. "
                    "0 = extreme low pole, 50 = neutral, 100 = extreme high pole. "
                    "Ignore whether the speaker is an AI. Do not reward naming the trait."
                ),
                "",
                it["question"] or "Describe yourself.",
                it["text"],
                project_id=args.project,
                model_name=args.judge_model,
                max_output_tokens=2048,
            )
            score, reason = int(js.score), js.short_reason
        except Exception as exc:
            logger.warning("judge failed %s: %s", it["id"], exc)
            score, reason = None, str(exc)[:160]
        row = {**it, "judge_score": score, "judge_reason": reason}
        scored.append(row)
        logger.info(
            "[%s/%s] %s %s mag=%s %s -> %s",
            n,
            len(order),
            it["trait"],
            it["kind"],
            it["magnitude"],
            it["toward"],
            score,
        )

    # Per trait × pole × kind: rho(signed magnitude, judge score)
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in scored:
        if row["judge_score"] is None:
            continue
        buckets[(row["trait"], row["toward"], row["kind"])].append(row)

    table = []
    for (trait, toward, kind), rows in sorted(buckets.items()):
        rows = sorted(rows, key=lambda r: r["magnitude"])
        xs = [float(r["magnitude"]) for r in rows]
        ys = [float(r["judge_score"]) for r in rows]
        rho = spearman(xs, ys)
        # expected: high pole rho>0, low pole rho<0 for the trait vector
        sign_ok = None
        if rho is not None:
            sign_ok = bool(rho > 0) if toward == "high" else bool(rho < 0)
        table.append(
            {
                "trait": trait,
                "toward": toward,
                "kind": kind,
                "n": len(rows),
                "rho_magnitude_vs_judge": None if rho is None else round(rho, 4),
                "sign_correct": sign_ok if kind == "trait" else None,
                "scores_by_mag": [
                    {
                        "magnitude": r["magnitude"],
                        "judge": r["judge_score"],
                        "inventory_ev": r["inventory_ev"],
                        "usable": r["usable"],
                    }
                    for r in rows
                ],
                "judge_range": [min(ys), max(ys)],
                "inventory_range": [
                    min(r["inventory_ev"] for r in rows if r["inventory_ev"] is not None),
                    max(r["inventory_ev"] for r in rows if r["inventory_ev"] is not None),
                ],
            }
        )

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "judge_steered_probes",
        "sweeps_dir": str(Path(args.sweeps_dir).resolve()),
        "judge_model": args.judge_model,
        "judge_project": args.project,
        "n_texts": len(items),
        "n_scored": sum(1 for r in scored if r["judge_score"] is not None),
        "table": table,
        "scored": scored,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 100)
    print("DO THE VECTORS MOVE A BLIND LLM JUDGE?  rho(signed mag, judge)  expected + on high / − on low")
    print("=" * 100)
    print(f"{'trait':18} {'pole':5} {'kind':8} {'n':>3} {'rho':>8} {'sign':>6} {'judge range':>14} {'inv range':>14}")
    for t in table:
        jr = f"{t['judge_range'][0]:.0f}-{t['judge_range'][1]:.0f}"
        ir = f"{t['inventory_range'][0]:.2f}-{t['inventory_range'][1]:.2f}"
        print(
            f"{t['trait']:18} {t['toward']:5} {t['kind']:8} {t['n']:>3} "
            f"{str(t['rho_magnitude_vs_judge']):>8} {str(t['sign_correct']):>6} {jr:>14} {ir:>14}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
