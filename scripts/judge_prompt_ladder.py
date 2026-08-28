#!/usr/bin/env python3
"""Blind LLM-as-judge: does prompting produce *behavioural* trait graduation?

The inventory ladder is the model grading itself. This generates free text under
each of the nine intensity prompts, shuffles away level labels, and has a
*different* model score each reply on the target trait. The statistic that
matters is Spearman(prompted level, judge score) per trait.

No Vertex / API keys required: the judge defaults to an ungated open-weight
model loaded after the subject is unloaded, so both fit on one L4 sequentially.

    python3 scripts/judge_prompt_ladder.py --out results/judge_ladder/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("judge_ladder")

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
DEFAULT_VERTEX_PROJECT = "project-amer-scs-sandbox"


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


def _trait_rubric(trait: str) -> str:
    poles = {
        "extraversion": (
            "high = talkative, assertive, energetic, outgoing, sociable; "
            "low = quiet, reserved, withdrawn, prefers solitude"
        ),
        "agreeableness": (
            "high = kind, cooperative, warm, trusting, considerate; "
            "low = cold, harsh, uncooperative, distrustful, blunt"
        ),
        "conscientiousness": (
            "high = organized, responsible, thorough, planful, reliable; "
            "low = careless, disorganized, last-minute, sloppy, unreliable"
        ),
        "neuroticism": (
            "high = anxious, tense, moody, irritable, worrying; "
            "low = calm, relaxed, even-tempered, emotionally stable"
        ),
        "openness": (
            "high = creative, imaginative, curious, intellectual, inventive; "
            "low = conventional, incurious, unimaginative, unreflective"
        ),
    }
    return poles[trait]


def build_judge_prompt(trait: str, question: str, answer: str) -> str:
    return f"""You are scoring a short self-description for one Big Five trait: {trait}.

Scale for {trait}: {_trait_rubric(trait)}

Score ONLY the ASSISTANT answer below. Ignore whether the speaker is an AI.
Do not reward mentioning the trait by name. Score expressed behaviour and
preferences only.

Question: {question}

Answer: {answer}

Return ONLY valid JSON with keys "score" (integer 0-100; 0 = extreme low pole,
50 = neutral, 100 = extreme high pole) and "short_reason" (one short sentence,
no quotes inside)."""


def parse_judge_json(text: str) -> tuple[int | None, str]:
    t = text.strip()
    m = re.search(r"\{[^{}]*\}", t, re.DOTALL)
    if not m:
        return None, t[:120]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, t[:120]
    score = obj.get("score")
    try:
        score_i = int(score)
    except (TypeError, ValueError):
        return None, str(obj.get("short_reason", ""))[:120]
    score_i = max(0, min(100, score_i))
    reason = str(obj.get("short_reason", ""))[:120]
    return score_i, reason


@torch.no_grad()
def generate_reply(
    model, tokenizer, device, system: str, question: str, *, max_new_tokens: int
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    # Gemma-3 chat template rejects a system role in some builds; fold it in.
    try:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        messages = [
            {
                "role": "user",
                "content": f"{system}\n\n{question}",
            }
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, ids.shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--subject-model", default="")
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Vertex model id (default: gemini-2.5-flash).",
    )
    p.add_argument(
        "--judge-backend",
        default="vertex",
        choices=("vertex", "hf"),
        help="vertex = Gemini via Vertex AI; hf = a local HuggingFace model.",
    )
    p.add_argument(
        "--project",
        default=DEFAULT_VERTEX_PROJECT,
        help="GCP project for Vertex judge (default: project-amer-scs-sandbox).",
    )
    p.add_argument("--traits", default="")
    p.add_argument("--levels", default="1,2,3,4,5,6,7,8,9")
    p.add_argument("--variants", type=int, default=1, help="Marker rotations per level.")
    p.add_argument("--probes", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--self-judge",
        action="store_true",
        help="Reuse the subject as judge (weaker; for when a second model will not fit).",
    )
    p.add_argument(
        "--generations-only",
        action="store_true",
        help="Stop after writing generations.json next to --out.",
    )
    p.add_argument(
        "--judge-only",
        default="",
        help="Path to an existing generations.json; skip subject generation.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.activations import load_model_and_tokenizer
    from app.persona.intensity_prompts import ladder_system_prompt
    from app.persona.inventory_ipip import TRAITS
    from app.persona.ocean_probes import PROBE_QUESTIONS

    traits = [t.strip() for t in args.traits.split(",") if t.strip()] or list(TRAITS)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    probes = list(PROBE_QUESTIONS[: max(1, args.probes)])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path = out_path.parent / "generations.json"

    if args.judge_only:
        generations = json.loads(Path(args.judge_only).read_text())
        logger.info("loaded %s generations from %s", len(generations), args.judge_only)
    else:
        model, tokenizer, device = load_model_and_tokenizer(
            args.subject_model or None
        )
        generations = []
        for trait in traits:
            for level in levels:
                for variant in range(args.variants):
                    system = ladder_system_prompt(
                        trait, level, n_markers=3, variant=variant
                    )
                    for qi, question in enumerate(probes):
                        text = generate_reply(
                            model,
                            tokenizer,
                            device,
                            system,
                            question,
                            max_new_tokens=args.max_new_tokens,
                        )
                        generations.append(
                            {
                                "id": f"{trait}-L{level}-v{variant}-q{qi}",
                                "trait": trait,
                                "level": level,
                                "variant": variant,
                                "question_idx": qi,
                                "question": question,
                                "system_prompt": system,
                                "text": text,
                            }
                        )
                        logger.info(
                            "[%s L%s v%s q%s] %s",
                            trait,
                            level,
                            variant,
                            qi,
                            text[:100].replace("\n", " "),
                        )
        gen_path.write_text(json.dumps(generations, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s (%s generations)", gen_path, len(generations))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.generations_only:
            return 0

    # Blind shuffle before judging so the judge never sees sequential levels.
    rng = random.Random(args.seed)
    order = list(range(len(generations)))
    rng.shuffle(order)

    judge_id = args.subject_model if args.self_judge else args.judge_model
    backend = "hf" if args.self_judge else args.judge_backend
    if args.self_judge:
        logger.warning("self-judge mode: subject scores its own replies (weaker evidence)")

    judge = jtok = jdev = None
    if backend == "hf":
        judge, jtok, jdev = load_model_and_tokenizer(judge_id or None)
    else:
        from app.persona.judge_vertex import score_transcript as vertex_score

        logger.info(
            "Vertex judge model=%s project=%s", judge_id, args.project
        )

    scored = []
    for idx in order:
        g = generations[idx]
        prompt = build_judge_prompt(g["trait"], g["question"], g["text"])
        if backend == "vertex":
            try:
                js = vertex_score(
                    (
                        f"Score ONLY the assistant reply for Big Five {g['trait']}. "
                        f"{_trait_rubric(g['trait'])}. "
                        "0 = extreme low pole, 50 = neutral, 100 = extreme high pole. "
                        "Ignore whether the speaker is an AI. Do not reward naming the trait."
                    ),
                    "",
                    g["question"],
                    g["text"],
                    project_id=args.project,
                    model_name=judge_id,
                    max_output_tokens=2048,
                )
                raw = json.dumps({"score": js.score, "short_reason": js.short_reason})
                score, reason = int(js.score), js.short_reason
            except Exception as exc:  # keep going; one failed item shouldn't sink the rho
                logger.warning("vertex judge failed on %s: %s", g["id"], exc)
                raw, score, reason = str(exc)[:300], None, str(exc)[:120]
        else:
            raw = generate_reply(
                judge, jtok, jdev, "You output JSON only.", prompt, max_new_tokens=80
            )
            score, reason = parse_judge_json(raw)
        row = {
            **{k: g[k] for k in ("id", "trait", "level", "variant", "question_idx", "question", "text")},
            "judge_score": score,
            "judge_reason": reason,
            "judge_raw": raw[:300],
            "judge_model": judge_id,
            "judge_backend": backend,
        }
        scored.append(row)
        logger.info(
            "judge %s L%s -> %s (%s)",
            g["trait"],
            g["level"],
            score,
            reason,
        )

    # Aggregate.
    by_trait: dict[str, list[dict]] = {}
    for row in scored:
        by_trait.setdefault(row["trait"], []).append(row)

    summary_traits = []
    for trait, rows in by_trait.items():
        usable = [r for r in rows if r["judge_score"] is not None]
        # Mean score per level across probes/variants.
        level_means: dict[int, float] = {}
        for level in sorted({r["level"] for r in usable}):
            vals = [r["judge_score"] for r in usable if r["level"] == level]
            level_means[level] = sum(vals) / len(vals)
        xs = list(level_means.keys())
        ys = [level_means[x] for x in xs]
        rho = spearman([float(x) for x in xs], ys)
        # Also item-level rho (every scored generation).
        rho_item = spearman(
            [float(r["level"]) for r in usable],
            [float(r["judge_score"]) for r in usable],
        )
        summary_traits.append(
            {
                "trait": trait,
                "n_scored": len(usable),
                "n_failed_parse": sum(1 for r in rows if r["judge_score"] is None),
                "level_mean_score": {str(k): round(v, 2) for k, v in level_means.items()},
                "score_range": (
                    [round(min(ys), 2), round(max(ys), 2)] if ys else None
                ),
                "spearman_level_vs_mean_score": round(rho, 4) if rho is not None else None,
                "spearman_level_vs_item_score": (
                    round(rho_item, 4) if rho_item is not None else None
                ),
                "graded": bool(rho is not None and rho >= 0.6),
            }
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "judge_prompt_ladder",
        "subject_model": args.subject_model or None,
        "judge_model": judge_id,
        "judge_backend": backend,
        "judge_project": args.project if backend == "vertex" else None,
        "n_levels": len(levels),
        "n_probes": len(probes),
        "n_variants": args.variants,
        "n_generations": len(generations),
        "traits": summary_traits,
        "scored": scored,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 88)
    print("DOES A BLIND JUDGE SEE GRADED TRAIT CHANGE ACROSS PROMPT LEVELS?")
    print("=" * 88)
    print(
        f"{'trait':18} {'rho(level,mean)':>15} {'rho(item)':>10} "
        f"{'range':>12} {'n':>5} {'graded?':>8}"
    )
    for t in summary_traits:
        rng = (
            f"{t['score_range'][0]}-{t['score_range'][1]}"
            if t["score_range"]
            else "-"
        )
        print(
            f"{t['trait']:18} {str(t['spearman_level_vs_mean_score']):>15} "
            f"{str(t['spearman_level_vs_item_score']):>10} {rng:>12} "
            f"{t['n_scored']:>5} {str(t['graded']):>8}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
