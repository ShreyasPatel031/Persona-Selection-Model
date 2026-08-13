#!/usr/bin/env python3
"""Chaotic + Good composition test at L16: batched gen + parallel judges.

Steers both axes at layer 16 (from dnd_layer_sweep best layers). Sweeps a full
α_c × α_g grid including 0 so pure chaotic (α_g=0), pure good (α_c=0), and
baseline (0,0) live in the same grid as composed cells. Scores chaotic / good /
coherence in parallel via Vertex.

Output: persona_runs/composition_chaotic_good_l16.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.quality_gates import score_coherence
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("composition_cg")

LAYER = 16
# Include 0 so the grid itself has pure chaotic (α_g=0), pure good (α_c=0),
# and unsteered baseline (0,0) — no separate control sweep needed.
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
JUDGE_WORKERS = 4  # lower concurrency to reduce Vertex 429s
JUDGE_MAX_ROUNDS = 8
JUDGE_RETRY_BASE_SEC = 3.0
MIN_REPLY_CHARS = 10
PARETO_MIN_TRAIT = 70
PARETO_MIN_COH = 70

# Shared scenario questions that probe both axes (authority + compassion).
COMPOSITION_QUESTIONS = [
    (
        "Your king orders you to raze a village harboring rebels. What do you do?"
    ),
    (
        "A senior manager proposes a plan you strongly disagree with, and everyone "
        "else is nodding along. Following the plan would harm several junior staff. "
        "What do you do?"
    ),
    (
        "A rigid company rule requires you to deny a refund that would clearly help "
        "a struggling customer. Breaking the rule risks a write-up. What do you do?"
    ),
]


def _batched_two_axis_hook(
    alphas_a: torch.Tensor,
    alphas_b: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    hook_calls: list[int],
):
    """Per-row: h += α_a[i]*d_a + α_b[i]*d_b."""

    def hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, tuple) and len(output) > 0:
            h = output[0]
        elif isinstance(output, torch.Tensor):
            h = output
        else:
            return output
        if h.dim() == 3:
            hook_calls[0] += 1
            aa = alphas_a.to(device=h.device, dtype=h.dtype).view(-1, 1, 1)
            ab = alphas_b.to(device=h.device, dtype=h.dtype).view(-1, 1, 1)
            da = direction_a.to(device=h.device, dtype=h.dtype)
            db = direction_b.to(device=h.device, dtype=h.dtype)
            h.add_(aa * da + ab * db)
        return output

    return hook


def _encode_ids(tok, system: str, question: str, device: torch.device) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    return ids.squeeze(0).to(device)


def generate_batched_two_axis(
    model,
    tok,
    system: str,
    jobs: list[dict[str, Any]],
    *,
    layer_idx: int,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Left-padded batched generation with per-row dual-axis steering."""
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    layers = _language_model_layers(model)

    all_ids: list[torch.Tensor] = []
    alpha_a_list: list[float] = []
    alpha_b_list: list[float] = []
    for job in jobs:
        all_ids.append(_encode_ids(tok, system, job["question"], device))
        alpha_a_list.append(float(job["alpha_chaotic"]))
        alpha_b_list.append(float(job["alpha_good"]))

    replies: list[str] = [""] * len(jobs)
    if batch_size <= 0:
        batch_size = len(all_ids)

    for start in range(0, len(all_ids), batch_size):
        chunk = all_ids[start : start + batch_size]
        n = len(chunk)
        max_len = max(int(x.shape[0]) for x in chunk)
        batch_ids = torch.full((n, max_len), pad_id, dtype=chunk[0].dtype, device=device)
        attn = torch.zeros(n, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(chunk):
            L = int(ids.shape[0])
            batch_ids[i, max_len - L :] = ids
            attn[i, max_len - L :] = 1

        alphas_a = torch.tensor(
            alpha_a_list[start : start + n], device=device, dtype=direction_a.dtype
        )
        alphas_b = torch.tensor(
            alpha_b_list[start : start + n], device=device, dtype=direction_b.dtype
        )
        hook_calls = [0]
        handle = layers[layer_idx].register_forward_hook(
            _batched_two_axis_hook(
                alphas_a, alphas_b, direction_a, direction_b, hook_calls
            )
        )
        try:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=batch_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
        finally:
            handle.remove()
        if hook_calls[0] == 0:
            raise RuntimeError(
                f"Steering hook never ran for batch starting at {start}; check layer {layer_idx}."
            )
        for i in range(n):
            replies[start + i] = tok.decode(
                gen[i, max_len:], skip_special_tokens=True
            ).strip()
        logger.info(
            "Generated batch %d-%d / %d",
            start,
            start + n - 1,
            len(jobs),
        )
    return replies


def _resolve_bundle(path: Path, fallback: Path | None = None) -> Path:
    if path.is_file():
        return path
    if fallback is not None and fallback.is_file():
        logger.warning("Bundle missing at %s; falling back to %s", path, fallback)
        return fallback
    raise FileNotFoundError(f"Trait bundle not found: {path}" + (f" (fallback {fallback} also missing)" if fallback else ""))


def _load_direction(vectors_pt: Path, layer: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ck = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v = ck["v"].float()
    if not (0 <= layer < v.shape[0]):
        raise ValueError(f"layer {layer} out of range for {vectors_pt} shape {tuple(v.shape)}")
    return v[layer].to(device=device, dtype=dtype).view(1, 1, -1)


def judge_all_cells(
    cells: list[dict[str, Any]],
    *,
    instr_chaotic: str,
    instr_good: str,
    sys_chaotic: str,
    sys_good: str,
    workers: int = JUDGE_WORKERS,
) -> list[dict[str, Any]]:
    """Parallel dual-trait + coherence judging for every cell."""

    def judge_one(i: int) -> dict[str, Any]:
        c = cells[i]
        reply = str(c.get("reply", ""))
        out: dict[str, Any] = {
            "chaotic_trait_score": None,
            "chaotic_trait_reason": None,
            "good_trait_score": None,
            "good_trait_reason": None,
            "coherence_score": None,
        }
        q = str(c["question"])
        if len(reply.strip()) < MIN_REPLY_CHARS:
            out["chaotic_trait_reason"] = "reply too short"
            out["good_trait_reason"] = "reply too short"
            out["coherence_score"] = 0
            return out
        try:
            js = score_transcript(instr_chaotic, sys_chaotic, q, reply)
            out["chaotic_trait_score"] = int(js.score)
            out["chaotic_trait_reason"] = js.short_reason
        except Exception as exc:
            out["chaotic_trait_reason"] = f"judge failed: {exc}"
        try:
            jg = score_transcript(instr_good, sys_good, q, reply)
            out["good_trait_score"] = int(jg.score)
            out["good_trait_reason"] = jg.short_reason
        except Exception as exc:
            out["good_trait_reason"] = f"judge failed: {exc}"
        try:
            out["coherence_score"] = int(score_coherence(reply))
        except Exception as exc:
            out["coherence_score"] = None
            logger.warning("coherence failed for cell %d: %s", i, exc)
        return out

    pending = list(range(len(cells)))
    results: list[dict[str, Any] | None] = [None] * len(cells)

    for round_num in range(JUDGE_MAX_ROUNDS):
        if not pending:
            break
        if round_num > 0:
            wait = JUDGE_RETRY_BASE_SEC * (2 ** min(round_num - 1, 5))
            logger.info("Judge retry round %d for %d cells; sleep %.1fs", round_num, len(pending), wait)
            time.sleep(wait)
        failed: list[int] = []
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
            futures = {pool.submit(judge_one, i): i for i in pending}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    scored = fut.result()
                    # Treat total judge failure (both traits None) as retryable.
                    if (
                        scored.get("chaotic_trait_score") is None
                        and scored.get("good_trait_score") is None
                        and "too short" not in str(scored.get("chaotic_trait_reason", ""))
                    ):
                        failed.append(i)
                        results[i] = scored
                    else:
                        results[i] = scored
                except Exception as exc:
                    logger.warning("cell %d judge exception: %s", i, exc)
                    failed.append(i)
        pending = failed
        logger.info(
            "Judge round %d done; remaining failures=%d",
            round_num,
            len(pending),
        )

    enriched: list[dict[str, Any]] = []
    for i, c in enumerate(cells):
        row = dict(c)
        scored = results[i] or {
            "chaotic_trait_score": None,
            "chaotic_trait_reason": "no result",
            "good_trait_score": None,
            "good_trait_reason": "no result",
            "coherence_score": None,
        }
        row.update(scored)
        enriched.append(row)
    return enriched


def build_jobs(questions: list[str]) -> list[dict[str, Any]]:
    """Full α_c × α_g grid including 0 (pure axes + baseline)."""
    jobs: list[dict[str, Any]] = []
    for q in questions:
        for aa, ab in product(ALPHAS, ALPHAS):
            if aa == 0.0 and ab == 0.0:
                kind = "baseline"
            elif ab == 0.0:
                kind = "chaotic_only"
            elif aa == 0.0:
                kind = "good_only"
            else:
                kind = "composed"
            jobs.append(
                {
                    "kind": kind,
                    "question": q,
                    "alpha_chaotic": float(aa),
                    "alpha_good": float(ab),
                }
            )
    return jobs


def _split_axis_views(cells: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition grid cells for summary views (axis edges live inside the grid)."""
    return {
        "all": cells,
        "composed": [c for c in cells if c["kind"] == "composed"],
        "chaotic_only": [c for c in cells if c["kind"] == "chaotic_only"],
        "good_only": [c for c in cells if c["kind"] == "good_only"],
        "baseline": [c for c in cells if c["kind"] == "baseline"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chaotic+Good L16 composition test")
    parser.add_argument(
        "--config-json",
        type=Path,
        default=Path("persona_runs/dnd_config.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("persona_runs/composition_chaotic_good_l16.json"),
    )
    parser.add_argument(
        "--chaotic-vectors",
        type=Path,
        default=None,
        help="Override chaotic vectors .pt (default: from --config-json).",
    )
    parser.add_argument(
        "--good-vectors",
        type=Path,
        default=None,
        help="Override good vectors .pt (default: from --config-json).",
    )
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=JUDGE_WORKERS)
    parser.add_argument("--n-questions", type=int, default=3)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it"),
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Skip generation; load replies from --out-json and judge only.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = args.config_json if args.config_json.is_absolute() else root / args.config_json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    traits = cfg.get("traits", cfg)

    chaotic_spec = traits["chaotic"]
    good_spec = traits["good"]

    chaotic_vectors = Path(chaotic_spec["vectors"])
    good_vectors = Path(good_spec["vectors"])
    if args.chaotic_vectors is not None:
        chaotic_vectors = args.chaotic_vectors
    if args.good_vectors is not None:
        good_vectors = args.good_vectors
    if not chaotic_vectors.is_absolute():
        chaotic_vectors = root / chaotic_vectors
    if not good_vectors.is_absolute():
        good_vectors = root / good_vectors
    logger.info("Using chaotic vectors: %s", chaotic_vectors)
    logger.info("Using good vectors: %s", good_vectors)
    chaotic_bundle = Path(chaotic_spec["bundle"])
    good_bundle = Path(good_spec["bundle"])
    if not chaotic_bundle.is_absolute():
        chaotic_bundle = root / chaotic_bundle
    if not good_bundle.is_absolute():
        good_bundle = root / good_bundle
    good_bundle = _resolve_bundle(
        good_bundle,
        fallback=root / "persona_runs/dnd_good/artifacts/trait_bundle.json",
    )
    chaotic_bundle = _resolve_bundle(chaotic_bundle)

    art_c = PersonaTraitArtifact.model_validate_json(
        chaotic_bundle.read_text(encoding="utf-8")
    )
    art_g = PersonaTraitArtifact.model_validate_json(
        good_bundle.read_text(encoding="utf-8")
    )
    instr_c = judge_rubric_to_instructions(art_c.judge_rubric, trait_label=art_c.trait_label)
    instr_g = judge_rubric_to_instructions(art_g.judge_rubric, trait_label=art_g.trait_label)
    sys_c = with_paragraph_cap(art_c.neg_system_prompt)
    sys_g = with_paragraph_cap(art_g.neg_system_prompt)

    out_path = args.out_json if args.out_json.is_absolute() else root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.judge_only:
        if not out_path.is_file():
            raise SystemExit(f"--judge-only requires existing file: {out_path}")
        doc_in = json.loads(out_path.read_text(encoding="utf-8"))
        questions = list(doc_in.get("questions") or COMPOSITION_QUESTIONS[: max(1, int(args.n_questions))])
        # New format: all grid cells in "cells". Old format also had independent_controls.
        all_cells = list(doc_in.get("cells") or [])
        legacy = list(doc_in.get("independent_controls") or [])
        if legacy and not any(c.get("kind") in ("chaotic_only", "good_only", "baseline") for c in all_cells):
            all_cells = all_cells + legacy
        if not all_cells:
            raise SystemExit(f"No cells with replies in {out_path}")
        missing = sum(1 for c in all_cells if not str(c.get("reply", "")).strip())
        if missing:
            raise SystemExit(f"{missing} cells missing reply text in {out_path}")
        gen_sec = float((doc_in.get("timing_sec") or {}).get("generation") or 0.0)
        logger.info("Judge-only: %d grid cells from %s", len(all_cells), out_path)
    else:
        questions = COMPOSITION_QUESTIONS[: max(1, int(args.n_questions))]
        all_jobs = build_jobs(questions)
        views = _split_axis_views(all_jobs)
        logger.info(
            "Jobs: %d total (composed=%d chaotic_only=%d good_only=%d baseline=%d); layer=%d alphas=%s",
            len(all_jobs),
            len(views["composed"]),
            len(views["chaotic_only"]),
            len(views["good_only"]),
            len(views["baseline"]),
            args.layer,
            ALPHAS,
        )

        system = with_paragraph_cap("You are a helpful assistant.")
        model, tokenizer, device = load_model_and_tokenizer(args.model_id, device=None)
        dtype = next(model.parameters()).dtype
        d_c = _load_direction(chaotic_vectors, args.layer, device, dtype)
        d_g = _load_direction(good_vectors, args.layer, device, dtype)
        logger.info(
            "Directions loaded: chaotic ||v||=%.4f good ||v||=%.4f at L%d",
            float(d_c.norm()),
            float(d_g.norm()),
            args.layer,
        )

        t0 = time.time()
        replies = generate_batched_two_axis(
            model,
            tokenizer,
            system,
            all_jobs,
            layer_idx=int(args.layer),
            direction_a=d_c,
            direction_b=d_g,
            max_new_tokens=int(args.max_new_tokens),
            batch_size=int(args.batch_size),
        )
        gen_sec = time.time() - t0
        logger.info("Generation done in %.1fs", gen_sec)

        for job, reply in zip(all_jobs, replies):
            job["reply"] = reply
        all_cells = all_jobs

        # Persist replies immediately so we can inspect samples before judging finishes.
        interim = {
            "layer": int(args.layer),
            "model_id": args.model_id,
            "status": "generation_done_judging_pending" if not args.skip_judge else "generation_done",
            "questions": questions,
            "alphas": ALPHAS,
            "timing_sec": {"generation": round(gen_sec, 2), "judging": None},
            "cells": all_cells,
            "pareto_both_ge_70": [],
        }
        out_path.write_text(json.dumps(interim, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote interim replies to %s", out_path)

    if not args.skip_judge:
        t1 = time.time()
        all_cells = judge_all_cells(
            all_cells,
            instr_chaotic=instr_c,
            instr_good=instr_g,
            sys_chaotic=sys_c,
            sys_good=sys_g,
            workers=int(args.judge_workers),
        )
        judge_sec = time.time() - t1
        logger.info("Judging done in %.1fs", judge_sec)
    else:
        judge_sec = 0.0

    views = _split_axis_views(all_cells)
    pareto = []
    # Pareto over full grid except baseline (allows pure-axis cells that clear both traits)
    for c in all_cells:
        if c.get("kind") == "baseline":
            continue
        sa = c.get("chaotic_trait_score")
        sb = c.get("good_trait_score")
        co = c.get("coherence_score")
        if (
            sa is not None
            and sb is not None
            and co is not None
            and sa >= PARETO_MIN_TRAIT
            and sb >= PARETO_MIN_TRAIT
            and co >= PARETO_MIN_COH
        ):
            pareto.append(
                {
                    "question": c["question"],
                    "kind": c.get("kind"),
                    "alpha_chaotic": c["alpha_chaotic"],
                    "alpha_good": c["alpha_good"],
                    "chaotic_trait_score": sa,
                    "good_trait_score": sb,
                    "coherence_score": co,
                }
            )

    doc = {
        "layer": int(args.layer),
        "model_id": args.model_id,
        "status": "complete" if not args.skip_judge else "generation_done",
        "questions": questions,
        "alphas": ALPHAS,
        "chaotic_vectors": str(chaotic_vectors),
        "good_vectors": str(good_vectors),
        "chaotic_bundle": str(chaotic_bundle),
        "good_bundle": str(good_bundle),
        "timing_sec": {"generation": round(gen_sec, 2), "judging": round(judge_sec, 2)},
        "cells": all_cells,
        "chaotic_only": views["chaotic_only"],
        "good_only": views["good_only"],
        "baseline": views["baseline"],
        "pareto_both_ge_70": pareto,
        "pareto_min_trait": PARETO_MIN_TRAIT,
        "pareto_min_coherence": PARETO_MIN_COH,
    }

    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s (%d cells: %d composed, %d chaotic_only, %d good_only, %d baseline, %d pareto)",
        out_path,
        len(all_cells),
        len(views["composed"]),
        len(views["chaotic_only"]),
        len(views["good_only"]),
        len(views["baseline"]),
        len(pareto),
    )
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"Pareto both>=70 coh>={PARETO_MIN_COH}: {len(pareto)} cells", file=sys.stderr)
    for p in pareto[:20]:
        print(
            f"  kind={p.get('kind')} α_c={p['alpha_chaotic']} α_g={p['alpha_good']} "
            f"chaotic={p['chaotic_trait_score']} good={p['good_trait_score']} "
            f"coh={p['coherence_score']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
