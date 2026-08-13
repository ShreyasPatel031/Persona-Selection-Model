"""Step C §2.2: Gemma extraction rollouts + Vertex judge + filter + rollouts.jsonl."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Canonical filenames under <run>/rollouts/ — do not introduce alternate names.
EXTRACTION_ROLLOUTS_JSON = "extraction_rollouts.json"
ROLLOUTS_JSONL = "rollouts.jsonl"
ROLLOUTS_LATEST_JSON = "latest.json"

from app.persona.config import (
    JUDGE_NEG_KEEP_IF_SCORE_LT,
    JUDGE_POS_KEEP_IF_SCORE_GT,
)
from app.persona.gemma_client import chat_nonstream
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import ContrastPromptPair, PersonaTraitArtifact

logger = logging.getLogger(__name__)


def _pair_prompts(pair: ContrastPromptPair, *, paragraph_cap: bool) -> tuple[str, str]:
    if paragraph_cap:
        return with_paragraph_cap(pair.positive), with_paragraph_cap(pair.negative)
    return pair.positive, pair.negative


def _dedupe_questions(questions: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        key = q.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def questions_for_source(
    artifact: PersonaTraitArtifact,
    questions_source: str,
) -> list[str]:
    """Resolve rollout question list from bundle fields."""
    source = (questions_source or "extraction").strip().lower()
    if source == "extraction":
        return list(artifact.extraction_questions)
    if source == "eval":
        eq = list(getattr(artifact, "eval_questions", None) or ())
        if not eq:
            raise ValueError("Bundle has no eval_questions.")
        return eq
    if source in ("scenarios", "scenario"):
        eval_q = list(getattr(artifact, "eval_questions", None) or ())
        contrast = list(getattr(artifact, "contrast_scenarios", None) or ())
        combined = _dedupe_questions(eval_q + contrast)
        if not combined:
            raise ValueError(
                "Bundle has no eval_questions or contrast_scenarios for scenarios source."
            )
        return combined
    raise ValueError(
        f"Unknown questions_source={questions_source!r}; "
        "use extraction, eval, or scenarios."
    )


def _set_gen_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _generate_reply_local(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    question: str,
    *,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int | None = None,
) -> str:
    """In-process CUDA generation (no HTTP)."""
    _set_gen_seed(seed)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    raw_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = (
        raw_ids.to(device)
        if isinstance(raw_ids, torch.Tensor)
        else raw_ids["input_ids"].to(device)
    )
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    in_len = int(input_ids.shape[-1])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kw["temperature"] = float(temperature)
    with torch.no_grad():
        gen_ids = model.generate(**gen_kw)
    return tokenizer.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()


def _generate_batch_local(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompts: list[tuple[str, str, int | None]],
    *,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[str]:
    """Batched in-process generation. Each prompt is (system, question, seed).

    Left-pads inputs so shorter sequences align on the right. Returns one
    decoded reply string per prompt in the same order.
    """
    if not prompts:
        return []

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    all_ids: list[torch.Tensor] = []
    for system, question, seed in prompts:
        _set_gen_seed(seed)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        raw = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        ids = raw if isinstance(raw, torch.Tensor) else raw["input_ids"]
        all_ids.append(ids.squeeze(0))

    max_len = max(t.shape[0] for t in all_ids)
    padded = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros_like(padded)
    in_lens: list[int] = []
    for i, ids in enumerate(all_ids):
        seq_len = ids.shape[0]
        in_lens.append(seq_len)
        padded[i, max_len - seq_len:] = ids.to(device)
        attn[i, max_len - seq_len:] = 1

    gen_kw: dict[str, Any] = {
        "input_ids": padded,
        "attention_mask": attn,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kw["temperature"] = float(temperature)
    with torch.no_grad():
        gen_ids = model.generate(**gen_kw)

    replies: list[str] = []
    for i, il in enumerate(in_lens):
        offset = max_len - il
        new_tokens = gen_ids[i, max_len:]
        replies.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return replies


_DEFAULT_BATCH_SIZE = 8


def _generate_items(
    artifact: PersonaTraitArtifact,
    gemma_url: str,
    *,
    limit: int,
    timeout: int,
    paragraph_cap: bool,
    rollouts_per_q: int = 10,
    sampling_temperature: float = 1.0,
    questions_override: list[str] | None = None,
    max_pairs: int | None = None,
    local_model: PreTrainedModel | None = None,
    local_tokenizer: AutoTokenizer | None = None,
    local_device: torch.device | None = None,
    max_new_tokens: int = 200,
    on_item: Any | None = None,
) -> list[dict[str, Any]]:
    """
    For each contrastive_system_prompts pair, each extraction question, and each
    rollout replicate, call Gemma pos/neg. Paper §2.2: 10 rollouts per question
    × 5 prompt pairs × 20 questions = 1000 per arm before filtering.
    """
    questions = list(questions_override or artifact.extraction_questions)
    if limit and limit < len(questions):
        questions = questions[:limit]
    pairs = list(artifact.contrastive_system_prompts or ())
    if max_pairs is not None and int(max_pairs) > 0:
        pairs = pairs[: int(max_pairs)]
    if not pairs:
        raise ValueError("Artifact has no contrastive_system_prompts.")
    use_local = local_model is not None and local_tokenizer is not None and local_device is not None
    base = gemma_url.rstrip("/") if not use_local else ""
    if use_local:
        logger.info(
            "Using in-process GPU generation on %s (max_new_tokens=%s)",
            local_device,
            max_new_tokens,
        )
    items: list[dict[str, Any]] = []
    linear = 0
    do_sample = rollouts_per_q > 1
    temp = sampling_temperature if do_sample else None

    if use_local:
        batch_size = _DEFAULT_BATCH_SIZE
        work: list[dict[str, Any]] = []
        for pair_index, pair in enumerate(pairs):
            pos_sys, neg_sys = _pair_prompts(pair, paragraph_cap=paragraph_cap)
            for question_index, q in enumerate(questions):
                for rollout_index in range(rollouts_per_q):
                    seed_base = pair_index * 1_000_000 + question_index * 10_000 + rollout_index
                    work.append({
                        "pair_index": pair_index,
                        "question_index": question_index,
                        "rollout_index": rollout_index,
                        "question": q,
                        "pos_sys": pos_sys,
                        "neg_sys": neg_sys,
                        "seed_base": seed_base,
                    })

        total = len(work)
        logger.info("Batched generation: %d items, batch_size=%d", total, batch_size)

        for batch_start in range(0, total, batch_size):
            batch = work[batch_start : batch_start + batch_size]
            pos_prompts = [(w["pos_sys"], w["question"], w["seed_base"] + 1) for w in batch]
            neg_prompts = [(w["neg_sys"], w["question"], w["seed_base"] + 2) for w in batch]

            logger.info(
                "Batch %d–%d / %d (pos)",
                batch_start + 1, batch_start + len(batch), total,
            )
            try:
                pos_replies = _generate_batch_local(
                    local_model, local_tokenizer, local_device,
                    pos_prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temp or 1.0,
                )
            except Exception as e:
                logger.exception("pos batch failed, falling back to singles: %s", e)
                pos_replies = []
                for sys_p, q_p, seed_p in pos_prompts:
                    try:
                        pos_replies.append(_generate_reply_local(
                            local_model, local_tokenizer, local_device,
                            sys_p, q_p, max_new_tokens=max_new_tokens,
                            do_sample=do_sample, temperature=temp or 1.0, seed=seed_p,
                        ))
                    except Exception as e2:
                        pos_replies.append(f"<error: {e2}>")

            logger.info(
                "Batch %d–%d / %d (neg)",
                batch_start + 1, batch_start + len(batch), total,
            )
            try:
                neg_replies = _generate_batch_local(
                    local_model, local_tokenizer, local_device,
                    neg_prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temp or 1.0,
                )
            except Exception as e:
                logger.exception("neg batch failed, falling back to singles: %s", e)
                neg_replies = []
                for sys_n, q_n, seed_n in neg_prompts:
                    try:
                        neg_replies.append(_generate_reply_local(
                            local_model, local_tokenizer, local_device,
                            sys_n, q_n, max_new_tokens=max_new_tokens,
                            do_sample=do_sample, temperature=temp or 1.0, seed=seed_n,
                        ))
                    except Exception as e2:
                        neg_replies.append(f"<error: {e2}>")

            for i, w in enumerate(batch):
                items.append({
                    "index": linear,
                    "pair_index": w["pair_index"],
                    "question_index": w["question_index"],
                    "rollout_index": w["rollout_index"],
                    "question": w["question"],
                    "pos_reply": pos_replies[i],
                    "neg_reply": neg_replies[i],
                })
                if on_item is not None:
                    on_item(items[-1])
                linear += 1
        return items

    for pair_index, pair in enumerate(pairs):
        pos_sys, neg_sys = _pair_prompts(pair, paragraph_cap=paragraph_cap)
        for question_index, q in enumerate(questions):
            for rollout_index in range(rollouts_per_q):
                logger.info(
                    "Rollout pair=%s q=%s/%s r=%s/%s",
                    pair_index,
                    question_index + 1,
                    len(questions),
                    rollout_index + 1,
                    rollouts_per_q,
                )
                seed_base = (
                    pair_index * 1_000_000 + question_index * 10_000 + rollout_index
                )
                try:
                    pos = chat_nonstream(
                        base, q, pos_sys, timeout=timeout,
                        do_sample=do_sample, temperature=temp, seed=seed_base + 1,
                    )
                except Exception as e:
                    logger.exception("pos failed: %s", e)
                    pos = f"<error: {e}>"
                try:
                    neg = chat_nonstream(
                        base, q, neg_sys, timeout=timeout,
                        do_sample=do_sample, temperature=temp, seed=seed_base + 2,
                    )
                except Exception as e:
                    logger.exception("neg failed: %s", e)
                    neg = f"<error: {e}>"
                items.append(
                    {
                        "index": linear,
                        "pair_index": pair_index,
                        "question_index": question_index,
                        "rollout_index": rollout_index,
                        "question": q,
                        "pos_reply": pos,
                        "neg_reply": neg,
                    }
                )
                if on_item is not None:
                    on_item(items[-1])
                linear += 1
    return items


def run_step_c(
    bundle_path: Path,
    gemma_url: str,
    rollouts_json_path: Path,
    *,
    jsonl_path: Path | None = None,
    limit: int = 0,
    timeout: int = 720,
    paragraph_cap: bool = True,
    skip_judge: bool = False,
    from_rollouts_json: Path | None = None,
    project_id: str | None = None,
    location: str | None = None,
    judge_model: str | None = None,
    pos_threshold: int | None = None,
    neg_threshold: int | None = None,
    rollouts_per_q: int = 10,
    sampling_temperature: float = 1.0,
    questions_source: str = "extraction",
    max_pairs: int | None = None,
    use_local_gpu: bool = False,
    max_new_tokens: int = 200,
    judge_workers: int = 8,
    model_id: str | None = None,
) -> tuple[Path, Path | None]:
    """
    Step C: extraction rollouts, optional Vertex judge + filter, writes
    extraction_rollouts.json and (if judged) rollouts.jsonl per plan §2.2.

    Paper spec: 10 rollouts/question × 5 prompt pairs × 20 questions = 1000/arm.
    """
    raw = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw)

    pos_thr = (
        pos_threshold if pos_threshold is not None else JUDGE_POS_KEEP_IF_SCORE_GT
    )
    neg_thr = (
        neg_threshold if neg_threshold is not None else JUDGE_NEG_KEEP_IF_SCORE_LT
    )
    resolved_questions = questions_for_source(artifact, questions_source)
    pairs_list = list(artifact.contrastive_system_prompts or ())
    rollouts_json_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_out: Path | None = None
    jsonl_lines: list[dict[str, Any]] = []
    summary_items: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "pos_kept": 0,
        "neg_kept": 0,
        "pos_judged": 0,
        "neg_judged": 0,
        "pos_errors": 0,
        "neg_errors": 0,
    }

    if skip_judge:
        local_model = local_tokenizer = None
        local_device: torch.device | None = None
        if use_local_gpu and from_rollouts_json is None:
            from app.persona.activations import load_model_and_tokenizer

            logger.info("Loading Gemma on GPU for in-process step-c rollouts...")
            local_model, local_tokenizer, local_device = load_model_and_tokenizer(model_id)
        try:
            if from_rollouts_json is not None:
                prev = json.loads(from_rollouts_json.read_text(encoding="utf-8"))
                items = prev.get("items") or []
                if limit and limit < len(items):
                    items = items[:limit]
                logger.info("Loaded %s rollout rows from %s", len(items), from_rollouts_json)
            else:
                items = _generate_items(
                    artifact,
                    gemma_url,
                    limit=limit,
                    timeout=timeout,
                    paragraph_cap=paragraph_cap,
                    rollouts_per_q=max(1, int(rollouts_per_q)),
                    sampling_temperature=float(sampling_temperature),
                    questions_override=resolved_questions,
                    max_pairs=max_pairs,
                    local_model=local_model,
                    local_tokenizer=local_tokenizer,
                    local_device=local_device,
                    max_new_tokens=max_new_tokens,
                )
        finally:
            if local_model is not None:
                del local_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Released in-process Gemma model from GPU memory")

        for it in items:
            summary_items.append(dict(it))
        doc: dict[str, Any] = {
            "step": "C",
            "kind": "extraction",
            "judge": None,
            "gemma_url": "local-gpu" if use_local_gpu else gemma_url,
            "trait_bundle": str(bundle_path.resolve()),
            "trait_label": artifact.trait_label,
            "paragraph_cap": paragraph_cap,
            "questions_source": questions_source,
            "question_count": len(resolved_questions),
            "rollouts_per_q": max(1, int(rollouts_per_q)),
            "sampling_temperature": float(sampling_temperature),
            "contrast_pair_count": artifact.contrast_pair_count(),
            "filter": None,
            "items": summary_items,
        }
        rollouts_json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        jsonl_out = jsonl_path or (rollouts_json_path.parent / ROLLOUTS_JSONL)
        extraction_json_to_rollouts_jsonl(
            rollouts_json_path,
            bundle_path,
            jsonl_out,
            paragraph_cap=paragraph_cap,
        )
        run_dir = _run_dir_from_rollouts_path(rollouts_json_path)
        if run_dir is not None:
            write_rollouts_latest(run_dir, doc)
        return rollouts_json_path, jsonl_out

    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript

    judge_instructions = judge_rubric_to_instructions(artifact.judge_rubric, trait_label=artifact.trait_label)
    jproj = project_id
    jloc = location
    jmodel = judge_model
    workers = max(1, int(judge_workers))
    logger.info("Judging with %s parallel workers (overlapped with GPU generation)", workers)

    def _judge_arm(
        it: dict[str, Any],
        arm: str,
        sys_text: str,
        rep: str,
    ) -> tuple[int, str, dict[str, Any], dict[str, Any]]:
        idx = int(it.get("index", 0))
        pair_idx = int(it.get("pair_index", 0))
        q = it["question"]
        err = rep.startswith("<error:")
        if err:
            rec = {
                "q_index": idx,
                "pair_index": pair_idx,
                "question_index": it.get("question_index", 0),
                "rollout_index": it.get("rollout_index", 0),
                "arm": arm,
                "question": q,
                "system": sys_text,
                "assistant_a": rep,
                "score": None,
                "short_reason": None,
                "kept": False,
                "error": True,
            }
            row_part = {
                f"{arm}_reply": rep,
                f"{arm}_score": None,
                f"{arm}_short_reason": None,
                f"{arm}_kept": False,
            }
            stat_part = {"arm": arm, "judged": False, "kept": False, "error": True}
            return idx, arm, rec, {**row_part, **stat_part}

        try:
            js = score_transcript(
                judge_instructions,
                sys_text,
                q,
                rep,
                project_id=jproj,
                location=jloc,
                model_name=jmodel,
            )
        except Exception as e:
            logger.exception("judge failed %s/%s: %s", idx, arm, e)
            rec = {
                "q_index": idx,
                "pair_index": pair_idx,
                "question_index": it.get("question_index", 0),
                "rollout_index": it.get("rollout_index", 0),
                "arm": arm,
                "question": q,
                "system": sys_text,
                "assistant_a": rep,
                "score": None,
                "short_reason": str(e),
                "kept": False,
                "error": True,
            }
            row_part = {
                f"{arm}_reply": rep,
                f"{arm}_score": None,
                f"{arm}_short_reason": f"<judge_error: {e}>",
                f"{arm}_kept": False,
            }
            stat_part = {"arm": arm, "judged": False, "kept": False, "error": True}
            return idx, arm, rec, {**row_part, **stat_part}

        kept = js.score > pos_thr if arm == "pos" else js.score < neg_thr
        rec = {
            "q_index": idx,
            "pair_index": pair_idx,
            "question_index": it.get("question_index", 0),
            "rollout_index": it.get("rollout_index", 0),
            "arm": arm,
            "question": q,
            "system": sys_text,
            "assistant_a": rep,
            "score": js.score,
            "short_reason": js.short_reason,
            "kept": kept,
            "error": False,
        }
        row_part = {
            f"{arm}_reply": rep,
            f"{arm}_score": js.score,
            f"{arm}_short_reason": js.short_reason,
            f"{arm}_kept": kept,
        }
        stat_part = {"arm": arm, "judged": True, "kept": kept, "error": False}
        return idx, arm, rec, {**row_part, **stat_part}

    judged_by_key: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    pending_futures: list[Any] = []

    local_model = local_tokenizer = None
    local_device: torch.device | None = None
    if use_local_gpu and from_rollouts_json is None:
        from app.persona.activations import load_model_and_tokenizer

        logger.info("Loading Gemma on GPU for in-process step-c rollouts...")
        local_model, local_tokenizer, local_device = load_model_and_tokenizer(model_id)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        def _submit_item_judges(it: dict[str, Any]) -> None:
            pair_idx = int(it.get("pair_index", 0))
            if pair_idx >= len(pairs_list):
                pair_idx = 0
            pair = pairs_list[pair_idx]
            pos_sys, neg_sys = _pair_prompts(pair, paragraph_cap=paragraph_cap)
            pending_futures.append(pool.submit(_judge_arm, it, "pos", pos_sys, it["pos_reply"]))
            pending_futures.append(pool.submit(_judge_arm, it, "neg", neg_sys, it["neg_reply"]))

        try:
            if from_rollouts_json is not None:
                prev = json.loads(from_rollouts_json.read_text(encoding="utf-8"))
                items = prev.get("items") or []
                if limit and limit < len(items):
                    items = items[:limit]
                logger.info("Loaded %s rollout rows from %s", len(items), from_rollouts_json)
                for it in items:
                    _submit_item_judges(it)
            else:
                items = _generate_items(
                    artifact,
                    gemma_url,
                    limit=limit,
                    timeout=timeout,
                    paragraph_cap=paragraph_cap,
                    rollouts_per_q=max(1, int(rollouts_per_q)),
                    sampling_temperature=float(sampling_temperature),
                    questions_override=resolved_questions,
                    max_pairs=max_pairs,
                    local_model=local_model,
                    local_tokenizer=local_tokenizer,
                    local_device=local_device,
                    max_new_tokens=max_new_tokens,
                    on_item=_submit_item_judges,
                )
        finally:
            if local_model is not None:
                del local_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Released in-process Gemma model from GPU memory")

        done = 0
        for fut in as_completed(pending_futures):
            idx, arm, rec, meta = fut.result()
            judged_by_key[(idx, arm)] = (rec, meta)
            done += 1
            if done % 50 == 0 or done == len(pending_futures):
                logger.info("Judge progress: %s/%s", done, len(pending_futures))

    for it in items:
        q = it["question"]
        idx = int(it.get("index", 0))
        pair_idx = int(it.get("pair_index", 0))
        row: dict[str, Any] = {
            "index": idx,
            "pair_index": pair_idx,
            "question_index": it.get("question_index", idx),
            "rollout_index": it.get("rollout_index", 0),
            "question": q,
        }
        for arm in ("pos", "neg"):
            rec, meta = judged_by_key[(idx, arm)]
            jsonl_lines.append(rec)
            row[f"{arm}_reply"] = meta[f"{arm}_reply"]
            row[f"{arm}_score"] = meta[f"{arm}_score"]
            row[f"{arm}_short_reason"] = meta[f"{arm}_short_reason"]
            row[f"{arm}_kept"] = meta[f"{arm}_kept"]
            if meta.get("error"):
                if arm == "pos":
                    stats["pos_errors"] += 1
                else:
                    stats["neg_errors"] += 1
            elif meta.get("judged"):
                if arm == "pos":
                    stats["pos_judged"] += 1
                    if meta.get("kept"):
                        stats["pos_kept"] += 1
                else:
                    stats["neg_judged"] += 1
                    if meta.get("kept"):
                        stats["neg_kept"] += 1
        summary_items.append(row)

    jsonl_out = jsonl_path or (rollouts_json_path.parent / ROLLOUTS_JSONL)
    with jsonl_out.open("w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    doc = {
        "step": "C",
        "kind": "extraction",
        "judge": {
            "vertex_model": jmodel or "(env default)",
            "pos_keep_if_score_gt": pos_thr,
            "neg_keep_if_score_lt": neg_thr,
        },
        "gemma_url": "local-gpu" if use_local_gpu else gemma_url,
        "generation": {
            "local_gpu": use_local_gpu,
            "max_new_tokens": int(max_new_tokens),
            "judge_workers": int(judge_workers),
        },
        "trait_bundle": str(bundle_path.resolve()),
        "trait_label": artifact.trait_label,
        "paragraph_cap": paragraph_cap,
        "questions_source": questions_source,
        "question_count": len(resolved_questions),
        "rollouts_per_q": max(1, int(rollouts_per_q)),
        "sampling_temperature": float(sampling_temperature),
        "contrast_pair_count": len(pairs_list),
        "max_pairs": max_pairs,
        "rollouts_jsonl": str(jsonl_out.resolve()),
        "stats": stats,
        "items": summary_items,
    }
    rollouts_json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    run_dir = _run_dir_from_rollouts_path(rollouts_json_path)
    if run_dir is not None:
        write_rollouts_latest(run_dir, doc)
    return rollouts_json_path, jsonl_out


def rollouts_dir(run_dir: Path) -> Path:
    return run_dir / "rollouts"


def canonical_extraction_json(run_dir: Path) -> Path:
    return rollouts_dir(run_dir) / EXTRACTION_ROLLOUTS_JSON


def canonical_rollouts_jsonl(run_dir: Path) -> Path:
    return rollouts_dir(run_dir) / ROLLOUTS_JSONL


def canonical_rollouts_latest(run_dir: Path) -> Path:
    return rollouts_dir(run_dir) / ROLLOUTS_LATEST_JSON


def _run_dir_from_rollouts_path(rollouts_json_path: Path) -> Path | None:
    if rollouts_json_path.parent.name == "rollouts":
        return rollouts_json_path.parent.parent
    return None


def _bundle_path_for_extraction_doc(
    doc: dict[str, Any],
    run_dir: Path,
) -> Path:
    bundle_hint = doc.get("trait_bundle")
    if bundle_hint:
        bundle_path = Path(bundle_hint)
        if bundle_path.is_file():
            return bundle_path
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Cannot sync {ROLLOUTS_JSONL}: trait bundle not found at {bundle_path}"
        )
    return bundle_path


def write_rollouts_latest(run_dir: Path, step_c_doc: dict[str, Any]) -> Path:
    """Record which canonical rollout files step-C just wrote (fixed names under rollouts/)."""
    rdir = rollouts_dir(run_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    extraction = rdir / EXTRACTION_ROLLOUTS_JSON
    jsonl = rdir / ROLLOUTS_JSONL
    latest: dict[str, Any] = {
        "schema_version": "1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_rollouts_json": EXTRACTION_ROLLOUTS_JSON,
        "rollouts_jsonl": ROLLOUTS_JSONL,
        "questions_source": step_c_doc.get("questions_source"),
        "question_count": step_c_doc.get("question_count"),
        "item_count": len(step_c_doc.get("items") or ()),
        "skip_judge": step_c_doc.get("judge") is None,
    }
    if extraction.is_file():
        latest["extraction_mtime"] = extraction.stat().st_mtime
    if jsonl.is_file():
        latest["rollouts_jsonl_mtime"] = jsonl.stat().st_mtime
    out = rdir / ROLLOUTS_LATEST_JSON
    out.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote rollouts pointer %s (questions_source=%s)", out, latest.get("questions_source"))
    return out


def sync_rollouts_jsonl_from_extraction(
    run_dir: Path,
    *,
    paragraph_cap: bool = True,
) -> Path:
    """Rebuild rollouts.jsonl from extraction_rollouts.json when step-C output is newer."""
    extraction = canonical_extraction_json(run_dir)
    jsonl = canonical_rollouts_jsonl(run_dir)
    if not extraction.is_file():
        raise FileNotFoundError(f"Missing step-C output: {extraction}")
    doc = json.loads(extraction.read_text(encoding="utf-8"))
    bundle_path = _bundle_path_for_extraction_doc(doc, run_dir)
    extraction_json_to_rollouts_jsonl(
        extraction,
        bundle_path,
        jsonl,
        paragraph_cap=paragraph_cap,
    )
    write_rollouts_latest(run_dir, doc)
    return jsonl


def run_extraction_rollouts(
    bundle_path: Path,
    gemma_url: str,
    out_path: Path,
    *,
    limit: int = 0,
    timeout: int = 720,
    paragraph_cap: bool = True,
) -> Path:
    """Rollouts only (no judge); backward-compatible wrapper."""
    p, _ = run_step_c(
        bundle_path,
        gemma_url,
        out_path,
        limit=limit,
        timeout=timeout,
        paragraph_cap=paragraph_cap,
        skip_judge=True,
        rollouts_per_q=1,
        sampling_temperature=1.0,
    )
    return p


def extraction_json_to_rollouts_jsonl(
    extraction_json_path: Path,
    bundle_path: Path,
    jsonl_out: Path,
    *,
    paragraph_cap: bool = True,
) -> Path:
    """
    Build rollouts.jsonl from step-C extraction_rollouts.json (e.g. --skip-judge).
    All pos/neg pairs are marked kept so step-D can consume them without a judge pass.

    WARNING: This bypasses quality filtering. The paper (Chen et al., 2025 §2.2)
    requires filtering by trait score (>50 for pos, <50 for neg) to remove rollouts
    where the model didn't follow the system prompt. Without filtering, noisy
    rollouts dilute the contrastive signal and produce weaker vectors.
    """
    logger.warning(
        "SKIP-JUDGE: converting extraction_rollouts.json → rollouts.jsonl WITHOUT "
        "quality filtering. All rollouts marked kept=True with synthetic scores. "
        "For production vectors, re-run Step C with judge enabled (no --skip-judge)."
    )
    doc = json.loads(extraction_json_path.read_text(encoding="utf-8"))
    items = doc.get("items") or []
    if not items:
        raise ValueError(f"No rollout items in {extraction_json_path}")

    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    pairs = list(artifact.contrastive_system_prompts or ())
    if not pairs:
        raise ValueError("Trait bundle has no contrastive_system_prompts.")

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[dict[str, Any]] = []
    for it in items:
        idx = int(it.get("index", len(lines) // 2))
        pair_idx = int(it.get("pair_index", 0))
        if pair_idx >= len(pairs):
            pair_idx = 0
        pair = pairs[pair_idx]
        pos_sys, neg_sys = _pair_prompts(pair, paragraph_cap=paragraph_cap)
        q = it["question"]
        base = {
            "q_index": idx,
            "pair_index": pair_idx,
            "question_index": it.get("question_index", idx),
            "rollout_index": it.get("rollout_index", 0),
            "question": q,
            "kept": True,
            "error": False,
        }
        for arm, sys_text, rep, score in (
            ("pos", pos_sys, it["pos_reply"], 100),
            ("neg", neg_sys, it["neg_reply"], 0),
        ):
            lines.append(
                {
                    **base,
                    "arm": arm,
                    "system": sys_text,
                    "assistant_a": rep,
                    "score": score,
                    "short_reason": "from_extraction_rollouts_json",
                }
            )

    with jsonl_out.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote %s rollout rows (%s items) from %s",
        len(lines),
        len(items),
        extraction_json_path.name,
    )
    return jsonl_out


def resolve_rollouts_jsonl(
    run_dir: Path,
    *,
    explicit_jsonl: Path | None = None,
    paragraph_cap: bool = True,
) -> tuple[Path, str]:
    """
    Resolve rollouts.jsonl for step-D and downstream tools.

    Uses fixed canonical paths under ``<run>/rollouts/`` only:
    ``extraction_rollouts.json`` (step-C JSON), ``rollouts.jsonl`` (step-D input),
    ``latest.json`` (pointer written by step-C).

    If step-C JSON is newer than rollouts.jsonl, rebuild jsonl and refresh latest.json.
    Raises if rollouts.jsonl would be stale relative to extraction_rollouts.json.
    """
    if explicit_jsonl is not None:
        if not explicit_jsonl.is_file():
            raise FileNotFoundError(f"Missing rollouts jsonl: {explicit_jsonl}")
        return explicit_jsonl.resolve(), "explicit"

    extraction = canonical_extraction_json(run_dir)
    jsonl = canonical_rollouts_jsonl(run_dir)
    latest = canonical_rollouts_latest(run_dir)

    if not extraction.is_file() and not jsonl.is_file():
        raise FileNotFoundError(
            f"Missing rollouts under {rollouts_dir(run_dir)}: run step-C first "
            f"(expects {EXTRACTION_ROLLOUTS_JSON} and/or {ROLLOUTS_JSONL})."
        )

    if extraction.is_file():
        ext_mtime = extraction.stat().st_mtime
        jsonl_mtime = jsonl.stat().st_mtime if jsonl.is_file() else 0.0
        if not jsonl.is_file() or ext_mtime > jsonl_mtime:
            if jsonl.is_file():
                logger.warning(
                    "%s is newer than %s — syncing jsonl from latest step-C output.",
                    EXTRACTION_ROLLOUTS_JSON,
                    ROLLOUTS_JSONL,
                )
            sync_rollouts_jsonl_from_extraction(
                run_dir,
                paragraph_cap=paragraph_cap,
            )
            return canonical_rollouts_jsonl(run_dir).resolve(), EXTRACTION_ROLLOUTS_JSON

    if not jsonl.is_file():
        raise FileNotFoundError(
            f"Missing {ROLLOUTS_JSONL} under {rollouts_dir(run_dir)}; "
            f"re-run step-C or pass --rollouts-jsonl."
        )

    if latest.is_file():
        try:
            meta = json.loads(latest.read_text(encoding="utf-8"))
            if meta.get("extraction_rollouts_json") != EXTRACTION_ROLLOUTS_JSON:
                logger.warning(
                    "Ignoring stale %s: non-canonical extraction path %r",
                    ROLLOUTS_LATEST_JSON,
                    meta.get("extraction_rollouts_json"),
                )
            elif meta.get("rollouts_jsonl") != ROLLOUTS_JSONL:
                logger.warning(
                    "Ignoring stale %s: non-canonical jsonl path %r",
                    ROLLOUTS_LATEST_JSON,
                    meta.get("rollouts_jsonl"),
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", ROLLOUTS_LATEST_JSON, e)

    return jsonl.resolve(), ROLLOUTS_JSONL


def resolve_step_d_rollouts_jsonl(
    run_dir: Path,
    *,
    explicit_jsonl: Path | None = None,
    paragraph_cap: bool = True,
) -> tuple[Path, str]:
    """Backward-compatible alias for :func:`resolve_rollouts_jsonl`."""
    return resolve_rollouts_jsonl(
        run_dir,
        explicit_jsonl=explicit_jsonl,
        paragraph_cap=paragraph_cap,
    )
