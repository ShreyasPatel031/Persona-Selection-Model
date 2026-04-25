"""Step C §2.2: Gemma extraction rollouts + Vertex judge + filter + rollouts.jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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


def _generate_items(
    artifact: PersonaTraitArtifact,
    gemma_url: str,
    *,
    limit: int,
    timeout: int,
    paragraph_cap: bool,
    rollouts_per_q: int = 1,
    sampling_temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """
    For each contrastive_system_prompts pair, each extraction question, and each
    rollout replicate, call Gemma pos/neg. Matches paper §2.2 scale when
    rollouts_per_q=10 and len(pairs)=5.
    """
    questions = artifact.extraction_questions
    if limit and limit < len(questions):
        questions = questions[:limit]
    pairs = list(artifact.contrastive_system_prompts or ())
    if not pairs:
        raise ValueError("Artifact has no contrastive_system_prompts.")
    base = gemma_url.rstrip("/")
    items: list[dict[str, Any]] = []
    linear = 0
    do_sample = rollouts_per_q > 1
    temp = sampling_temperature if do_sample else None
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
                        base,
                        q,
                        pos_sys,
                        timeout=timeout,
                        do_sample=do_sample,
                        temperature=temp,
                        seed=seed_base + 1,
                    )
                except Exception as e:
                    logger.exception("pos failed: %s", e)
                    pos = f"<error: {e}>"
                try:
                    neg = chat_nonstream(
                        base,
                        q,
                        neg_sys,
                        timeout=timeout,
                        do_sample=do_sample,
                        temperature=temp,
                        seed=seed_base + 2,
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
    rollouts_per_q: int = 1,
    sampling_temperature: float = 1.0,
) -> tuple[Path, Path | None]:
    """
    Step C: extraction rollouts, optional Vertex judge + filter, writes
    extraction_rollouts.json and (if judged) rollouts.jsonl per plan §2.2.
    """
    raw = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw)

    pos_thr = (
        pos_threshold if pos_threshold is not None else JUDGE_POS_KEEP_IF_SCORE_GT
    )
    neg_thr = (
        neg_threshold if neg_threshold is not None else JUDGE_NEG_KEEP_IF_SCORE_LT
    )

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
        )

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
        for it in items:
            summary_items.append(dict(it))
        doc: dict[str, Any] = {
            "step": "C",
            "kind": "extraction",
            "judge": None,
            "gemma_url": gemma_url,
            "trait_bundle": str(bundle_path.resolve()),
            "trait_label": artifact.trait_label,
            "paragraph_cap": paragraph_cap,
            "rollouts_per_q": max(1, int(rollouts_per_q)),
            "sampling_temperature": float(sampling_temperature),
            "contrast_pair_count": artifact.contrast_pair_count(),
            "filter": None,
            "items": summary_items,
        }
        rollouts_json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return rollouts_json_path, None

    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript

    judge_instructions = judge_rubric_to_instructions(artifact.judge_rubric)
    jproj = project_id
    jloc = location
    jmodel = judge_model

    pairs_list = list(artifact.contrastive_system_prompts or ())
    for it in items:
        q = it["question"]
        idx = int(it.get("index", 0))
        pair_idx = int(it.get("pair_index", 0))
        if pair_idx >= len(pairs_list):
            pair_idx = 0
        pair = pairs_list[pair_idx]
        pos_sys, neg_sys = _pair_prompts(pair, paragraph_cap=paragraph_cap)
        pos_rep = it["pos_reply"]
        neg_rep = it["neg_reply"]
        row: dict[str, Any] = {
            "index": idx,
            "pair_index": pair_idx,
            "question_index": it.get("question_index", idx),
            "rollout_index": it.get("rollout_index", 0),
            "question": q,
        }

        for arm, sys_text, rep in (
            ("pos", pos_sys, pos_rep),
            ("neg", neg_sys, neg_rep),
        ):
            err = rep.startswith("<error:")
            if err:
                if arm == "pos":
                    stats["pos_errors"] += 1
                else:
                    stats["neg_errors"] += 1
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
                jsonl_lines.append(rec)
                row[f"{arm}_reply"] = rep
                row[f"{arm}_score"] = None
                row[f"{arm}_short_reason"] = None
                row[f"{arm}_kept"] = False
                continue

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
                if arm == "pos":
                    stats["pos_errors"] += 1
                else:
                    stats["neg_errors"] += 1
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
                jsonl_lines.append(rec)
                row[f"{arm}_reply"] = rep
                row[f"{arm}_score"] = None
                row[f"{arm}_short_reason"] = f"<judge_error: {e}>"
                row[f"{arm}_kept"] = False
                continue

            if arm == "pos":
                stats["pos_judged"] += 1
                kept = js.score > pos_thr
                if kept:
                    stats["pos_kept"] += 1
            else:
                stats["neg_judged"] += 1
                kept = js.score < neg_thr
                if kept:
                    stats["neg_kept"] += 1

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
            jsonl_lines.append(rec)
            row[f"{arm}_reply"] = rep
            row[f"{arm}_score"] = js.score
            row[f"{arm}_short_reason"] = js.short_reason
            row[f"{arm}_kept"] = kept

        summary_items.append(row)

    jsonl_out = jsonl_path or (rollouts_json_path.parent / "rollouts.jsonl")
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
        "gemma_url": gemma_url,
        "trait_bundle": str(bundle_path.resolve()),
        "trait_label": artifact.trait_label,
        "paragraph_cap": paragraph_cap,
        "rollouts_per_q": max(1, int(rollouts_per_q)),
        "sampling_temperature": float(sampling_temperature),
        "contrast_pair_count": len(pairs_list),
        "rollouts_jsonl": str(jsonl_out.resolve()),
        "stats": stats,
        "items": summary_items,
    }
    rollouts_json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return rollouts_json_path, jsonl_out


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
