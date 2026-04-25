"""Call Gemma /chat for each eval question under pos vs neg system prompts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.persona.gemma_client import chat_nonstream
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact

logger = logging.getLogger(__name__)


def run_eval_answers(
    bundle_path: Path,
    gemma_url: str,
    out_path: Path,
    *,
    limit: int = 0,
    timeout: int = 720,
    paragraph_cap: bool = True,
) -> Path:
    raw = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw)
    questions = artifact.eval_questions
    if limit and limit < len(questions):
        questions = questions[:limit]

    pos_sys = (
        with_paragraph_cap(artifact.pos_system_prompt)
        if paragraph_cap
        else artifact.pos_system_prompt
    )
    neg_sys = (
        with_paragraph_cap(artifact.neg_system_prompt)
        if paragraph_cap
        else artifact.neg_system_prompt
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    base = gemma_url.rstrip("/")
    n = len(questions)
    for i, q in enumerate(questions):
        logger.info("Eval question %s/%s", i + 1, n)
        try:
            pos = chat_nonstream(base, q, pos_sys, timeout=timeout)
        except Exception as e:
            logger.exception("pos failed: %s", e)
            pos = f"<error: {e}>"
        try:
            neg = chat_nonstream(base, q, neg_sys, timeout=timeout)
        except Exception as e:
            logger.exception("neg failed: %s", e)
            neg = f"<error: {e}>"
        items.append(
            {
                "index": i,
                "question": q,
                "pos_reply": pos,
                "neg_reply": neg,
            }
        )

    doc = {
        "gemma_url": gemma_url,
        "trait_bundle": str(bundle_path.resolve()),
        "trait_label": artifact.trait_label,
        "paragraph_cap": paragraph_cap,
        "items": items,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out_path
