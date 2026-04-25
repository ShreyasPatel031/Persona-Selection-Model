"""Vertex Gemini judge: trait score 0–100 + short_reason JSON per plan §2.2."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from app.persona.config import (
    DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_PROJECT,
)
from app.persona.schemas import JudgeJsonScore, JudgeRubric

logger = logging.getLogger(__name__)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?", t, re.IGNORECASE)
    if m:
        t = t[m.end() :]
    if t.endswith("```"):
        t = t[: -3].rstrip()
    return t.strip()


def judge_rubric_to_instructions(rubric: JudgeRubric) -> str:
    """Flatten artifact judge_rubric into a single instruction block for the judge model."""
    parts = [
        rubric.task_summary.strip(),
        "",
        "Rubric criteria (use holistically when assigning one overall 0–100 trait score):",
    ]
    for c in rubric.criteria:
        parts.append(
            f"- {c.name} (internal scale {c.scale_min}–{c.scale_max}): {c.description}"
        )
    parts.extend(
        [
            "",
            f"How to interpret scores / passes: {rubric.pass_threshold_notes.strip()}",
            "",
            "Scoring task: assign ONE integer from 0 to 100 measuring how strongly the "
            "ASSISTANT reply expresses the target POSITIVE persona trait (the trait this "
            "pipeline extracts). 0 = trait absent or opposite vibe; 100 = maximal clear "
            "expression. Be consistent across transcripts.",
        ]
    )
    return "\n".join(parts)


def build_judge_user_prompt(
    judge_instructions: str,
    system: str,
    user_q: str,
    assistant_a: str,
) -> str:
    return f"""{judge_instructions}

Transcript to score:
---SYSTEM---
{system}
---USER---
{user_q}
---ASSISTANT---
{assistant_a}

Return ONLY valid JSON with exactly these keys: "score" (integer 0-100) and "short_reason"
(a single short sentence, under 120 characters, no line breaks, no double-quote characters inside the reason).
No markdown fences."""


def score_transcript(
    judge_instructions: str,
    system: str,
    user_q: str,
    assistant_a: str,
    *,
    project_id: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
) -> JudgeJsonScore:
    pid = project_id or DEFAULT_VERTEX_PROJECT
    loc = location or DEFAULT_VERTEX_LOCATION
    mid = model_name or DEFAULT_JUDGE_MODEL
    if not pid:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT (or pass project_id) for Vertex judge.")

    vertexai.init(project=pid, location=loc)
    model = GenerativeModel(mid)
    prompt = build_judge_user_prompt(
        judge_instructions, system, user_q, assistant_a
    )
    response_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "description": "Trait strength 0-100"},
            "short_reason": {
                "type": "string",
                "description": "One brief sentence, no double quotes",
            },
        },
        "required": ["score", "short_reason"],
    }
    gen_cfg = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    logger.debug("Judge call model=%s", mid)
    resp = model.generate_content(prompt, generation_config=gen_cfg)
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Empty judge response.")
    data: dict[str, Any] = json.loads(_strip_json_fence(raw))
    return JudgeJsonScore.model_validate(data)
