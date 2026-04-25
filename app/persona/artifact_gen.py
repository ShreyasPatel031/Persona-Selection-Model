"""Step B: ask Vertex Gemini to emit a validated PersonaTraitArtifact JSON object."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from app.persona.config import (
    DEFAULT_ARTIFACT_MODEL,
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_PROJECT,
    PERSONA_FULL_SCALE,
)
from app.persona.schemas import PersonaTraitArtifact

logger = logging.getLogger(__name__)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?", t, re.IGNORECASE)
    if m:
        t = t[m.end() :]
    if t.endswith("```"):
        t = t[: -3].rstrip()
    return t.strip()


def parse_artifact_json(raw: str) -> dict[str, Any]:
    """Parse model output; tolerate optional markdown fences."""
    text = _strip_json_fence(raw)
    return json.loads(text)


def build_artifact_user_prompt(trait_label: str, trait_description: str) -> str:
    if PERSONA_FULL_SCALE:
        scale_block = """
SCALE MODE: **full** (paper-style). You MUST include these additional keys and sizes:
- "contrastive_system_prompts": array of **exactly 5** objects, each {{"positive": string, "negative": string}}.
  Each pair must use **different wording and angles** (e.g. tone, metaphor, role) while keeping the same law–chaos (or trait) axis.
  Pair 1 is the primary pair: set "pos_system_prompt" and "neg_system_prompt" to **identical** to pair 1's positive and negative.
- "extraction_questions": array of **exactly 20** distinct user strings (rollouts run under each pair).
- "eval_questions": array of **exactly 20** situational user strings (held-out).
- "contrast_scenarios": array of **8–12** short user strings.
"""
        extraction_eval = ""
    else:
        scale_block = """
SCALE MODE: **pilot**. Use a single primary pair only (pos_system_prompt / neg_system_prompt); you may omit
"contrastive_system_prompts" or set it to a one-element array mirroring that pair.
"""
        extraction_eval = ""

    pilot_schema = """
- "pos_system_prompt": string — system prompt that strongly elicits the trait (humor rules, institutional/gesture notes, willingness to be judged for what one mocks, if relevant).
  It MUST explicitly require that every assistant reply stays within one short paragraph (5 sentences max), plain prose, no lists or essays.
- "neg_system_prompt": string — the author's contrast persona: conventionally correct, proper, surface-true speech;
  avoids giving offense; does not worst-case or mock; does not use gallows humor; stays within "what one is supposed to say"
  and how things seem on the surface unless the notes demand a different neg voice.
  It MUST include the same one-paragraph / 5-sentence cap as pos_system_prompt.
- "contrast_scenarios": array of 6–10 short user strings (one sentence each) where pos vs neg should diverge
- "extraction_questions": array of 10–16 distinct user strings (questions or prompts) for rollouts under pos vs neg system
- "eval_questions": array of 6–10 **situational** user strings: concrete, plausible real-life moments (not abstract essay topics).
  Each item should be one or two sentences setting a scene — work, family, money, transit, bureaucracy, neighbors, school,
  healthcare admin, roommates, customer service, dating apps, etc. Use first person ("I…") or direct "You…" address.
  Avoid prompts like "Discuss the concept of X" or "What is your take on society"; prefer "Your landlord just…", "At the pharmacy…", etc.
"""

    full_schema_extra = """
- "contrastive_system_prompts": array of 5 objects {{"positive", "negative"}} (required in full scale; see SCALE MODE above).
- "pos_system_prompt": string — **copy of contrastive_system_prompts[0].positive**
- "neg_system_prompt": string — **copy of contrastive_system_prompts[0].negative**
- "contrast_scenarios": array of 8–12 short user strings
- "extraction_questions": array of exactly 20 distinct user strings
- "eval_questions": array of exactly 20 situational user strings (same style as pilot eval_questions)
"""

    schema_body = full_schema_extra if PERSONA_FULL_SCALE else pilot_schema

    return f"""You are designing data for a research pipeline that extracts a "persona direction" from a language model
using contrastive activations (positive persona vs a deliberately chosen negative / contrast persona).

Given the TARGET TRAIT and the AUTHOR'S NORMATIVE / STYLE NOTES, produce ONE JSON object that matches the schema below.
Be specific to this trait. Questions and scenarios must be answerable in plain chat (no tools, no web).

If the notes specify what the "opposite" of the trait is, treat that as the contrast axis for neg_system_prompt.
Typical contrast (when described by the author): not cheerful optimism, but a voice of **conventional propriety** —
always the socially "right" thing to say, never offensive, never transgressive, never probing beneath how things
comfortably appear; "anonymously good," never having questioned the scene. The pos persona should match the author's
positive target (e.g. humorous pessimism, institutional edge, self-exposure to judgment about what is mocked).

TARGET TRAIT (short label):
{trait_label}

TRAIT NOTES (what counts as in-character; opposition / contrast persona; humor / institutions / justification rules):
{trait_description}
{scale_block}
Return ONLY valid JSON (no markdown) with exactly these keys and types:

- "schema_version": string, must be "1"
- "trait_label": string (you may echo or slightly tighten the label)
- "trait_description": string (clear paraphrase combining label + notes, including the contrast axis)
{schema_body}
- "judge_rubric": object with
    - "task_summary": string — what the judge is scoring
    - "criteria": array of 4–6 objects, each {{"name", "description", "scale_min": 1, "scale_max": 5}}
    - "pass_threshold_notes": string — e.g. which criteria must be high for a "pos" rollout to count as on-persona

All strings must be non-empty. Keep scenarios and questions in English unless the trait explicitly requires another language.
{extraction_eval}
"""


def generate_trait_artifact(
    trait_label: str,
    trait_description: str,
    *,
    project_id: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.35,
    max_output_tokens: int = 8192,
) -> PersonaTraitArtifact:
    pid = project_id or DEFAULT_VERTEX_PROJECT
    loc = location or DEFAULT_VERTEX_LOCATION
    mid = model_name or DEFAULT_ARTIFACT_MODEL
    if not pid:
        raise ValueError(
            "Set GOOGLE_CLOUD_PROJECT (or pass project_id) for Vertex AI."
        )

    vertexai.init(project=pid, location=loc)
    model = GenerativeModel(mid)
    prompt = build_artifact_user_prompt(trait_label, trait_description)
    out_tok = max_output_tokens
    if PERSONA_FULL_SCALE and out_tok < 16384:
        out_tok = 16384  # five contrast pairs + 20+20 questions need headroom

    gen_cfg = GenerationConfig(
        temperature=temperature,
        max_output_tokens=out_tok,
        response_mime_type="application/json",
    )
    logger.info("Calling Vertex model %s (%s) for trait artifact…", mid, loc)
    resp = model.generate_content(
        prompt,
        generation_config=gen_cfg,
    )
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Empty response from artifact model.")
    data = parse_artifact_json(raw)
    return PersonaTraitArtifact.model_validate(data)
