"""Shared defaults for persona extraction / Vertex judging."""

from __future__ import annotations

import os
from pathlib import Path

PERSONA_RUNS_DIR = Path(
    os.environ.get("PERSONA_RUNS_DIR", "persona_runs")
).expanduser().resolve()

DEFAULT_VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "") or None
DEFAULT_VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
DEFAULT_JUDGE_MODEL = os.environ.get("PERSONA_JUDGE_MODEL", "gemini-2.5-flash")
DEFAULT_ARTIFACT_MODEL = os.environ.get("PERSONA_ARTIFACT_MODEL", DEFAULT_JUDGE_MODEL)

JUDGE_POS_KEEP_IF_SCORE_GT = int(os.environ.get("PERSONA_JUDGE_POS_MIN", "50"))
JUDGE_NEG_KEEP_IF_SCORE_LT = int(os.environ.get("PERSONA_JUDGE_NEG_MAX", "50"))
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = int(os.environ.get("JUDGE_MAX_OUTPUT_TOKENS", "16384"))

PERSONA_FULL_SCALE = os.environ.get("PERSONA_FULL_SCALE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
