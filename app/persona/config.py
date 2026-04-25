import os
from pathlib import Path

# All pipeline outputs live under this directory (gitignored).
PERSONA_RUNS_DIR = Path(os.environ.get("PERSONA_RUNS_DIR", "persona_runs")).resolve()

DEFAULT_VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
DEFAULT_VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
DEFAULT_JUDGE_MODEL = os.environ.get("PERSONA_JUDGE_MODEL", "gemini-2.5-flash")
DEFAULT_ARTIFACT_MODEL = os.environ.get(
    "PERSONA_ARTIFACT_MODEL", DEFAULT_JUDGE_MODEL
)

# Step C filter (paper §2.2): keep pos if score > threshold, neg if score < threshold.
JUDGE_POS_KEEP_IF_SCORE_GT = int(os.environ.get("PERSONA_JUDGE_POS_MIN", "50"))
JUDGE_NEG_KEEP_IF_SCORE_LT = int(os.environ.get("PERSONA_JUDGE_NEG_MAX", "50"))
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = int(
    os.environ.get("JUDGE_MAX_OUTPUT_TOKENS", "16384")
)

# Step B artifact scale (plan §2.1): 0 = pilot (1 contrast pair, 8+8 questions), 1 = paper-style (5 pairs, 20+20).
PERSONA_FULL_SCALE = os.environ.get("PERSONA_FULL_SCALE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
