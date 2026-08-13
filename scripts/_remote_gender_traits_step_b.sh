#!/usr/bin/env bash
# Step B (full-scale bundles) for gender_male + gender_female on gemma-dsweep-good.
set -euo pipefail

cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"
LOG_DIR="$HOME/gemma-chat/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/gender_traits_step_b_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== gender traits Step B started $(date -Is) ==="
echo "log: $MAIN_LOG"
echo "host: $(hostname)"

PERSONA_FULL_SCALE=1 "$PY" -u <<'PY'
import os, subprocess, sys
sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
from app.persona.vector_compose import GENDER_TRAIT_DESCRIPTIONS, GENDER_TRAIT_RUNS

for label, rid in GENDER_TRAIT_RUNS.items():
    desc = GENDER_TRAIT_DESCRIPTIONS[label]
    print(f"\n=== step-b {rid} ({label}) PERSONA_FULL_SCALE=1 ===", flush=True)
    env = os.environ.copy()
    env["PERSONA_FULL_SCALE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.persona.run",
            "step-b",
            "--trait",
            label,
            "--trait-description",
            desc,
            "--run-id",
            rid,
        ],
        check=True,
        env=env,
    )
PY

echo ""
echo "=== verify bundle sizes ==="
"$PY" - <<'PY'
import json
from pathlib import Path
for rid in ["gender_male", "gender_female"]:
    p = Path("persona_runs") / rid / "artifacts" / "trait_bundle.json"
    if not p.is_file():
        print(f"MISSING {p}", flush=True)
        raise SystemExit(1)
    b = json.loads(p.read_text())
    print(
        rid,
        "pairs", len(b.get("contrast_scenarios", [])),
        "ext", len(b.get("extraction_questions", [])),
        "eval", len(b.get("eval_questions", [])),
        flush=True,
    )
print("BUNDLE_DONE", flush=True)
PY

echo "=== gender traits Step B finished $(date -Is) ==="
