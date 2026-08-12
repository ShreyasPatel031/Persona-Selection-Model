#!/usr/bin/env bash
# Consolidate all traits: backup bundles, regen full-scale, step-c, z-cache, stage2 d-sweep.
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
MAIN_LOG="$LOG_DIR/consolidate_traits_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== consolidate traits pipeline started $(date -Is) ==="
echo "log: $MAIN_LOG"

# --- Phase 0: backup pilot bundles ---
BACKUP="$HOME/gemma-chat/persona_runs/_backup_pilot_${TS}"
mkdir -p "$BACKUP"
for rid in dnd_evil dnd_lawful dnd_chaotic; do
  if [ -f "persona_runs/$rid/artifacts/trait_bundle.json" ]; then
    mkdir -p "$BACKUP/$rid/artifacts"
    cp -a "persona_runs/$rid/artifacts/trait_bundle.json" "$BACKUP/$rid/artifacts/"
    echo "Backed up $rid bundle -> $BACKUP/$rid/artifacts/"
  fi
done

# --- Phase 1: regenerate evil/lawful/chaotic bundles (full scale) ---
PERSONA_FULL_SCALE=1 "$PY" -u <<'PY'
import os, subprocess, sys
sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
from app.persona.vector_compose import DND_TRAIT_DESCRIPTIONS

pairs = [
    ("dnd_evil", "Evil"),
    ("dnd_lawful", "Lawful"),
    ("dnd_chaotic", "Chaotic"),
]
for rid, label in pairs:
    desc = DND_TRAIT_DESCRIPTIONS[label]
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

# Verify bundle sizes
"$PY" - <<'PY'
import json
from pathlib import Path
for rid in ["dnd_evil", "dnd_lawful", "dnd_chaotic", "dnd_good_scale"]:
    p = Path("persona_runs") / rid / "artifacts" / "trait_bundle.json"
    b = json.loads(p.read_text())
    print(
        rid,
        "pairs", len(b.get("contrast_scenarios", [])),
        "ext", len(b.get("extraction_questions", [])),
        "eval", len(b.get("eval_questions", [])),
        flush=True,
    )
PY

# --- Phase 2: step-c for all traits (in-process GPU; no uvicorn) ---
stop_uvicorn_for_gpu() {
  if curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    echo "Stopping uvicorn to free GPU for in-process step-c..."
    pkill -f "uvicorn app.main:app" || true
    sleep 5
  fi
}

stop_uvicorn_for_gpu

for rid in dnd_evil dnd_lawful dnd_chaotic dnd_good_scale; do
  echo ""
  echo "=== step-c $rid ==="
  "$PY" -m app.persona.run step-c \
    --run-id "$rid" \
    --local-gpu \
    --max-new-tokens 200 \
    --judge-workers 16 \
    --questions-source scenarios \
    --rollouts-per-q 10 \
    --no-paragraph-cap \
    --project "$GOOGLE_CLOUD_PROJECT"
  wc -l "persona_runs/$rid/rollouts/rollouts.jsonl"
done

# --- Phase 3: rebuild z-caches ---
for trait in evil lawful chaotic good; do
  echo ""
  echo "=== z-cache rebuild trait=$trait ==="
  PYTHONPATH=. "$PY" -u scripts/sae_ssv_optimize.py \
    --trait "$trait" \
    --optimize-only \
    --skip-ref \
    --ks 50 \
    --n-iter 1 \
    --n-questions 5
done

# --- Phase 4: fine d-sweep (stage2) ---
DS="5,10,20,30,40,50,60,80,100,150,200,500"
for trait in evil lawful chaotic good; do
  echo ""
  echo "=== stage2 d-sweep trait=$trait ==="
  PYTHONPATH=. "$PY" -u scripts/ssv_stage2_test.py \
    --trait "$trait" \
    --n-questions 20 \
    --ds "$DS"
done

echo ""
echo "=== consolidate traits pipeline finished $(date -Is) ==="
