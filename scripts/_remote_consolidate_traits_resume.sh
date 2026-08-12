#!/usr/bin/env bash
# Resume consolidate pipeline from step-c (bundles already regenerated).
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
MAIN_LOG="$LOG_DIR/consolidate_traits_resume_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== consolidate traits RESUME started $(date -Is) ==="

stop_uvicorn_for_gpu() {
  if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    echo "Stopping uvicorn to free GPU for in-process step-c..."
    pkill -9 -f "uvicorn app.main:app" || true
    sleep 8
  fi
  if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
  fi
}

stop_uvicorn_for_gpu

# --- Phase 2: step-c ---
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

# --- Phase 3: z-cache ---
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

# --- Phase 4: stage2 d-sweep ---
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
echo "=== consolidate traits RESUME finished $(date -Is) ==="
