#!/usr/bin/env bash
# Minimal consolidation: 1 pair × extraction Qs × 10 rollouts per trait.
# Gives ~400 lines per trait (~50 min each on T4, ~3.5h total).
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
MAIN_LOG="$LOG_DIR/consolidate_minimal_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== consolidate minimal started $(date -Is) ==="

# Kill anything using the GPU
pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
sleep 3
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true

# --- Phase 2: step-c (in-process GPU, 1 pair, extraction Qs, 10 rollouts) ---
for rid in dnd_evil dnd_lawful dnd_chaotic dnd_good_scale; do
  echo ""
  echo "=== step-c $rid ==="
  "$PY" -m app.persona.run step-c \
    --run-id "$rid" \
    --local-gpu \
    --max-new-tokens 200 \
    --judge-workers 16 \
    --questions-source extraction \
    --rollouts-per-q 10 \
    --max-pairs 1 \
    --no-paragraph-cap \
    --project "$GOOGLE_CLOUD_PROJECT"
  wc -l "persona_runs/$rid/rollouts/rollouts.jsonl"
done

# --- Phase 2.5: step-d (dense persona vectors from new rollouts) ---
for rid in dnd_evil dnd_lawful dnd_chaotic dnd_good_scale; do
  echo ""
  echo "=== step-d $rid ==="
  "$PY" -m app.persona.run step-d --run-id "$rid"
done

# --- Phase 3: z-cache rebuild + minimal SSV optimize ---
# This re-collects latents from new rollouts, rebuilds z-cache, then does
# a quick 1-iteration optimize just to validate the cache is good.
for trait in evil lawful chaotic good; do
  echo ""
  echo "=== z-cache + SSV optimize trait=$trait ==="
  PYTHONPATH=. "$PY" -u scripts/sae_ssv_optimize.py \
    --trait "$trait" \
    --optimize-only \
    --skip-ref \
    --ks 50 \
    --n-iter 1 \
    --n-questions 5
done

echo "=== Phase 3 complete. z-caches rebuilt for all traits ==="

# --- Phase 4: stage2 classifier d-sweep ---
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
echo "=== consolidate minimal finished $(date -Is) ==="
