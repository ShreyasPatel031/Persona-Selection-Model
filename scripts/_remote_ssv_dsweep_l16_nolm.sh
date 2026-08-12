#!/usr/bin/env bash
# Run SSV d-sweep good L16 with lambda_lm=0 (no LM penalty), skip gates, full cache.
set -euo pipefail

cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"

pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "scripts/ssv_dsweep" 2>/dev/null || true
sleep 2

mkdir -p "$HOME/gemma-chat/logs"
rm -f persona_runs/dnd_good_scale/sae/ssv_dsweep_residual_full_l16.json
LOG="$HOME/gemma-chat/logs/ssv_dsweep_good_l16_nolm.log"

echo "=== SSV d-sweep good L16 lambda_lm=0 started $(date -Is) ===" | tee "$LOG"

PYTHONPATH=. "$PY" -u scripts/ssv_dsweep.py \
  --trait good \
  --layer 16 \
  --build-full-cache \
  --lambda-lm 0 \
  --n-questions 5 \
  --n-iter 100 \
  --ds 5,10,20,50,100 \
  --scales 1,2,3,5,8 \
  --judge-workers 16 \
  --skip-gates 2>&1 | tee -a "$LOG"

echo "=== good SSV L16 nolm DONE $(date -Is) ===" | tee -a "$LOG"
