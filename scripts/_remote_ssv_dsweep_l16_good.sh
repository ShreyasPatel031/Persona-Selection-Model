#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"

pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

mkdir -p "$HOME/gemma-chat/logs"
LOG="$HOME/gemma-chat/logs/ssv_dsweep_good_l16.log"

echo "=== SSV d-sweep good L16 started $(date -Is) ===" | tee "$LOG"

PYTHONPATH=. "$PY" -u scripts/ssv_dsweep.py \
  --trait good \
  --layer 16 \
  --n-questions 5 \
  --n-iter 100 \
  --ds 5,10,20,50,100 \
  --scales 1,2,3,5,8 \
  --judge-workers 16 \
  --resume 2>&1 | tee -a "$LOG"

echo "=== good SSV d-sweep L16 DONE $(date -Is) ===" | tee -a "$LOG"
