#!/usr/bin/env bash
# Run SSV SAE d-sweep for one trait (used on gemma-mvp).
set -euo pipefail

trait="${1:?usage: $0 <trait> [judge_workers]}"
judge_workers="${2:-16}"

cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"

pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

DS="5,10,20,50,100"
SCALES="1,2,3,5,8"
mkdir -p "$HOME/gemma-chat/logs"
LOG="$HOME/gemma-chat/logs/ssv_dsweep_${trait}.log"

echo "=== SSV d-sweep $trait started $(date -Is) judge_workers=$judge_workers ===" | tee "$LOG"

PYTHONPATH=. "$PY" -u scripts/ssv_dsweep.py \
  --trait "$trait" \
  --n-questions 5 \
  --n-iter 100 \
  --ds "$DS" \
  --scales "$SCALES" \
  --judge-workers "$judge_workers" \
  --resume 2>&1 | tee -a "$LOG"

echo "=== $trait SSV d-sweep DONE $(date -Is) ===" | tee -a "$LOG"
