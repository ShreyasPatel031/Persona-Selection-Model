#!/usr/bin/env bash
# Run d-sweep for one trait (used on worker VMs).
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

DS="20,30,50,80,100,150,200,500"
LOG="$HOME/gemma-chat/logs/dsweep_${trait}.log"

echo "=== d-sweep $trait started $(date -Is) judge_workers=$judge_workers ===" | tee "$LOG"

PYTHONPATH=. "$PY" -u scripts/ssv_stage2_test.py \
  --trait "$trait" \
  --n-questions 20 \
  --ds "$DS" \
  --judge-workers "$judge_workers" \
  2>&1 | tee -a "$LOG"

echo "=== $trait d-sweep DONE $(date -Is) ===" | tee -a "$LOG"
