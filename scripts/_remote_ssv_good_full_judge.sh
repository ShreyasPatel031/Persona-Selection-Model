#!/usr/bin/env bash
# Run SSV optimization + judging for Good at all K levels.
# Uses existing z-cache so skips model activation collection.
set -euo pipefail
cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

OUT="persona_runs/dnd_good_scale/sae/sae_ssv_full_judged_262k_l16.json"
LOG="logs/ssv_good_full_judge.log"

nohup .venv/bin/python3 -u scripts/sae_ssv_optimize.py \
  --trait good \
  --ks 5,10,20,50,100,128,200,256,512,750,1000 \
  --n-iter 100 \
  --lr 0.05 \
  --lambda-lm 0.5 \
  --beta 0.01 \
  --skip-collect \
  --skip-ref \
  --n-questions 5 \
  --out "$OUT" \
  > "$LOG" 2>&1 &

echo "PID=$! log=$LOG"
sleep 5
tail -20 "$LOG" 2>/dev/null || true
