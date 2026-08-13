#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

pkill -f ssv_real_lm_test.py 2>/dev/null || true
sleep 1

nohup .venv/bin/python3 -u scripts/ssv_real_lm_test.py \
  --trait good --d 50 --n-iter 100 --lm-batch-size 1 \
  --normalize-dist --lambda-lm 0.5 \
  > logs/ssv_real_lm_normdist.log 2>&1 &

echo "PID=$!"
sleep 5
tail -20 logs/ssv_real_lm_normdist.log
