#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

pkill -f sae_ssv_optimize.py 2>/dev/null || true
pkill -f probe_steer_sweep.py 2>/dev/null || true
sleep 1

nohup .venv/bin/python3 -u scripts/sae_ssv_optimize.py \
  --trait good \
  --n-questions 5 \
  --ks 5,10,20,50,100,128,200,256,512,750,1000 \
  --n-iter 100 \
  --lr 0.05 \
  --lambda-lm 0.5 \
  --beta 0.01 \
  --skip-collect \
  --skip-ref \
  --out persona_runs/dnd_good_scale/sae/sae_ssv_results_262k_l16.json \
  > logs/sae_ssv.log 2>&1 &

echo "PID=$!"
sleep 5
tail -10 logs/sae_ssv_full.log 2>/dev/null || echo "(log not ready yet)"
