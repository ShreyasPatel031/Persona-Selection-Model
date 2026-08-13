#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

pkill -f probe_steer_sweep.py 2>/dev/null || true
sleep 1

nohup .venv/bin/python3 -u scripts/probe_steer_sweep.py \
  --trait good \
  --n-questions 5 \
  --k-select 512 \
  --skip-collect \
  --out persona_runs/dnd_good_scale/sae/probe_subspace_262k_l16.json \
  > logs/probe_subspace.log 2>&1 &

echo "PID=$!"
sleep 3
tail -5 logs/probe_subspace.log 2>/dev/null || echo "(log not ready yet)"
