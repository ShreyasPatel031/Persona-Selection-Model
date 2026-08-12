#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

pkill -f "ssv_dsweep.py" 2>/dev/null || true
sleep 1

echo "=== reopt d=5 started $(date -Is) ==="
.venv/bin/python3 -u scripts/ssv_dsweep.py \
  --trait good \
  --layer 16 \
  --ds 5 \
  --skip-gates \
  --reopt-iters 100 \
  --build-full-cache \
  --out persona_runs/dnd_good_scale/sae/ssv_dsweep_reopt_l16.json \
  2>&1

echo "=== reopt d=5 DONE $(date -Is) ==="
