#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

pkill -f "ssv_dsweep.py" 2>/dev/null || true
sleep 1

echo "=== beta-auto=0.2 d=5,10,20 started $(date -Is) ==="
.venv/bin/python3 -u scripts/ssv_dsweep.py \
  --trait good \
  --layer 16 \
  --ds 5,10,20 \
  --skip-gates \
  --build-full-cache \
  --beta-auto 0.2 \
  --out persona_runs/dnd_good_scale/sae/ssv_dsweep_betaauto_l16.json \
  2>&1

echo "=== beta-auto DONE $(date -Is) ==="
