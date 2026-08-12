#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

echo "=== Chen M.3.2 fid 3333 scale ramp started $(date -Is) ==="
.venv/bin/python3 -u scripts/chen_m32_feature_sweep.py \
  --trait good \
  --layer 16 \
  --n-features 1 \
  --n-questions 5 \
  --judge-workers 10 \
  --out persona_runs/dnd_good_scale/sae/chen_m32_fid3333_l16.json \
  2>&1
echo "=== Chen M.3.2 fid 3333 DONE $(date -Is) ==="
