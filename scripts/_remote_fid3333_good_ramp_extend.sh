#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

echo "=== GOOD fid 3333 alpha 7-15 x 20Q started $(date -Is) ==="
.venv/bin/python3 -u scripts/fid_scale_to_incoherence.py \
  --trait good --layer 16 --fid 3333 --sign 1.0 \
  --alphas 7,8,9,10,12,15,18,20 \
  --n-questions 20 --judge-workers 10 \
  --out persona_runs/dnd_good_scale/sae/fid3333_scale_to_incoherence_l16_extend.json \
  2>&1
echo "=== DONE $(date -Is) ==="
