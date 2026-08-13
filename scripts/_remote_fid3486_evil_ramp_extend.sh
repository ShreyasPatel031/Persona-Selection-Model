#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

echo "=== EVIL fid 3486 alpha 7+ until incoherence started $(date -Is) ==="
.venv/bin/python3 -u scripts/fid_scale_to_incoherence.py \
  --trait evil --layer 15 --fid 3486 --sign 1.0 \
  --alphas 7,8,9,10,12,15,18,20,25,30 \
  --n-questions 20 --judge-workers 10 \
  --out persona_runs/dnd_evil/sae/fid3486_scale_to_incoherence_l15_extend.json \
  2>&1
echo "=== DONE $(date -Is) ==="
