#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

OUT="persona_runs/dnd_good_scale/sae/chen_m32_top50_20q_l16_part1.json"
LOG="logs/chen_m32_top50_part1.log"

echo "=== Chen M.3.2 top-50 GOOD ranks 1-25 x 20Q started $(date -Is) ==="
.venv/bin/python3 -u scripts/chen_m32_feature_sweep.py \
  --trait good \
  --layer 16 \
  --top-k 50 \
  --rank-start 1 \
  --rank-end 25 \
  --n-questions 20 \
  --judge-workers 4 \
  --conditions residual_pos_only \
  --t-pass 50 \
  --out "$OUT" \
  2>&1 | tee "$LOG"
echo "=== DONE $(date -Is) ==="
