#!/usr/bin/env bash
# Re-run sufficiency matrix with SSV/OMP weighted residuals (fix v1 unweighted bug).
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

OUT="persona_runs/dnd_good_scale/sae/sufficiency_baseline_matrix_v2_l16.json"
LOG="logs/sufficiency_baseline_v2.log"

echo "=== Sufficiency v2 weighted $(date -Is) ===" | tee "$LOG"
.venv/bin/python3 -u scripts/sufficiency_baseline_matrix.py \
  --trait good --layer 16 --alpha 2.0 \
  --n-questions 20 --judge-workers 2 --resume \
  --out "$OUT" \
  2>&1 | tee -a "$LOG"
echo "=== DONE $(date -Is) ===" | tee -a "$LOG"
