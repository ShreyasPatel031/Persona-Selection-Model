#!/usr/bin/env bash
# C2-C4 at validated L15 on gemma-dsweep-good (after C1 sources exist).
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

SAE_DIR="persona_runs/dnd_good_scale/sae"
LOG="logs/convergence_framework_l15_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

exec > >(tee -a "$LOG") 2>&1
echo "=== Convergence framework L15 $(date -Is) ==="

.venv/bin/python3 -u scripts/necessity_default_good.py \
  --trait good --n-questions 20 \
  2>&1 | tee -a "$LOG"

.venv/bin/python3 -u scripts/sufficiency_baseline_matrix.py \
  --trait good --n-questions 20 --judge-workers 4 \
  2>&1 | tee -a "$LOG"

.venv/bin/python3 -u scripts/build_evidence_matrix.py \
  --trait good \
  --out "$SAE_DIR/good_feature_evidence_matrix_l15.json" \
  2>&1 | tee -a "$LOG"

echo "=== DONE $(date -Is) ==="
