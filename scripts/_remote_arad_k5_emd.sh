#!/usr/bin/env bash
# K=5 EMD steer with Arad output-score feature file on gemma-dsweep-good.
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

LOG="logs/ssv_k_sweep_l15_20q_emd_arad.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Arad output-score K=5 EMD $(date -Is) ==="
.venv/bin/python3 -u scripts/ssv_omp_k_sweep.py \
  --trait good --method ssv --steer-mode emd --scale 3.0 \
  --feature-file persona_runs/dnd_good_scale/sae/sae_output_score_l15.json \
  --ks 5 --n-questions 20 --judge-workers 10 --gen-batch-size 4 \
  --experiment arad_output_score \
  --out persona_runs/dnd_good_scale/sae/ssv_k_sweep_l15_20q_emd_arad.json
echo "=== DONE $(date -Is) ==="
