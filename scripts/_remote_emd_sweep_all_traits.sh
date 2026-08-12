#!/usr/bin/env bash
# 16-K OMP EMD sweep for evil, lawful, chaotic @ L15 on gemma-mvp
# Replicates the Good trait sweep (omp_k_sweep_l15_20q_emd.json)
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

KS="5,10,15,20,25,30,40,50,75,100,150,200,300,450,750,1000"

for TRAIT in evil lawful chaotic; do
  LOG="logs/omp_k_sweep_l15_20q_emd_${TRAIT}.log"
  echo "=== EMD K sweep TRAIT=$TRAIT $(date -Is) ===" | tee -a "$LOG"
  .venv/bin/python3 -u scripts/ssv_omp_k_sweep.py \
    --trait "$TRAIT" --method omp --steer-mode emd --scale 3.0 \
    --ks "$KS" --n-questions 20 --judge-workers 20 --gen-batch-size 20 \
    --run-dense-ref --resume \
    --out "persona_runs/dnd_${TRAIT}/sae/omp_k_sweep_l15_20q_emd.json" \
    2>&1 | tee -a "$LOG"
  echo "=== DONE TRAIT=$TRAIT $(date -Is) ===" | tee -a "$LOG"
done

echo "=== ALL TRAITS DONE $(date -Is) ==="
