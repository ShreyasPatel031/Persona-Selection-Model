#!/usr/bin/env bash
# OMP fine K sweep @ L15 on gemma-mvp
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

KS="5,10,15,20,25,30,40,50,75,100,150,200,300,450,750,1000"
LOG="logs/omp_k_sweep_l15_20q.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== OMP K sweep L15 $(date -Is) ==="
.venv/bin/python3 -u scripts/ssv_omp_k_sweep.py \
  --trait good --method omp \
  --ks "$KS" --n-questions 20 --judge-workers 10 --resume \
  --out persona_runs/dnd_good_scale/sae/omp_k_sweep_l15_20q.json
echo "=== DONE $(date -Is) ==="
