#!/usr/bin/env bash
# OMP bubble viz pipeline for gender_male @ L15, K sweep through K=20 only.
set -euo pipefail

cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"
LOG_DIR="$HOME/gemma-chat/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/gender_male_omp_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== gender_male OMP pipeline started $(date -Is) ==="

echo "=== Step 1: OMP decomposition (k-max=20) ==="
PYTHONPATH=. "$PY" -u scripts/omp_decompose.py --trait male --k-max 20

echo "=== Step 2: EMD K sweep K=5,10,15,20 ==="
PYTHONPATH=. "$PY" -u scripts/ssv_omp_k_sweep.py \
  --trait male \
  --method omp \
  --steer-mode emd \
  --ks 5,10,15,20 \
  --n-questions 20 \
  --gen-batch-size 20 \
  --judge-workers 20 \
  --run-dense-ref

echo "=== Step 3: Logit lens top-20 ==="
PYTHONPATH=. "$PY" -u scripts/ssv_feature_logit_lens.py \
  --decomp-traits male \
  --decomp-top-k 20 \
  --layer 15

echo "=== Step 4: Gemini labels top-20 ==="
PYTHONPATH=. "$PY" -u scripts/omp_lens_interp.py \
  --traits male \
  --decomp-top-k 20 \
  --project "$GOOGLE_CLOUD_PROJECT"

echo "=== Step 5: Rebuild viz data (male only via full script) ==="
PYTHONPATH=. "$PY" -u scripts/rebuild_ssv_bubble_viz_omp_data.py

echo "=== DONE $(date -Is) ==="
