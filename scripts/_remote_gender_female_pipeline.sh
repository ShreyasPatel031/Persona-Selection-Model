#!/usr/bin/env bash
# Step D + OMP bubble viz for gender_female @ L15, K sweep through K=20.
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
LOG="$LOG_DIR/gender_female_pipeline_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== gender_female pipeline started $(date -Is) ==="

echo "=== Step D (max-per-arm 200) ==="
"$PY" -m app.persona.run step-d --run-id gender_female --max-per-arm 200

echo "=== Split-half ==="
"$PY" - <<PY
import json
from pathlib import Path
s = json.loads(Path("persona_runs/gender_female/vectors/summary.json").read_text())
sh = s.get("split_half_cosine", {})
print("kept_pos", s.get("kept_pos"), "kept_neg", s.get("kept_neg"))
print("split_half", sh.get("mean_cosine_at_argmax_norm"), sh.get("interpretation"))
PY

echo "=== OMP decompose k-max=20 ==="
PYTHONPATH=. "$PY" -u scripts/omp_decompose.py --trait female --k-max 20

echo "=== EMD K sweep K=5,10,15,20 ==="
PYTHONPATH=. "$PY" -u scripts/ssv_omp_k_sweep.py \
  --trait female \
  --method omp \
  --steer-mode emd \
  --ks 5,10,15,20 \
  --n-questions 20 \
  --gen-batch-size 20 \
  --judge-workers 20 \
  --run-dense-ref

echo "=== Logit lens top-20 ==="
PYTHONPATH=. "$PY" -u scripts/ssv_feature_logit_lens.py \
  --decomp-traits female \
  --decomp-top-k 20 \
  --layer 15

echo "=== Gemini labels top-20 ==="
PYTHONPATH=. "$PY" -u scripts/omp_lens_interp.py \
  --traits female \
  --decomp-top-k 20 \
  --project "$GOOGLE_CLOUD_PROJECT"

echo "=== Rebuild viz data ==="
PYTHONPATH=. "$PY" -u scripts/rebuild_ssv_bubble_viz_omp_data.py

echo "=== DONE $(date -Is) ==="
