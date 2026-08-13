#!/usr/bin/env bash
# Run convergence framework GPU steps on gemma-dsweep-good.
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

SAE_DIR="persona_runs/dnd_good_scale/sae"
LOG="logs/convergence_framework_gpu.log"

echo "=== Convergence framework GPU $(date -Is) ===" | tee "$LOG"

# C1 convergence (CPU-fast; ensures VM has latest selectors)
.venv/bin/python3 -u scripts/feature_set_convergence.py \
  --sae-dir "$SAE_DIR" --k 20 \
  --out "$SAE_DIR/feature_convergence_l16_k20.json" \
  2>&1 | tee -a "$LOG"

# C2 necessity on default good (pos prompt)
.venv/bin/python3 -u scripts/necessity_default_good.py \
  --trait good --layer 16 \
  --convergence-json "$SAE_DIR/feature_convergence_l16_k20.json" \
  --n-questions 20 \
  --out "$SAE_DIR/necessity_default_good_l16.json" \
  2>&1 | tee -a "$LOG"

# C3 sufficiency + baselines
.venv/bin/python3 -u scripts/sufficiency_baseline_matrix.py \
  --trait good --layer 16 --alpha 2.0 \
  --convergence-json "$SAE_DIR/feature_convergence_l16_k20.json" \
  --n-questions 20 --judge-workers 4 \
  --out "$SAE_DIR/sufficiency_baseline_matrix_l16.json" \
  2>&1 | tee -a "$LOG"

# C4 evidence matrix
.venv/bin/python3 -u scripts/build_evidence_matrix.py \
  --sae-dir "$SAE_DIR" \
  --out "$SAE_DIR/good_feature_evidence_matrix.json" \
  2>&1 | tee -a "$LOG"

echo "=== DONE $(date -Is) ===" | tee -a "$LOG"
