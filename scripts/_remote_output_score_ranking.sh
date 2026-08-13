#!/usr/bin/env bash
# Trait-directional Arad output-score ranking (CPU only, ~10 min).
# Run ON gemma-dsweep-good (or any VM with ~/gemma-chat + persona vectors).
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
export PERSONA_FORCE_CPU=1

LOG="logs/output_score_ranking_l15.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Arad output-score ranking L15 $(date -Is) ==="
.venv/bin/python3 -u scripts/output_score_ranking.py \
  --trait good --layer 15 \
  --out persona_runs/dnd_good_scale/sae/sae_output_score_l15.json \
  --ranking-out persona_runs/dnd_good_scale/sae/output_score_ranking_l15.json
echo "=== DONE $(date -Is) ==="
