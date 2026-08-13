#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

pkill -f single_feature_scale_sweep 2>/dev/null || true
sleep 1

echo "=== evil SAE-hook single-feature sweep started $(date -Is) ==="
.venv/bin/python3 -u scripts/single_feature_scale_sweep.py \
  --trait evil \
  --layer 15 \
  --alpha-dense 4.0 \
  --method sae_hook \
  --out persona_runs/dnd_evil/sae/single_feature_scale_sweep_sae_hook_l15.json \
  2>&1
echo "=== evil SAE-hook sweep DONE $(date -Is) ==="
