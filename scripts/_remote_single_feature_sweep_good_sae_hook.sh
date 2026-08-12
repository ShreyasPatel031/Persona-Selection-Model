#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

echo "=== good SAE-hook single-feature sweep started $(date -Is) ==="
.venv/bin/python3 -u scripts/single_feature_scale_sweep.py \
  --trait good \
  --layer 16 \
  --alpha-dense 2.0 \
  --method sae_hook \
  --out persona_runs/dnd_good_scale/sae/single_feature_scale_sweep_sae_hook_l16.json \
  2>&1
echo "=== good SAE-hook sweep DONE $(date -Is) ==="
