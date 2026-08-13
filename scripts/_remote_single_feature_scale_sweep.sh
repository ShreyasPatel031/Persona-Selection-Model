#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

echo "=== single_feature_scale_sweep started $(date -Is) ==="
.venv/bin/python3 -u scripts/single_feature_scale_sweep.py \
  --trait good \
  --layer 16 \
  2>&1
echo "=== single_feature_scale_sweep DONE $(date -Is) ==="
