#!/usr/bin/env bash
# Run SAE-SSV full sweeps sequentially: evil -> lawful -> chaotic (foreground)
set -euo pipefail

cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

LOG="logs/sae_ssv_all_traits.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== SAE-SSV all traits started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

for TRAIT in evil lawful chaotic; do
  echo ""
  echo "========== Starting $TRAIT =========="
  bash scripts/_remote_sae_ssv_trait.sh "$TRAIT"
  echo "=== $TRAIT finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done

echo ""
echo "=== ALL TRAITS COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
