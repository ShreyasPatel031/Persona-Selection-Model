#!/usr/bin/env bash
# Logit lens for SSV features (262k) — sequential layer runs on CPU.
set -euo pipefail
cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"
export PERSONA_FORCE_CPU=1

PY=".venv/bin/python3 -u scripts/ssv_feature_logit_lens.py"

echo "=== L16 good+evil (force recompute with RMSNorm-folded lm_head) ==="
$PY --layer 16 --fids-from good,evil --device cpu --force \
  > logs/ssv_logit_lens_l16.log 2>&1

echo "=== L15 lawful+chaotic (force recompute) ==="
$PY --layer 15 --fids-from lawful,chaotic --device cpu --force \
  > logs/ssv_logit_lens_l15.log 2>&1

echo "Done. Cache sizes:"
wc -c persona_runs/_shared/l16_262k_logit_lens_cache.json \
      persona_runs/_shared/l15_262k_logit_lens_cache.json 2>/dev/null || true
