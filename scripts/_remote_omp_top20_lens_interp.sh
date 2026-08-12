#!/usr/bin/env bash
# Logit lens + Gemini labels for top-20 OMP decomposition features (all traits).
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

PY="$HOME/gemma-chat/.venv/bin/python3"
TRAITS="good,evil,lawful,chaotic"
TOP_K=20
LOG="logs/omp_top20_lens_interp.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== OMP top-${TOP_K} lens + Gemini $(date -Is) ==="

echo "--- Step 1: logit lens tokens ---"
"$PY" -u scripts/ssv_feature_logit_lens.py \
  --decomp-traits "$TRAITS" \
  --decomp-top-k "$TOP_K" \
  --layer 15

echo "--- Step 2: Gemini title/desc ---"
"$PY" -u scripts/omp_lens_interp.py \
  --traits "$TRAITS" \
  --decomp-top-k "$TOP_K"

echo "=== DONE $(date -Is) ==="
for trait in good evil lawful chaotic; do
  if [ "$trait" = "good" ]; then
    d="persona_runs/dnd_good_scale/sae"
  else
    d="persona_runs/dnd_${trait}/sae"
  fi
  echo "  $d/ssv_omp_lens_interp.json"
  wc -c "$d/ssv_omp_lens_interp.json" 2>/dev/null || echo "    MISSING"
done
