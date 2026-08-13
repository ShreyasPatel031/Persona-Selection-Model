#!/usr/bin/env bash
# Run SAE-SSV full K sweep for one trait (foreground).
# Usage: _remote_sae_ssv_trait.sh <evil|lawful|chaotic|good>
set -euo pipefail

TRAIT="${1:?Usage: $0 <evil|lawful|chaotic|good>}"

case "$TRAIT" in
  evil)    RUN_ID=dnd_evil;     LAYER=16 ;;
  lawful)  RUN_ID=dnd_lawful;   LAYER=15 ;;
  chaotic) RUN_ID=dnd_chaotic;  LAYER=15 ;;
  good)    RUN_ID=dnd_good_scale; LAYER=16 ;;
  *) echo "Unknown trait: $TRAIT" >&2; exit 1 ;;
esac

cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

mkdir -p logs "persona_runs/${RUN_ID}/sae"

# Wait for GPU (other jobs may hold it); fall back to CPU if still busy.
wait_for_gpu() {
  local max_min="${1:-30}"
  for ((i=1; i<=max_min; i++)); do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      echo "GPU available"
      return 0
    fi
    if (( i == 1 )); then echo "GPU busy — waiting up to ${max_min}min..."; fi
    sleep 60
  done
  return 1
}

if ! wait_for_gpu 30; then
  export PERSONA_FORCE_CPU=1
  echo "GPU still busy — using CPU (PERSONA_FORCE_CPU=1)"
fi

OUT="persona_runs/${RUN_ID}/sae/sae_ssv_results_262k_l${LAYER}.json"
ZCACHE="persona_runs/${RUN_ID}/sae/probe_z_cache_l${LAYER}.npz"
LOG="logs/sae_ssv_${TRAIT}.log"

SKIP_COLLECT=()
if [[ -f "$ZCACHE" ]]; then
  SKIP_COLLECT=(--skip-collect)
  echo "Using existing z-cache: $ZCACHE"
fi

echo "=== SAE-SSV trait=$TRAIT run=$RUN_ID layer=$LAYER ==="
echo "out=$OUT log=$LOG"

pkill -f "sae_ssv_optimize.py.*--trait ${TRAIT}" 2>/dev/null || true
sleep 1

.venv/bin/python3 -u scripts/sae_ssv_optimize.py \
  --trait "$TRAIT" \
  --n-questions 5 \
  --ks 5,10,20,50,100,128,200,256,512,750,1000 \
  --n-iter 100 \
  --lr 0.05 \
  --lambda-lm 0.5 \
  --beta 0.01 \
  "${SKIP_COLLECT[@]}" \
  --out "$OUT" \
  2>&1 | tee "$LOG"

echo "=== Done trait=$TRAIT ==="
