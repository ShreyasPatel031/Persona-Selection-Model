#!/usr/bin/env bash
# OMP feature logit lens + corpus Gemini interpretation on gemma-mvp.
# Step 1 (logit lens) runs on CPU. Step 2 (corpus interp) uses GPU.
set -euo pipefail
cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

PY="$HOME/gemma-chat/.venv/bin/python3"
N_TOKENS="${N_TOKENS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAITS="${TRAITS:-good,evil,lawful,chaotic}"

mkdir -p "$HOME/gemma-chat/logs"

pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

echo "=== Step 1: OMP logit lens (L15, CPU — already done, skip if cache exists) ==="
if [ -f persona_runs/_shared/l15_262k_logit_lens_cache.json ]; then
  echo "  cache exists, skipping lens step"
else
  PERSONA_FORCE_CPU=1 "$PY" -u scripts/ssv_feature_logit_lens.py \
    --omp-traits "$TRAITS" --layer 15 --device cpu \
    2>&1 | tee logs/omp_interp_lens.log
fi

echo "=== Step 2: OMP corpus interp per trait (GPU, 50K tokens, batch=$BATCH_SIZE) ==="
IFS=',' read -ra TRAIT_ARR <<< "$TRAITS"
for trait in "${TRAIT_ARR[@]}"; do
  trait="$(echo "$trait" | xargs)"
  [ -z "$trait" ] && continue
  sweep="persona_runs/dnd_${trait}_scale/sae/ssv_omp_dsweep_l15.json"
  if [ "$trait" != "good" ]; then
    sweep="persona_runs/dnd_${trait}/sae/ssv_omp_dsweep_l15.json"
  fi
  if [ ! -f "$sweep" ]; then
    echo "SKIP $trait: missing $sweep"
    continue
  fi
  out_dir="$(dirname "$sweep")"
  out="${out_dir}/ssv_omp_corpus_interp.json"
  log="logs/omp_corpus_interp_${trait}.log"
  echo "--- corpus interp $trait (GPU) ---"
  "$PY" -u scripts/ssv_corpus_interp.py \
    --trait "$trait" \
    --omp-sweep "$sweep" \
    --n-tokens "$N_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --out "$out" \
    --cache "${out_dir}/ssv_omp_corpus_cache.json" \
    --force \
    2>&1 | tee "$log"
  echo "Done $trait -> $out"
done

echo "=== All OMP interp done $(date -Is) ==="
for trait in good evil lawful chaotic; do
  if [ "$trait" = "good" ]; then
    f="persona_runs/dnd_good_scale/sae/ssv_omp_corpus_interp.json"
  else
    f="persona_runs/dnd_${trait}/sae/ssv_omp_corpus_interp.json"
  fi
  [ -f "$f" ] && echo "  $f $(wc -c < "$f") bytes" || echo "  MISSING $f"
done
