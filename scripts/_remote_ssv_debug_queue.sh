#!/usr/bin/env bash
# Sequential SSV debug queue on gemma-dsweep-good (Plans B, C, E)
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

LOG="logs/ssv_debug_queue.log"
exec > >(tee -a "$LOG") 2>&1

PY=".venv/bin/python3 -u"
KS_FULL="5,10,20,50,100,128,200,256,512,750,1000"
KS_QUICK="5,10,20,50,100"
COMMON="--trait good --method ssv --steer-mode emd --n-questions 20 --judge-workers 10 --gen-batch-size 4"

echo "=== SSV debug queue start $(date -Is) ==="

# Wait for Plan A (SSV EMD scale=3 full sweep) to finish
echo "Waiting for Plan A (ssv_k_sweep_l15_20q_emd.json) to finish..."
while pgrep -f "ssv_k_sweep_l15_20q_emd.json" >/dev/null 2>&1; do
  grep PROGRESS logs/ssv_k_sweep_l15_20q_emd.log 2>/dev/null | tail -1 || true
  sleep 120
done
echo "Plan A done $(date -Is)"

# --- Plan B: scale sweep (SSV features, EMD, vary scale) ---
for SCALE in 1 2 5 10; do
  OUT="persona_runs/dnd_good_scale/sae/ssv_k_sweep_l15_20q_emd_scale${SCALE}.json"
  echo ""
  echo "=== Plan B scale=${SCALE} $(date -Is) ==="
  $PY scripts/ssv_omp_k_sweep.py $COMMON \
    --scale "$SCALE" --ks "$KS_QUICK" \
    --experiment "plan_b_scale${SCALE}" \
    --out "$OUT"
done

# --- Plan C: SSV fids + OMP weights ---
echo ""
echo "=== Plan C: SSV fids + OMP weights $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON \
  --scale 3 --ks "$KS_FULL" \
  --weight-mode omp-for-ssv-fids \
  --experiment plan_c_ssvfids_ompw \
  --out persona_runs/dnd_good_scale/sae/ssv_k_sweep_l15_20q_emd_plan_c.json

# --- Plan E: SSV optimize on OMP feature mask, then steer ---
echo ""
echo "=== Plan E step 1: optimize on OMP mask $(date -Is) ==="
$PY scripts/ssv_optimize_omp_mask.py \
  --trait good --ks "$KS_FULL" \
  --out persona_runs/dnd_good_scale/sae/sae_ssv_omp_mask_l15.json

echo ""
echo "=== Plan E step 2: EMD steer with optimized weights $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON \
  --scale 3 --ks "$KS_FULL" \
  --feature-file persona_runs/dnd_good_scale/sae/sae_ssv_omp_mask_l15.json \
  --experiment plan_e_ompfeat_ssvw \
  --out persona_runs/dnd_good_scale/sae/ssv_k_sweep_l15_20q_emd_plan_e.json

echo ""
echo "=== SSV debug queue DONE $(date -Is) ==="
