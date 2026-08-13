#!/usr/bin/env bash
# E1–E5: EMD hyperparameter experiments @ K=5, L15, good trait (gemma-dsweep-good)
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

LOG="logs/emd_hyperparam_experiments.log"
exec > >(tee -a "$LOG") 2>&1

PY=".venv/bin/python3 -u"
COMMON="--trait good --steer-mode emd --ks 5 --n-questions 20 --judge-workers 10 --gen-batch-size 4"
SCALES="1,5,10,25,50,100,250,500,1000"
SAE="persona_runs/dnd_good_scale/sae"
ARAD_FF="$SAE/sae_output_score_l15.json"

echo "=== EMD hyperparam experiments start $(date -Is) ==="

# E1: SSV EMD scale sweep
echo ""
echo "=== E1: SSV scale sweep $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method ssv \
  --scales "$SCALES" \
  --experiment e1_ssv_scale_sweep \
  --out "$SAE/emd_e1_ssv_scale_sweep_k5.json"

# E2: Arad output-score scale sweep
echo ""
echo "=== E2: Arad scale sweep $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method ssv \
  --feature-file "$ARAD_FF" \
  --scales "$SCALES" \
  --experiment e2_arad_scale_sweep \
  --out "$SAE/emd_e2_arad_scale_sweep_k5.json"

# E3: Norm-matched EMD (SSV, Arad, OMP)
echo ""
echo "=== E3a: norm-match SSV $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method ssv \
  --norm-match-emd \
  --experiment e3a_normmatch_ssv \
  --out "$SAE/emd_e3a_normmatch_ssv_k5.json"

echo ""
echo "=== E3b: norm-match Arad $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method ssv \
  --feature-file "$ARAD_FF" \
  --norm-match-emd \
  --experiment e3b_normmatch_arad \
  --out "$SAE/emd_e3b_normmatch_arad_k5.json"

echo ""
echo "=== E3c: norm-match OMP $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method omp \
  --norm-match-emd \
  --experiment e3c_normmatch_omp \
  --out "$SAE/emd_e3c_normmatch_omp_k5.json"

# E4: Arad fids + OMP weights @ scale=3
echo ""
echo "=== E4: Arad fids + OMP weights scale=3 $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method ssv \
  --feature-file "$ARAD_FF" \
  --weight-mode omp-for-ssv-fids \
  --scale 3 \
  --experiment e4_arad_fids_omp_weights \
  --out "$SAE/emd_e4_arad_fids_omp_weights_k5.json"

# E5: SSV re-optimize on OMP mask, EMD @ scale=1
echo ""
echo "=== E5 step 1: SSV optimize on OMP mask K=5 $(date -Is) ==="
$PY scripts/ssv_optimize_omp_mask.py \
  --trait good --ks 5 \
  --out "$SAE/sae_ssv_omp_mask_k5_l15.json"

echo ""
echo "=== E5 step 2: EMD steer optimized weights scale=1 $(date -Is) ==="
$PY scripts/ssv_omp_k_sweep.py $COMMON --method ssv \
  --feature-file "$SAE/sae_ssv_omp_mask_k5_l15.json" \
  --scale 1 \
  --experiment e5_ssv_omp_mask_scale1 \
  --out "$SAE/emd_e5_ssv_omp_mask_k5.json"

echo ""
echo "=== EMD hyperparam experiments DONE $(date -Is) ==="
