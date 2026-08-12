#!/bin/bash
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH=$HOME/gemma-chat
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
LOG=logs/good_ablate_v2_$(date +%Y%m%d_%H%M%S).log

.venv/bin/python3 -m app.persona.sae_experiment ablate \
  --run-id dnd_good --layer 31 --steer-alpha 2.1 \
  --attribution-json persona_runs/dnd_good/sae/feature_attribution.json \
  --out-json persona_runs/dnd_good/sae/ablation_eval_questions_v2.json \
  --sae-release gemma-scope-2-4b-it-res-all \
  --sae-id layer_31_width_16k_l0_small \
  --top-k 50 --use-eval-questions \
  --project applied-ai-practice00 2>&1 | tee -a "$LOG"

echo DONE | tee -a "$LOG"
