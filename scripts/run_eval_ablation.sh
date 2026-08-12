#!/bin/bash
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH=$HOME/gemma-chat
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
SAE_REL=gemma-scope-2-4b-it-res-all
LOG=logs/eval_q_ablation_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== good ablation on eval_questions @ 1.5 ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.sae_experiment ablate \
  --run-id dnd_good_scale --layer 16 \
  --steer-alpha 1.5 \
  --attribution-json persona_runs/dnd_good_scale/sae/feature_attribution.json \
  --out-json persona_runs/dnd_good_scale/sae/ablation_eval_questions.json \
  --sae-release "$SAE_REL" --sae-id layer_16_width_262k_l0_small \
  --top-k 50 --use-eval-questions \
  --project applied-ai-practice00 2>&1 | tee -a "$LOG"

echo "=== evil ablation on eval_questions @ 1.5 ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.sae_experiment ablate \
  --run-id dnd_evil --layer 16 \
  --steer-alpha 1.5 \
  --attribution-json persona_runs/dnd_evil/sae/feature_attribution.json \
  --out-json persona_runs/dnd_evil/sae/ablation_eval_questions.json \
  --sae-release "$SAE_REL" --sae-id layer_16_width_262k_l0_small \
  --top-k 50 --use-eval-questions \
  --project applied-ai-practice00 2>&1 | tee -a "$LOG"

echo "DONE" | tee -a "$LOG"
