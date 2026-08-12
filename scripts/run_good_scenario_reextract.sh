#!/bin/bash
# Re-extract dnd_good vector from scenario questions (eval + contrast_scenarios).
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH=$HOME/gemma-chat
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
LOG=logs/good_scenario_reextract_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

start_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 2
  set -a
  # shellcheck disable=SC1091
  . ./.hf.env
  set +a
  nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 \
    >> /tmp/gemma-uvicorn.log 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8080/health 2>/dev/null | grep -q '"model_loaded":true'; then
      echo "uvicorn ready (model loaded)" | tee -a "$LOG"
      return 0
    fi
    sleep 5
  done
  echo "uvicorn failed to start or model not loaded" | tee -a "$LOG"
  tail -20 /tmp/gemma-uvicorn.log | tee -a "$LOG"
  return 1
}

stop_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 3
}

echo "=== backup old vector + rollouts ===" | tee -a "$LOG"
cp -a persona_runs/dnd_good/vectors/persona_vectors.pt \
  persona_runs/dnd_good/vectors/persona_vectors_extraction_v1.pt 2>/dev/null || true
cp -a persona_runs/dnd_good/rollouts/rollouts.jsonl \
  persona_runs/dnd_good/rollouts/rollouts_extraction_v1.jsonl 2>/dev/null || true

start_uvicorn || exit 1

echo "=== step-c scenarios (eval + contrast_scenarios) ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.run step-c \
  --run-id dnd_good \
  --gemma-url http://127.0.0.1:8080 \
  --questions-source scenarios \
  --project applied-ai-practice00 2>&1 | tee -a "$LOG"

stop_uvicorn

echo "=== step-d re-extract vector ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.run step-d \
  --run-id dnd_good 2>&1 | tee -a "$LOG"

echo "=== coherence sweep (good only, 9 eval questions) ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.vector_compose calibrate \
  --config-json persona_runs/dnd_config.json \
  --traits-filter good \
  --n-questions 9 \
  --step 0.3 \
  --out-json persona_runs/dnd_good/vectors/calibration_scenarios_v1.json 2>&1 | tee -a "$LOG"

ALPHA=$(.venv/bin/python3 -c "
import json
d=json.load(open('persona_runs/dnd_good/vectors/calibration_scenarios_v1.json'))
print(d['good']['scale_recommended'])
")
echo "calibrated alpha=$ALPHA" | tee -a "$LOG"

echo "=== ablation on eval_questions (new vector) ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.sae_experiment ablate \
  --run-id dnd_good --layer 31 \
  --steer-alpha "$ALPHA" \
  --attribution-json persona_runs/dnd_good/sae/feature_attribution.json \
  --out-json persona_runs/dnd_good/sae/ablation_eval_questions_v2.json \
  --sae-release gemma-scope-2-4b-it-res-all \
  --sae-id layer_31_width_16k_l0_small \
  --top-k 50 --use-eval-questions \
  --project applied-ai-practice00 2>&1 | tee -a "$LOG"

echo DONE | tee -a "$LOG"
