#!/usr/bin/env bash
# After step-c finished (SSH dropped): step-d → eval-answers → sanity → validate
set -euo pipefail
RUN_ID="${1:?run_id}"
cd ~/gemma-chat-probe
export PYTHONPATH=.
export HF_TOKEN=$(cat .hf_token_once)
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export DISABLE_SAE=1
export GEMMA_MAX_NEW_TOKENS=128
PROJ="$GOOGLE_CLOUD_PROJECT"
LOC="${VERTEX_LOCATION:-us-central1}"

wait_health() {
  for _i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8080/health | grep -q '"model_loaded":true'; then echo "health_ok"; return 0; fi
    sleep 10
  done
  echo "health timeout" >&2
  return 1
}

echo "=== stop uvicorn; step-d ==="
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 8
.venv/bin/python -m app.persona.run step-d --run-id "$RUN_ID"

echo "=== uvicorn for eval-answers ==="
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
wait_health

echo "=== eval-answers ==="
.venv/bin/python -m app.persona.run eval-answers \
  --run-id "$RUN_ID" \
  --gemma-url http://127.0.0.1:8080

echo "=== stop uvicorn; sanity-eval + validate ==="
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 8
.venv/bin/python -m app.persona.run sanity-eval-projection --run-id "$RUN_ID"
.venv/bin/python -m app.persona.run validate --run-id "$RUN_ID" \
  --n-candidate-layers 3 \
  --n-questions 2 \
  --alphas 0.5,1.0,1.5,2.0 \
  --project "$PROJ" \
  --location "$LOC"

nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
echo "resume_done_${RUN_ID}"
