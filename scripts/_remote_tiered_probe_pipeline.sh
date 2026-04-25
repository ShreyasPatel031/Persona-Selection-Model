#!/usr/bin/env bash
# Usage: _remote_tiered_probe_pipeline.sh <run_id> <limit> <rollouts_per_q>
# Runs step-c → step-d → eval-answers → sanity-eval-projection → validate on gemma-chat-probe.
# Stops Uvicorn around step-d / teacher loads to avoid T4 OOM (Gemma-4B + server).
set -euo pipefail
RUN_ID="${1:?run_id}"
LIMIT="${2:?limit}"
RPQ="${3:?rollouts_per_q}"
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

echo "=== uvicorn for step-c / eval-answers ==="
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 3
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
wait_health

echo "=== step-c $RUN_ID limit=$LIMIT rollouts_per_q=$RPQ ==="
.venv/bin/python -m app.persona.run step-c \
  --run-id "$RUN_ID" \
  --gemma-url http://127.0.0.1:8080 \
  --limit "$LIMIT" \
  --rollouts-per-q "$RPQ" \
  --no-paragraph-cap \
  --project "$PROJ" \
  --location "$LOC"

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

echo "=== restart uvicorn ==="
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
echo "pipeline_done_${RUN_ID}"
