#!/usr/bin/env bash
# Run ON gemma-mvp under ~/gemma-chat-probe. Tiered Uvicorn: on for HTTP steps, off for GPU-heavy steps.
set -euo pipefail
cd ~/gemma-chat-probe
export HF_TOKEN="$(cat .hf_token_once)"
export PYTHONPATH=.
export DISABLE_SAE=1
export GEMMA_MAX_NEW_TOKENS=128
export GEMMA_URL="${GEMMA_URL:-http://127.0.0.1:8080}"
# Vertex judges (step-c / validate)
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"

log() { echo "[$(date -Iseconds)] $*"; }

wait_health() {
  local i
  for i in $(seq 1 90); do
    if curl -sS -m 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q model_loaded; then
      log "Uvicorn health OK"
      return 0
    fi
    sleep 2
  done
  log "ERROR: Uvicorn did not become healthy"
  return 1
}

start_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 2
  nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
  wait_health
}

stop_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 3
}

log "=== step-c ==="
start_uvicorn
.venv/bin/python -m app.persona.run step-c \
  --run-id chaos_iter1 \
  --gemma-url "$GEMMA_URL" \
  --limit 8 \
  --rollouts-per-q 2

log "=== step-d (stop Uvicorn first) ==="
stop_uvicorn
.venv/bin/python -m app.persona.run step-d --run-id chaos_iter1

log "=== eval-answers ==="
start_uvicorn
.venv/bin/python -m app.persona.run eval-answers --run-id chaos_iter1 --gemma-url "$GEMMA_URL"

log "=== sanity-eval-projection ==="
stop_uvicorn
.venv/bin/python -m app.persona.run sanity-eval-projection --run-id chaos_iter1

log "=== validate (α 1–10, coherence stop 15) ==="
.venv/bin/python -m app.persona.run validate \
  --run-id chaos_iter1 \
  --alphas "1,2,3,4,5,6,7,8,9,10" \
  --sweep-coherence-stop 15

log "=== CHAOS_ITER1_DONE ==="
