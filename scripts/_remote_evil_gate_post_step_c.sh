#!/usr/bin/env bash
# Run on GPU VM after evil_gate_v0 step-c + step-d: refresh eval, sanity projection, validate.
set -euo pipefail
cd ~/gemma-chat-probe
export HF_TOKEN=$(cat .hf_token_once)
export PYTHONPATH=.
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export GEMMA_MAX_NEW_TOKENS=128
export DISABLE_SAE=1
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 3
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-gpu-probe.log 2>&1 &
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:8080/health | grep -q '"model_loaded":true'; then
    echo "health_ok"
    break
  fi
  sleep 15
done
curl -s http://127.0.0.1:8080/health | grep -q '"model_loaded":true'

.venv/bin/python -m app.persona.run sanity-eval-projection \
  --run-id evil_gate_v0 \
  --refresh-eval \
  --gemma-url http://127.0.0.1:8080 \
  --force-cpu

.venv/bin/python -m app.persona.run validate \
  --run-id evil_gate_v0 \
  --n-candidate-layers 3 \
  --n-questions 2 \
  --alphas 0.5,1.0,1.5 \
  --project applied-ai-practice00 \
  --location us-central1 \
  --force-cpu

rm -f .hf_token_once
echo GATE_PIPELINE_DONE
