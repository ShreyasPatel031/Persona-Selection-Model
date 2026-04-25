#!/usr/bin/env bash
set -euo pipefail
cd ~/gemma-chat-probe
export PYTHONPATH=.
export HF_TOKEN=$(cat .hf_token_once)
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00

echo "=== stop uvicorn (free VRAM for teacher model) ==="
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 8

echo "=== sanity-eval-projection (eval_answers must exist) ==="
.venv/bin/python -m app.persona.run sanity-eval-projection --run-id evil_iter1

echo "=== validate ==="
.venv/bin/python -m app.persona.run validate --run-id evil_iter1 \
  --n-candidate-layers 3 \
  --n-questions 2 \
  --alphas 0.5,1.0,1.5,2.0 \
  --project applied-ai-practice00 \
  --location us-central1

echo "=== restart uvicorn ==="
export DISABLE_SAE=1
export GEMMA_MAX_NEW_TOKENS=128
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
echo "sanity_validate_done"
