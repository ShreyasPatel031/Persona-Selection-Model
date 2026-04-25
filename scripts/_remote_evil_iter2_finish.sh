#!/usr/bin/env bash
set -euo pipefail
cd ~/gemma-chat-probe
export PYTHONPATH=.
export HF_TOKEN=$(cat .hf_token_once)
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 8
.venv/bin/python -m app.persona.run sanity-eval-projection --run-id evil_iter2
.venv/bin/python -m app.persona.run validate --run-id evil_iter2 \
  --n-candidate-layers 3 --n-questions 2 --alphas 0.5,1.0,1.5,2.0 \
  --project applied-ai-practice00 --location us-central1
export DISABLE_SAE=1 GEMMA_MAX_NEW_TOKENS=128
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
echo finish_done
