#!/usr/bin/env bash
set -euo pipefail
cd ~/gemma-chat-probe
export HF_TOKEN=$(cat .hf_token_once)
export PYTHONPATH=.
export DISABLE_SAE=1
export GEMMA_MAX_NEW_TOKENS=128
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-tiered.log 2>&1 &
echo $! > /tmp/gemma-uvicorn-tiered.pid
echo started_uvicorn
