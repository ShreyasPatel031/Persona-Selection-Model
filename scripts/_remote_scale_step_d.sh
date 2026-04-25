#!/usr/bin/env bash
set -euo pipefail
cd ~/gemma-chat-probe
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 8
export HF_TOKEN=$(cat .hf_token_once)
export PYTHONPATH=.
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
.venv/bin/python -m app.persona.run step-d --run-id evil_scale_v0
echo STEP_D_DONE
