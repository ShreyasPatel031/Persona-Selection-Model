#!/usr/bin/env bash
# Run ON THE VM (after .hf.env exists with HF_TOKEN=...).
set -euo pipefail
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2
cd ~/gemma-chat
set -a
# shellcheck disable=SC1091
. ./.hf.env
set +a
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 >> /tmp/gemma-uvicorn.log 2>&1 &
echo "Uvicorn started in background. tail -f /tmp/gemma-uvicorn.log"
