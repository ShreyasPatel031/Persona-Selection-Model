#!/usr/bin/env bash
set -euo pipefail
cd ~/gemma-chat-probe
export PYTHONPATH=.
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export HF_TOKEN=$(cat .hf_token_once)
PROJ="$GOOGLE_CLOUD_PROJECT"
LOC="${VERTEX_LOCATION:-us-central1}"

echo "=== step-c evil_iter1 ==="
.venv/bin/python -m app.persona.run step-c \
  --run-id evil_iter1 \
  --gemma-url http://127.0.0.1:8080 \
  --limit 8 \
  --rollouts-per-q 2 \
  --no-paragraph-cap \
  --project "$PROJ" \
  --location "$LOC"

echo "=== step-d evil_iter1 ==="
.venv/bin/python -m app.persona.run step-d --run-id evil_iter1

echo "=== sanity-eval-projection evil_iter1 ==="
.venv/bin/python -m app.persona.run sanity-eval-projection --run-id evil_iter1 \
  --project "$PROJ" \
  --location "$LOC"

echo "=== validate evil_iter1 ==="
.venv/bin/python -m app.persona.run validate --run-id evil_iter1 \
  --n-candidate-layers 3 \
  --n-questions 2 \
  --alphas 0.5,1.0,1.5,2.0 \
  --project "$PROJ" \
  --location "$LOC"

echo "=== done evil_iter1 ==="
