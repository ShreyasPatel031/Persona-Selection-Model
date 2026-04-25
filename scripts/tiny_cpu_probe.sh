#!/usr/bin/env bash
# Phase 0 tiny probe (CPU or any VM with Gemma on localhost:8080).
# Usage: from repo root, with Uvicorn already running and HF_TOKEN set on the server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
RUN_ID="${RUN_ID:-evil_probe_cpu}"
BUNDLE="${BUNDLE:-persona_runs/evil_paper_v0/artifacts/trait_bundle.json}"
GEMMA_URL="${GEMMA_URL:-http://127.0.0.1:8080}"
PROJECT="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
PY="${PYTHON:-python3}"
"$PY" -m app.persona.run step-c \
  --run-id "$RUN_ID" \
  --bundle "$BUNDLE" \
  --gemma-url "$GEMMA_URL" \
  --limit 2 \
  --rollouts-per-q 1 \
  --no-paragraph-cap \
  --project "$PROJECT" \
  --location "${VERTEX_LOCATION:-us-central1}"
"$PY" -m app.persona.run step-d --run-id "$RUN_ID"
"$PY" -m app.persona.run validate --run-id "$RUN_ID" --skip-model-gates
echo "Done. Record metrics in docs/GPU_HOUR_SCOREBOARD.md"
