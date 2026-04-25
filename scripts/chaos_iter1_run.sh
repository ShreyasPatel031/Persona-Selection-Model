#!/usr/bin/env bash
# chaos_iter1: rollouts → vectors → sanity (optional) → validate with α=1..10 and coherence cliff stop.
# Run from repo root on the GPU VM. Start Uvicorn for step-c (--gemma-url), stop if OOM before step-d.
set -euo pipefail
RUN_ID="${RUN_ID:-chaos_iter1}"
GEMMA_URL="${GEMMA_URL:-http://127.0.0.1:8080}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT"

echo "=== step-c (iter1: limit=8, rollouts_per_q=2) ==="
python -m app.persona.run step-c \
  --run-id "$RUN_ID" \
  --gemma-url "$GEMMA_URL" \
  --limit 8 \
  --rollouts-per-q 2

echo "=== step-d (persona_vectors.pt) ==="
python -m app.persona.run step-d --run-id "$RUN_ID"

echo "=== eval-answers (pos vs neg on eval_questions → Gate 1 input) ==="
python -m app.persona.run eval-answers --run-id "$RUN_ID" --gemma-url "$GEMMA_URL"

echo "=== sanity-eval-projection (Gate 1) ==="
python -m app.persona.run sanity-eval-projection --run-id "$RUN_ID"

echo "=== validate: α sweep 1..10, stop when mean coherence ≤ 15 ==="
python -m app.persona.run validate \
  --run-id "$RUN_ID" \
  --alphas "1,2,3,4,5,6,7,8,9,10" \
  --sweep-coherence-stop 15

echo "Done. See persona_runs/${RUN_ID}/eval/validation_report.json"
