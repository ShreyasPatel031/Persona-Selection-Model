#!/usr/bin/env bash
# Tail evil d-sweep log and write incremental checkpoint JSON every 30s.
set -euo pipefail

cd "$HOME/gemma-chat"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"
LOG="${1:-/tmp/dsweep_correct_alpha.log}"
OUT="${2:-persona_runs/dnd_evil/sae/ssv_stage2_test_l15.json}"
TRAIT="${3:-evil}"
ALPHA="${4:-4.0}"

echo "$(date -Is) evil checkpoint watcher: log=$LOG out=$OUT"
while pgrep -f "ssv_stage2_test.py --trait evil" >/dev/null 2>&1 || [ -f "$LOG" ]; do
  if [ -f "$LOG" ]; then
    "$PY" -u scripts/checkpoint_dsweep_from_log.py \
      --log "$LOG" \
      --out "$OUT" \
      --trait "$TRAIT" \
      --layer 15 \
      --alpha "$ALPHA" \
      2>&1 || true
  fi
  pgrep -f "ssv_stage2_test.py --trait evil" >/dev/null 2>&1 || break
  sleep 30
done
# final write after process exits
if [ -f "$LOG" ]; then
  "$PY" -u scripts/checkpoint_dsweep_from_log.py \
    --log "$LOG" --out "$OUT" --trait "$TRAIT" --layer 15 --alpha "$ALPHA" || true
fi
echo "$(date -Is) evil checkpoint watcher done"
