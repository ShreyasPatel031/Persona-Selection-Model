#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"

EVIL_PID="${1:-52120}"
echo "=== Queue: waiting for evil extend PID $EVIL_PID ==="
while kill -0 "$EVIL_PID" 2>/dev/null; do sleep 30; done
echo "=== Evil extend finished $(date -Is) ==="

bash scripts/_remote_fid53982_good_ramp.sh 2>&1 | tee logs/fid53982_good_ramp.log
bash scripts/_remote_fid3333_good_ramp_extend.sh 2>&1 | tee logs/fid3333_good_ramp_extend.log
echo "=== All queued good runs DONE $(date -Is) ==="
