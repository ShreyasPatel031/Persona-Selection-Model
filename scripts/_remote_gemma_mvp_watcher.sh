#!/usr/bin/env bash
# After evil d-sweep finishes on gemma-mvp, kill the all-traits loop and run good only.
set -euo pipefail

EVIL_PID="${1:?usage: $0 <evil_python_pid> <bash_loop_pid>}"
BASH_PID="${2:?usage: $0 <evil_python_pid> <bash_loop_pid>}"

echo "$(date -Is) watcher: waiting for evil pid=$EVIL_PID"
while kill -0 "$EVIL_PID" 2>/dev/null; do
  sleep 10
done

echo "$(date -Is) evil done — stopping sequential dsweep loop pid=$BASH_PID"
kill -9 "$BASH_PID" 2>/dev/null || true
sleep 3

cd "$HOME/gemma-chat"
bash scripts/_remote_dsweep_single_trait.sh good 16

echo "$(date -Is) gemma-mvp: good d-sweep finished"
