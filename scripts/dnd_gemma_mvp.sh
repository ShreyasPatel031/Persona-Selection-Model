#!/usr/bin/env bash
# D&D alignment first pass on gemma-mvp: sync code, run pipeline, pull artifacts.
# Requires: gcloud auth, IAP access, VM at ~/gemma-chat with .venv and Uvicorn for step-c.
set -euo pipefail

PROJECT="${GCP_PROJECT:-applied-ai-practice00}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GEMMA_MVP_INSTANCE:-gemma-mvp}"
# Remote repo root (literal for gcloud scp); use \$HOME in SSH heredocs for cd/pythonpath.
REMOTE_SCP="${INSTANCE}:~/gemma-chat"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

SSH_BASE=(gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap)

usage() {
  echo "Usage: $0 <command>"
  echo "  sync-code          SCP app/persona/ to VM"
  echo "  remote-env         Print one-liner to export GOOGLE_CLOUD_PROJECT + PYTHONPATH on VM"
  echo "  step-b-all         SSH: run step-b for dnd_lawful, dnd_chaotic, dnd_good, dnd_evil (Vertex)"
  echo "  step-c-all         SSH: step-c for each run-id (needs Uvicorn on 127.0.0.1:8080)"
  echo "  step-d-all         SSH: step-d for each run-id (GPU)"
  echo "  fetch-runs         SCP persona_runs/dnd_* from VM to local"
  echo "  push-config        SCP persona_runs/dnd_config.json to VM"
  echo "  calibrate          SSH: vector_compose calibrate (needs dnd_config.json on VM)"
  echo "  dnd-grid           SSH: vector_compose dnd-grid (needs calibration + config)"
  echo "  full-pipeline      sync-code + print next steps"
  echo "  start-uvicorn      SSH: nohup Uvicorn on 127.0.0.1:8080 (run before step-c if /health fails)"
}

remote_cd_py() {
  # shellcheck disable=SC2016
  echo 'cd "$HOME/gemma-chat" && set -a && [ -f .hf.env ] && . ./.hf.env; set +a && export GOOGLE_CLOUD_PROJECT='"$PROJECT"' && export PYTHONPATH="$HOME/gemma-chat"'
}

case "${1:-}" in
  sync-code)
    gcloud compute scp --recurse --tunnel-through-iap \
      "$repo_root/app/persona/" "${REMOTE_SCP}/app/persona/" \
      --zone="$ZONE" --project="$PROJECT"
    echo "Synced app/persona -> VM"
    ;;
  remote-env)
    echo "$(remote_cd_py)"
    ;;
  step-b-all)
    for pair in "dnd_lawful:Lawful" "dnd_chaotic:Chaotic" "dnd_good:Good" "dnd_evil:Evil"; do
      rid="${pair%%:*}"
      label="${pair##*:}"
      # shellcheck disable=SC2090,SC2091
      "${SSH_BASE[@]}" -- bash -s <<REMOTESCRIPT
set -euo pipefail
cd "\$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT=$PROJECT
export PYTHONPATH="\$HOME/gemma-chat"
export DND_LABEL=$label
export DND_RID=$rid
"\$HOME/gemma-chat/.venv/bin/python3" -u <<'PY'
import os, subprocess, sys
sys.path.insert(0, os.environ.get("PYTHONPATH", ".").split(os.pathsep)[0])
from app.persona.vector_compose import DND_TRAIT_DESCRIPTIONS
label = os.environ["DND_LABEL"]
rid = os.environ["DND_RID"]
desc = DND_TRAIT_DESCRIPTIONS[label]
subprocess.run(
    [
        sys.executable,
        "-m",
        "app.persona.run",
        "step-b",
        "--trait",
        label,
        "--trait-description",
        desc,
        "--run-id",
        rid,
    ],
    check=True,
)
PY
REMOTESCRIPT
    done
    ;;
  step-c-all)
    for rid in dnd_lawful dnd_chaotic dnd_good dnd_evil; do
      "${SSH_BASE[@]}" --command="$(remote_cd_py) && \"\$HOME/gemma-chat/.venv/bin/python3\" -m app.persona.run step-c \
        --run-id $rid --gemma-url http://127.0.0.1:8080"
    done
    ;;
  step-d-all)
    for rid in dnd_lawful dnd_chaotic dnd_good dnd_evil; do
      "${SSH_BASE[@]}" --command="$(remote_cd_py) && \"\$HOME/gemma-chat/.venv/bin/python3\" -m app.persona.run step-d --run-id $rid"
    done
    ;;
  fetch-runs)
    mkdir -p "$repo_root/persona_runs"
    for rid in dnd_lawful dnd_chaotic dnd_good dnd_evil; do
      gcloud compute scp --recurse --tunnel-through-iap \
        "${REMOTE_SCP}/persona_runs/$rid" \
        "$repo_root/persona_runs/" \
        --zone="$ZONE" --project="$PROJECT" || true
    done
    ;;
  push-config)
    gcloud compute scp --tunnel-through-iap \
      "$repo_root/persona_runs/dnd_config.json" \
      "${REMOTE_SCP}/persona_runs/dnd_config.json" \
      --zone="$ZONE" --project="$PROJECT"
    ;;
  calibrate)
    "${SSH_BASE[@]}" --command="$(remote_cd_py) && \"\$HOME/gemma-chat/.venv/bin/python3\" -m app.persona.vector_compose calibrate \
      --config-json persona_runs/dnd_config.json \
      --out-json persona_runs/dnd_calibration.json"
    mkdir -p "$repo_root/persona_runs"
    gcloud compute scp --tunnel-through-iap \
      "${REMOTE_SCP}/persona_runs/dnd_calibration.json" \
      "$repo_root/persona_runs/dnd_calibration.json" \
      --zone="$ZONE" --project="$PROJECT"
    ;;
  dnd-grid)
    q="${DND_GRID_QUESTION:-Your king orders you to raze a village harboring rebels. What do you do?}"
    q_json="$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$q")"
    "${SSH_BASE[@]}" --command="$(remote_cd_py) && \"\$HOME/gemma-chat/.venv/bin/python3\" -m app.persona.vector_compose dnd-grid \
      --config-json persona_runs/dnd_config.json \
      --calibration-json persona_runs/dnd_calibration.json \
      --question $q_json \
      --out-json persona_runs/dnd_grid_results.json"
    mkdir -p "$repo_root/persona_runs"
    gcloud compute scp --tunnel-through-iap \
      "${REMOTE_SCP}/persona_runs/dnd_grid_results.json" \
      "$repo_root/persona_runs/dnd_grid_results.json" \
      --zone="$ZONE" --project="$PROJECT"
    ;;
  start-uvicorn)
    "${SSH_BASE[@]}" --command='cd "$HOME/gemma-chat" && set -a && [ -f .hf.env ] && . ./.hf.env; set +a && \
      nohup "$HOME/gemma-chat/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8080 \
      >> /tmp/gemma-uvicorn.log 2>&1 & sleep 6 && curl -s -S "http://127.0.0.1:8080/health" | head -c 120'
    ;;
  full-pipeline)
    "$0" sync-code
    echo "On VM: run $0 start-uvicorn (or scripts/vm-restart.sh on the instance) until curl 127.0.0.1:8080/health works."
    echo "Then run: $0 step-b-all && $0 step-c-all && $0 step-d-all"
    echo "Copy persona_runs/dnd_config.example.json -> dnd_config.json; fix layers; $0 push-config"
    echo "Then: $0 calibrate && $0 dnd-grid && $0 fetch-runs"
    ;;
  ""|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
