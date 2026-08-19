#!/usr/bin/env bash
# Provision a GCP GPU VM, run E0 injection-scope ablation, pull results, delete VM.
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-project-amer-scs-sandbox}"
ZONE="${GPU_PROBE_ZONE:-us-central1-a}"
INSTANCE="${E0_INSTANCE:-e0-scope-$(date +%s)}"
REMOTE_DIR="e0_scope_run"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VECTORS_SRC="${VECTORS_DIR:-/tmp/keep_vectors}"
GCLOUD="${GCLOUD:-/tmp/google-cloud-sdk/bin/gcloud}"
CONTROLS="${E0_CONTROLS:-3}"
MODEL_ID="${E0_MODEL_ID:-unsloth/gemma-3-4b-it}"

log() { echo "[e0-gcp] $*" >&2; }

cleanup() {
  if [[ "${KEEP_VM:-0}" != "1" && "${CREATED:-0}" == "1" ]]; then
    log "Deleting VM ${INSTANCE}…"
    "$GCLOUD" compute instances delete "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" --quiet || true
  fi
}
trap cleanup EXIT

create_vm() {
  log "Creating GPU VM ${INSTANCE} in ${ZONE}…"
  for prof in \
    "g2-standard-4 type=nvidia-l4,count=1 g2+l4" \
    "n1-standard-8 type=nvidia-tesla-t4,count=1 n1-8+t4"; do
    read -r machine acc label <<<"$prof"
    if "$GCLOUD" compute instances create "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" \
      --machine-type="$machine" \
      --accelerator="$acc" \
      --maintenance-policy=TERMINATE \
      --network=main-vpc \
      --subnet=primary-subnet \
      --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=200GB \
      --scopes=https://www.googleapis.com/auth/cloud-platform; then
      log "Created with profile ${label}"
      CREATED=1
      return 0
    fi
    log "Profile ${label} failed, trying next…"
  done
  return 1
}

wait_ssh() {
  for i in $(seq 1 48); do
    if "$GCLOUD" compute ssh "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap \
      --command "echo ssh_ok" >/dev/null 2>&1; then
      log "SSH ready (${i})"
      return 0
    fi
    sleep 10
  done
  return 1
}

gssh() {
  "$GCLOUD" compute ssh "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap \
    --command "$1"
}

gscp() {
  "$GCLOUD" compute scp --recurse --tunnel-through-iap \
    "$1" "${INSTANCE}:~/${REMOTE_DIR}/$2" \
    --project="$PROJECT" --zone="$ZONE"
}

pull_results() {
  mkdir -p "$REPO_ROOT/results/injection_scope_ablation"
  "$GCLOUD" compute scp --recurse --tunnel-through-iap \
    "${INSTANCE}:~/${REMOTE_DIR}/results/injection_scope_ablation/*" \
    "$REPO_ROOT/results/injection_scope_ablation/" \
    --project="$PROJECT" --zone="$ZONE" || true
}

main() {
  cd "$REPO_ROOT"
  [[ -d "$VECTORS_SRC" ]] || { log "Missing vectors at $VECTORS_SRC"; exit 1; }

  create_vm
  wait_ssh

  gssh "mkdir -p ~/${REMOTE_DIR}/{vectors,results/injection_scope_ablation,scripts,results/gemma_final,results/e1_inspan}"

  log "Syncing repo slice…"
  gscp "$REPO_ROOT/app" "app"
  gscp "$REPO_ROOT/scripts/ablate_injection_scope.py" "scripts/ablate_injection_scope.py"
  gscp "$REPO_ROOT/scripts/measure_injection_span.py" "scripts/measure_injection_span.py"
  gscp "$REPO_ROOT/requirements.txt" "requirements.txt"
  gscp "$REPO_ROOT/data" "data"
  gscp "$REPO_ROOT/results/gemma_final/validated_sweep_conscientiousness_pc1_high.json" "results/gemma_final/"
  gscp "$REPO_ROOT/results/gemma_final/validated_sweep_conscientiousness_pc1_low.json" "results/gemma_final/"
  gscp "$REPO_ROOT/results/e1_inspan/validated_sweep_extraversion_pc1_high.json" "results/e1_inspan/"
  gscp "$VECTORS_SRC/ladder_vectors_conscientiousness.pt" "vectors/"
  gscp "$VECTORS_SRC/ladder_vectors_extraversion.pt" "vectors/"

  log "Remote bootstrap + E0 run…"
  gssh "bash -s" <<REMOTE
set -euo pipefail
cd ~/${REMOTE_DIR}
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip git
python3 -m venv .venv
.venv/bin/pip install -U pip wheel setuptools -q
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu129 -q
grep -v '^[[:space:]]*torch' requirements.txt > /tmp/req_notorch.txt
.venv/bin/pip install -r /tmp/req_notorch.txt -q
export PYTHONPATH=.
export GEMMA_MODEL_ID=${MODEL_ID}
nvidia-smi || true
.venv/bin/python scripts/ablate_injection_scope.py \\
  --vectors-dir vectors \\
  --out-dir results/injection_scope_ablation \\
  --model-id ${MODEL_ID} \\
  --random-controls ${CONTROLS} \\
  --probes 0
echo E0_EXIT=\$?
REMOTE

  pull_results
  log "Done. Results in results/injection_scope_ablation/"
  if [[ -f "$REPO_ROOT/results/injection_scope_ablation/summary.json" ]]; then
    cat "$REPO_ROOT/results/injection_scope_ablation/summary.json"
  fi
}

main "$@"
