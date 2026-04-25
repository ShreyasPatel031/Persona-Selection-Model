#!/usr/bin/env bash
# Local port forward: http://127.0.0.1:8080 -> VM Uvicorn (127.0.0.1:8080).
# Requires IAP (this project's firewall allows SSH only from specific IPs + IAP).
set -euo pipefail
PROJECT="${GCP_PROJECT:-applied-ai-practice00}"
ZONE="${GCP_ZONE:-us-central1-a}"
exec gcloud compute ssh gemma-mvp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --tunnel-through-iap \
  -- -L 8080:127.0.0.1:8080 -N
