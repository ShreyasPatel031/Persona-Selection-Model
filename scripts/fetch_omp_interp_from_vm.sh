#!/usr/bin/env bash
# Pull OMP interp artifacts from gemma-mvp to local repo.
set -euo pipefail
PROJECT="${GCP_PROJECT:-applied-ai-practice00}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GEMMA_MVP_INSTANCE:-gemma-mvp}"
REMOTE="${INSTANCE}:~/gemma-chat"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

SCP=(gcloud compute scp --tunnel-through-iap --project="$PROJECT" --zone="$ZONE")

echo "Fetching L15 shared logit lens cache..."
"${SCP[@]}" "${REMOTE}/persona_runs/_shared/l15_262k_logit_lens_cache.json" \
  persona_runs/_shared/l15_262k_logit_lens_cache.json

for spec in \
  "dnd_good_scale:persona_runs/dnd_good_scale/sae" \
  "dnd_evil:persona_runs/dnd_evil/sae" \
  "dnd_lawful:persona_runs/dnd_lawful/sae" \
  "dnd_chaotic:persona_runs/dnd_chaotic/sae"; do
  run="${spec%%:*}"
  local_dir="${spec#*:}"
  mkdir -p "$local_dir"
  for f in ssv_omp_feature_logit_lens_262k_l15.json ssv_omp_corpus_interp.json; do
    echo "  $run/$f"
    "${SCP[@]}" "${REMOTE}/${local_dir}/${f}" "${local_dir}/${f}" 2>/dev/null || echo "    (skip missing $f)"
  done
done

echo "Done. Rebuild with: python3 scripts/rebuild_ssv_bubble_viz_omp_data.py"
