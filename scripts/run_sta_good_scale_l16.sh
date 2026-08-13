#!/usr/bin/env bash
# STA pipeline for dnd_good_scale at L16 (generate -> encode -> validate-sta).
# Run on gemma-mvp with GPU attached.
# v2: correct alpha=1.5, generation grid includes 1.5, decoder projection filter.
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH="$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"

RUN_ID=dnd_good_scale
LAYER=16
SAE_RELEASE=gemma-scope-2-4b-it-res-all
SAE_ID=layer_16_width_16k_l0_small
STEER_ALPHA="${STEER_ALPHA:-1.5}"
AMP_THRESH="${AMP_THRESH:-0.3}"
FREQ_THRESH="${FREQ_THRESH:-0.4}"

LOG="logs/sta_good_scale_l16_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs persona_runs/"$RUN_ID"/sae

exec > >(tee -a "$LOG") 2>&1
echo "=== STA L16 v2 pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "run_id=$RUN_ID layer=$LAYER steer_alpha=$STEER_ALPHA sae=$SAE_ID"

PY="$HOME/gemma-chat/.venv/bin/python3"
GEN_JSON="persona_runs/$RUN_ID/sae/generations_l16_v2.json"
LATENTS_PT="persona_runs/$RUN_ID/sae/sae_latents_l16_v2.pt"
STA_JSON="persona_runs/$RUN_ID/sae/sta_validation_l16_v2.json"

echo "=== step 1: generate steered replies at L$LAYER (alphas include 1.5) ==="
"$PY" -m app.persona.sae_experiment generate \
  --run-id "$RUN_ID" \
  --layer "$LAYER" \
  --alphas "0,0.5,1.0,1.5,2.0" \
  --out "$GEN_JSON"

echo "=== step 2: SAE encode assistant spans ==="
"$PY" -m app.persona.sae_experiment encode \
  --run-id "$RUN_ID" \
  --layer "$LAYER" \
  --generations "$GEN_JSON" \
  --out-pt "$LATENTS_PT" \
  --sae-release "$SAE_RELEASE" \
  --sae-id "$SAE_ID"

echo "=== step 3: validate-sta (decoder projection + freq/amp filter) ==="
"$PY" -m app.persona.sae_experiment validate-sta \
  --run-id "$RUN_ID" \
  --layer "$LAYER" \
  --generations "$GEN_JSON" \
  --latents-pt "$LATENTS_PT" \
  --out-json "$STA_JSON" \
  --steer-alpha "$STEER_ALPHA" \
  --sae-release "$SAE_RELEASE" \
  --sae-id "$SAE_ID" \
  --amplitude-threshold "$AMP_THRESH" \
  --frequency-threshold "$FREQ_THRESH" \
  --top-k 50

echo "=== STA L16 v2 pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "outputs:"
echo "  $GEN_JSON"
echo "  $LATENTS_PT"
echo "  $STA_JSON"
