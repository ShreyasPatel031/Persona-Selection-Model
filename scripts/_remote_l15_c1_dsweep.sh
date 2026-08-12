#!/usr/bin/env bash
# L15 C1 source artifacts on gemma-dsweep-good (parallel to Chen L16 on gemma-mvp).
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

PY="$HOME/gemma-chat/.venv/bin/python3"
SAE_DIR="persona_runs/dnd_good_scale/sae"
RUN_ID="dnd_good_scale"
LAYER=15
SAE_ID="layer_15_width_262k_l0_small"
LOG="logs/l15_c1_dsweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs "$SAE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== L15 C1 sources on dsweep-good $(date -Is) ==="

echo "--- 1/3 GradSAE phase A ---"
"$PY" -u scripts/causal_feature_screen.py \
  --trait good --phase A --sae 262k \
  --n-questions-a 5

echo "--- 2/3 F-stat attribution (generate -> encode -> attribute) ---"
GEN_JSON="$SAE_DIR/generations_l15.json"
LATENTS_PT="$SAE_DIR/sae_latents_l15.pt"
ATTR_JSON="$SAE_DIR/feature_attribution_l15.json"

"$PY" -m app.persona.sae_experiment generate \
  --run-id "$RUN_ID" --layer "$LAYER" \
  --alphas "0,0.5,1.0,1.5,2.0,2.5" \
  --out "$GEN_JSON"

"$PY" -m app.persona.sae_experiment encode \
  --run-id "$RUN_ID" --layer "$LAYER" \
  --generations "$GEN_JSON" --out-pt "$LATENTS_PT" \
  --sae-id "$SAE_ID"

"$PY" -m app.persona.sae_experiment attribute \
  --run-id "$RUN_ID" \
  --latents-pt "$LATENTS_PT" \
  --steered-alpha 2.0 \
  --out-json "$ATTR_JSON" \
  --top-k 200

echo "--- 3/3 Ablation necessity ranking ---"
"$PY" -u scripts/ablation_necessity_sweep.py \
  --trait good --n-questions 5

echo "--- C1 join ---"
"$PY" -u scripts/feature_set_convergence.py --trait good --k 20

echo "=== DONE $(date -Is) log=$LOG ==="
