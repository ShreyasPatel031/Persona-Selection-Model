#!/bin/bash
# Overnight Good SAE milestone loop (Phase A data steps + SAE, Phase B alpha sweep).
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH="$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PERSONA_FULL_SCALE=1

RUN_ID=dnd_good_scale
LOG=logs/good_sae_milestone_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs persona_runs/"$RUN_ID"/artifacts persona_runs/"$RUN_ID"/sae_checkpoints

exec > >(tee -a "$LOG") 2>&1
echo "=== Good SAE milestone loop started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Stop any prior scaling loop (match script names, not this launcher)
pkill -f good_scaling_loop.py 2>/dev/null || true
pkill -f good_sae_milestone_loop.py 2>/dev/null || true
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

BUNDLE=persona_runs/"$RUN_ID"/artifacts/trait_bundle.json
if [ ! -f "$BUNDLE" ]; then
  echo "=== step-b full-scale Good bundle ==="
  .venv/bin/python3 -m app.persona.run step-b \
    --trait Good \
    --trait-description "$(.venv/bin/python3 -c "
from app.persona.vector_compose import DND_TRAIT_DESCRIPTIONS
print(DND_TRAIT_DESCRIPTIONS['Good'])
")" \
    --run-id "$RUN_ID" \
    --project "$GOOGLE_CLOUD_PROJECT"
else
  echo "=== bundle exists, skipping step-b ==="
fi

echo "=== SAE milestone loop (Phase A + Phase B) ==="
.venv/bin/python3 scripts/good_sae_milestone_loop.py \
  --run-id "$RUN_ID" \
  --project "$GOOGLE_CLOUD_PROJECT"

pkill -f "uvicorn app.main:app" 2>/dev/null || true
echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "persona_runs/$RUN_ID/DONE"
echo "Results: persona_runs/$RUN_ID/sae_milestone_results.json"
echo "Alpha sweep: persona_runs/$RUN_ID/sae_alpha_sweep.json"
