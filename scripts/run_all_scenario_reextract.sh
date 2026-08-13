#!/bin/bash
# Re-extract all D&D trait vectors from scenario questions (eval + contrast_scenarios).
# Uses fixed step-C → step-D wiring (rollouts/latest.json sync) and judge filtering.
#
# Run on gemma-mvp:
#   cd ~/gemma-chat && bash scripts/run_all_scenario_reextract.sh
#
# Optional env:
#   ROLLOUTS_PER_Q=10   paper default (use 2–3 for a quicker smoke test)
#   MAX_PAIRS=1         contrastive prompt pairs per question (0 = all 5)
#   TRAITS=lawful,chaotic,good,evil
#   SKIP_LAYER_SWEEP=1  skip post-pass layer sweep + calibrate
#   N_CALIB_QUESTIONS=20
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH="$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"

ROLLOUTS_PER_Q="${ROLLOUTS_PER_Q:-10}"
MAX_PAIRS="${MAX_PAIRS:-1}"
TRAITS="${TRAITS:-lawful,chaotic,good,evil}"
N_CALIB_QUESTIONS="${N_CALIB_QUESTIONS:-20}"
SKIP_LAYER_SWEEP="${SKIP_LAYER_SWEEP:-0}"

LOG=logs/all_scenario_reextract_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "=== All-trait scenario re-extract started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "ROLLOUTS_PER_Q=$ROLLOUTS_PER_Q MAX_PAIRS=$MAX_PAIRS TRAITS=$TRAITS"

declare -A RUN_IDS=(
  [lawful]=dnd_lawful
  [chaotic]=dnd_chaotic
  [good]=dnd_good
  [evil]=dnd_evil
)

start_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 2
  set -a
  # shellcheck disable=SC1091
  . ./.hf.env
  set +a
  nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 \
    >> /tmp/gemma-uvicorn.log 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8080/health 2>/dev/null | grep -q '"model_loaded":true'; then
      echo "uvicorn ready (model loaded)"
      return 0
    fi
    sleep 5
  done
  echo "uvicorn failed to start or model not loaded"
  tail -20 /tmp/gemma-uvicorn.log || true
  return 1
}

stop_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 3
}

backup_run() {
  local rid=$1
  local run_dir="persona_runs/$rid"
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$run_dir/vectors" "$run_dir/rollouts"
  if [ -f "$run_dir/vectors/persona_vectors.pt" ]; then
    cp -a "$run_dir/vectors/persona_vectors.pt" \
      "$run_dir/vectors/persona_vectors_pre_scenario_${ts}.pt"
    echo "  backed up vector -> persona_vectors_pre_scenario_${ts}.pt"
  fi
  for f in rollouts.jsonl extraction_rollouts.json latest.json; do
    if [ -f "$run_dir/rollouts/$f" ]; then
      cp -a "$run_dir/rollouts/$f" \
        "$run_dir/rollouts/${f}.pre_scenario_${ts}.bak"
    fi
  done
}

step_c_args=(--rollouts-per-q "$ROLLOUTS_PER_Q" --questions-source scenarios --project "$GOOGLE_CLOUD_PROJECT")
if [ "$MAX_PAIRS" != "0" ]; then
  step_c_args+=(--max-pairs "$MAX_PAIRS")
fi

IFS=',' read -r -a TRAIT_LIST <<< "$TRAITS"
for trait in "${TRAIT_LIST[@]}"; do
  trait=$(echo "$trait" | tr -d ' ')
  rid="${RUN_IDS[$trait]:-}"
  if [ -z "$rid" ]; then
    echo "Unknown trait: $trait (skip)"
    continue
  fi
  echo ""
  echo "========== $trait ($rid) =========="
  backup_run "$rid"

  start_uvicorn || exit 1

  echo "--- step-c scenarios ---"
  .venv/bin/python3 -m app.persona.run step-c \
    --run-id "$rid" \
    --gemma-url http://127.0.0.1:8080 \
    "${step_c_args[@]}"

  stop_uvicorn

  echo "--- step-d extract (sync from extraction_rollouts.json) ---"
  .venv/bin/python3 -m app.persona.run step-d --run-id "$rid"

  .venv/bin/python3 -c "
import json
from pathlib import Path
run = Path('persona_runs/$rid')
ext = json.loads((run/'rollouts/extraction_rollouts.json').read_text())
latest = json.loads((run/'rollouts/latest.json').read_text()) if (run/'rollouts/latest.json').is_file() else {}
summary = json.loads((run/'vectors/summary.json').read_text()) if (run/'vectors/summary.json').is_file() else {}
sh = summary.get('split_half_cosine') or {}
if isinstance(sh, dict):
    sh_l31 = sh.get('mean_cosine_per_layer', [None]*34)
    sh_at = sh_l31[31] if len(sh_l31) > 31 else None
else:
    sh_at = sh
print('  step-c: source=%s items=%s skip_judge=%s' % (
    ext.get('questions_source'), len(ext.get('items',[])), latest.get('skip_judge')))
print('  step-d: kept_pos=%s kept_neg=%s split_half_L31=%s' % (
    summary.get('kept_pos'), summary.get('kept_neg'), sh_at))
"
done

if [ "$SKIP_LAYER_SWEEP" = "1" ]; then
  echo ""
  echo "SKIP_LAYER_SWEEP=1 — skipping layer sweep and calibrate"
  echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

echo ""
echo "========== causal layer sweep (all traits) =========="
.venv/bin/python3 -u scripts/all_traits_layer_sweep.py \
  --traits "$TRAITS" \
  --n-questions 5 \
  --out-json persona_runs/dnd_layer_sweep_scenario.json \
  --config-out persona_runs/dnd_config.json

echo ""
echo "========== calibrate alpha (all traits, ${N_CALIB_QUESTIONS} eval Q each) =========="
.venv/bin/python3 -m app.persona.vector_compose calibrate \
  --config-json persona_runs/dnd_config.json \
  --n-questions "$N_CALIB_QUESTIONS" \
  --step 0.3 \
  --max-alpha 3.0 \
  --coherence-floor 80.0 \
  --out-json persona_runs/dnd_calibration_scenario_reextract.json

echo ""
echo "=== Summary ==="
.venv/bin/python3 -c "
import json
from pathlib import Path
sweep = json.loads(Path('persona_runs/dnd_layer_sweep_scenario.json').read_text())
cal = json.loads(Path('persona_runs/dnd_calibration_scenario_reextract.json').read_text())
print('Layer sweep (@ alpha=%s):' % sweep.get('alpha'))
for t, b in sweep.get('traits', {}).items():
    print('  %-8s layer=%-2s mean_trait=%.1f' % (t, b.get('recommended_layer'), b.get('recommended_mean_trait', 0)))
print()
print('Calibration (coherence floor %s):' % cal.get('coherence_floor', 80))
for t in ['lawful','chaotic','good','evil']:
    if t not in cal: continue
    b = cal[t]
    sr = b.get('scale_recommended')
    rows = b.get('alpha_sweep', {}).get('rows', [])
    best = None
    for r in rows:
        if r.get('mean_coherence', 0) >= 80:
            if best is None or r.get('mean_trait', 0) > best[1]:
                best = (r.get('alpha'), r.get('mean_trait'), r.get('mean_coherence'))
    print('  %-8s scale=%s best@coh80=%s layer=%s' % (t, sr, best, b.get('layer')))
"

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: $LOG"
