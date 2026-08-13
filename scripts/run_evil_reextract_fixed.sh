#!/bin/bash
# Re-extract Evil vector after judge fix (bipolar rubric bug).
#
# The judge_rubric_to_instructions fix explicitly anchors scoring to the
# trait label ("Evil") and tells the judge to ignore the SYSTEM prompt
# persona.  This should fix:
#   1. step-C neg_kept: was 34/160 (judge scored good replies as evil)
#   2. calibration inversion: α=0 scored 84 evil (should be ~0)
#
# Run on gemma-mvp:
#   cd ~/gemma-chat && bash scripts/run_evil_reextract_fixed.sh
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH="$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"

ROLLOUTS_PER_Q="${ROLLOUTS_PER_Q:-10}"
MAX_PAIRS="${MAX_PAIRS:-1}"
RUN_ID=dnd_evil

LOG=logs/evil_reextract_fixed_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "=== Evil re-extract (judge fix) started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

TS=$(date +%Y%m%d_%H%M%S)
echo "--- backup ---"
for f in persona_runs/$RUN_ID/vectors/persona_vectors.pt \
         persona_runs/$RUN_ID/rollouts/extraction_rollouts.json \
         persona_runs/$RUN_ID/rollouts/rollouts.jsonl \
         persona_runs/$RUN_ID/rollouts/latest.json; do
  [ -f "$f" ] && cp -a "$f" "${f}.pre_judgefix_${TS}.bak" && echo "  backed up $f"
done

start_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 2
  set -a; . ./.hf.env; set +a
  nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 \
    >> /tmp/gemma-uvicorn.log 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8080/health 2>/dev/null | grep -q '"model_loaded":true'; then
      echo "uvicorn ready"
      return 0
    fi
    sleep 5
  done
  echo "uvicorn failed"
  return 1
}

stop_uvicorn() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 3
}

start_uvicorn || exit 1

echo "--- step-c scenarios (with fixed judge) ---"
.venv/bin/python3 -m app.persona.run step-c \
  --run-id "$RUN_ID" \
  --gemma-url http://127.0.0.1:8080 \
  --rollouts-per-q "$ROLLOUTS_PER_Q" \
  --questions-source scenarios \
  --max-pairs "$MAX_PAIRS" \
  --project "$GOOGLE_CLOUD_PROJECT"

stop_uvicorn

echo "--- step-d extract ---"
.venv/bin/python3 -m app.persona.run step-d --run-id "$RUN_ID"

echo "--- rollout stats ---"
.venv/bin/python3 -c "
import json
from pathlib import Path
ext = json.loads(Path('persona_runs/$RUN_ID/rollouts/extraction_rollouts.json').read_text())
s = ext.get('stats',{})
print('questions_source:', ext.get('questions_source'))
print('pos_kept:', s.get('pos_kept'), 'neg_kept:', s.get('neg_kept'))
print('pos_judged:', s.get('pos_judged'), 'neg_judged:', s.get('neg_judged'))
summ = json.loads(Path('persona_runs/$RUN_ID/vectors/summary.json').read_text())
print('kept_pos:', summ.get('kept_pos'), 'kept_neg:', summ.get('kept_neg'))
sh = summ.get('split_half_cosine',{})
if isinstance(sh, dict):
    pL = sh.get('mean_cosine_per_layer',[])
    for l in [12,15,16,22,31]:
        if l < len(pL): print('  split_half L%d: %.3f' % (l, pL[l]))
"

echo "--- layer sweep (evil only) ---"
.venv/bin/python3 -u scripts/all_traits_layer_sweep.py \
  --traits evil \
  --n-questions 5 \
  --out-json persona_runs/dnd_layer_sweep_evil_fixed.json \
  --config-out persona_runs/dnd_config_evil_fixed.json

echo "--- calibrate (20 eval Q) ---"
.venv/bin/python3 -m app.persona.vector_compose calibrate \
  --config-json persona_runs/dnd_config_evil_fixed.json \
  --n-questions 20 \
  --step 0.3 \
  --max-alpha 3.0 \
  --coherence-floor 80.0 \
  --out-json persona_runs/dnd_calibration_evil_fixed.json

echo ""
echo "=== results ==="
.venv/bin/python3 -c "
import json
from pathlib import Path
sw = json.loads(Path('persona_runs/dnd_layer_sweep_evil_fixed.json').read_text())
cal = json.loads(Path('persona_runs/dnd_calibration_evil_fixed.json').read_text())
et = sw['traits']['evil']
print('layer sweep: layer=%s mean_trait=%.1f' % (et['recommended_layer'], et.get('recommended_mean_trait',0)))
ec = cal['evil']
print('calibration: scale=%s layer=%s' % (ec.get('scale_recommended'), ec.get('layer')))
print()
for r in ec['alpha_sweep']['rows']:
    print('alpha=%.1f trait=%.1f coh=%.1f' % (r['alpha'], r['mean_trait'], r['mean_coherence']))
"

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: $LOG"
