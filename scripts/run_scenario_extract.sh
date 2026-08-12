#!/bin/bash
set -eo pipefail
cd ~/gemma-chat
export PYTHONPATH=$HOME/gemma-chat
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
LOG=/tmp/scenario_extract.log

echo "=== Backing up old rollouts/vector ===" | tee "$LOG"
cp persona_runs/dnd_good_scale/rollouts/extraction_rollouts.json \
   persona_runs/dnd_good_scale/rollouts/extraction_rollouts_abstract_backup.json 2>/dev/null || true
cp persona_runs/dnd_good_scale/vectors/persona_vectors.pt \
   persona_runs/dnd_good_scale/vectors/persona_vectors_abstract_backup.pt 2>/dev/null || true

echo "=== Step C: scenario rollouts (eval + contrast_scenarios) ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.run step-c \
  --run-id dnd_good_scale \
  --gemma-url http://127.0.0.1:8080 \
  --questions-source scenarios \
  --skip-judge \
  --rollouts-per-q 1 \
  --sampling-temperature 0.7 \
  --max-pairs 1 2>&1 | tee -a "$LOG"

echo "=== Step D: extract vector from scenario rollouts ===" | tee -a "$LOG"
.venv/bin/python3 -m app.persona.run step-d \
  --run-id dnd_good_scale 2>&1 | tee -a "$LOG"

echo "=== Vector summary ===" | tee -a "$LOG"
.venv/bin/python3 -c "
import torch
from pathlib import Path
vec_path = Path.home() / 'gemma-chat/persona_runs/dnd_good_scale/vectors/persona_vectors.pt'
vectors = torch.load(vec_path, map_location='cpu', weights_only=True)
print(f'Vector keys: {list(vectors.keys())}')
for k,v in vectors.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, norm={v.norm():.4f}')
" 2>&1 | tee -a "$LOG"

echo "EXTRACTION_DONE" | tee -a "$LOG"
