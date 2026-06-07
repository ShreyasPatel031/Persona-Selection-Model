#!/usr/bin/env bash
# Corpus-based SSV feature interpretation + cluster causal validation on gemma-mvp.
set -euo pipefail
cd "$HOME/gemma-chat"
export GOOGLE_CLOUD_PROJECT=applied-ai-practice00
export PYTHONPATH="$HOME/gemma-chat"

# Step 1: Install deps (delphi optional; custom corpus script is primary)
.venv/bin/pip install -q zstandard datasets scipy 2>&1 | tail -3 || true
if ! .venv/bin/python3 -c "import delphi" 2>/dev/null; then
  .venv/bin/pip install -q eai-delphi 2>&1 | tail -3 || echo "(delphi install failed; using custom corpus script)"
fi

# Verify GemmaScope SAE loads
.venv/bin/python3 - <<'PY'
import torch
from app.phase2 import load_sae_for_layer
sae, info = load_sae_for_layer(torch.device("cpu"), release="gemma-scope-2-4b-it-res-all", sae_id="layer_16_width_262k_l0_small", hidden_state_index=17)
print("SAE OK:", info)
PY

# Step 2: Corpus activation cache + Gemini explanations
# Use 2M tokens for faster iteration; set N_TOKENS=10000000 for full plan
N_TOKENS="${N_TOKENS:-2000000}"

nohup .venv/bin/python3 -u scripts/ssv_corpus_interp.py \
  --trait good \
  --ssv persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json \
  --k-levels 100,512 \
  --n-tokens "$N_TOKENS" \
  --batch-size 1 \
  --out persona_runs/dnd_good_scale/sae/ssv_corpus_interp.json \
  > logs/ssv_corpus_interp.log 2>&1 &

CORPUS_PID=$!
echo "Corpus interp PID=$CORPUS_PID"

# Wait for corpus interp to finish
wait $CORPUS_PID || { tail -30 logs/ssv_corpus_interp.log; exit 1; }
echo "Corpus interp done."
tail -10 logs/ssv_corpus_interp.log

# Step 3: Cluster + causal validation
nohup .venv/bin/python3 -u scripts/ssv_cluster_causal.py \
  --trait good \
  --ssv persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json \
  --interp persona_runs/dnd_good_scale/sae/ssv_corpus_interp.json \
  --k 100 \
  --n-clusters 8 \
  --n-questions 5 \
  --out persona_runs/dnd_good_scale/sae/ssv_cluster_report.json \
  > logs/ssv_cluster_causal.log 2>&1 &

echo "Cluster causal PID=$!"
wait $!
echo "All done."
tail -20 logs/ssv_cluster_causal.log
