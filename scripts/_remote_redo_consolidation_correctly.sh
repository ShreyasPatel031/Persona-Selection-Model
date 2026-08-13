#!/usr/bin/env bash
# Redo consolidation correctly: layer sweep -> update registry -> z-cache -> d-sweep -> viz
set -euo pipefail

cd "$HOME/gemma-chat"
set -a
[ -f .hf.env ] && . ./.hf.env
set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY="$HOME/gemma-chat/.venv/bin/python3"
LOG_DIR="$HOME/gemma-chat/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/redo_consolidation_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== redo consolidation started $(date -Is) ==="

# Free GPU from uvicorn if running
pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
sleep 3

# --- Phase 1: causal layer sweep (Appendix B.4) ---
echo ""
echo "=== Phase 1: causal layer sweep ==="
PYTHONPATH=. "$PY" -u scripts/all_traits_layer_sweep.py \
  --out-json persona_runs/dnd_layer_sweep.json \
  --config-out persona_runs/dnd_config.json \
  --n-questions 10 \
  --alpha 1.5 \
  --traits lawful,chaotic,good,evil

# --- Phase 2: update TRAIT_REGISTRY from sweep ---
echo ""
echo "=== Phase 2: update TRAIT_REGISTRY ==="
PYTHONPATH=. "$PY" -u << 'PYEOF'
import json
import re
from pathlib import Path

sweep = json.loads(Path("persona_runs/dnd_layer_sweep.json").read_text())
cfg_path = Path("scripts/trait_sae_config.py")
text = cfg_path.read_text()

for trait, block in sweep["traits"].items():
    layer = int(block["recommended_layer"])
    mean = block.get("recommended_mean_trait")
    print(f"  {trait}: L{layer} (mean_trait={mean})")

    # Update layer in TRAIT_REGISTRY for this trait
    pattern = rf'("{trait}":\s*\{{\s*"run_id":\s*"[^"]+",\s*"layer":\s*)\d+'
    repl = rf'\g<1>{layer}'
    new_text, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        raise RuntimeError(f"Failed to update layer for {trait} in trait_sae_config.py")
    text = new_text

# Update comment
text = re.sub(
    r"# Layers from causal sweep.*",
    f"# Layers from causal sweep (dnd_layer_sweep.json, {sweep.get('n_questions', '?')} eval Qs).",
    text,
    count=1,
)
cfg_path.write_text(text)
print("Updated scripts/trait_sae_config.py")
PYEOF

# --- Phase 3: rebuild z-caches at correct layers ---
echo ""
echo "=== Phase 3: z-cache rebuild ==="
for trait in evil lawful chaotic good; do
  echo ""
  echo "--- z-cache trait=$trait ---"
  PYTHONPATH=. "$PY" -u scripts/sae_ssv_optimize.py \
    --trait "$trait" \
    --optimize-only \
    --skip-ref \
    --ks 50 \
    --n-iter 1 \
    --n-questions 5
done

# --- Phase 4: stage2 d-sweep at correct layers ---
DS="5,10,20,30,40,50,60,80,100,150,200,500"
echo ""
echo "=== Phase 4: stage2 d-sweep ==="
for trait in evil lawful chaotic good; do
  echo ""
  echo "--- stage2 trait=$trait ---"
  PYTHONPATH=. "$PY" -u scripts/ssv_stage2_test.py \
    --trait "$trait" \
    --n-questions 20 \
    --ds "$DS"
done

# --- Phase 5: rebuild classifier viz JSONs ---
echo ""
echo "=== Phase 5: rebuild classifier viz ==="
PYTHONPATH=. "$PY" -u scripts/rebuild_ssv_bubble_viz_classifier_data.py

echo ""
echo "=== redo consolidation finished $(date -Is) ==="
