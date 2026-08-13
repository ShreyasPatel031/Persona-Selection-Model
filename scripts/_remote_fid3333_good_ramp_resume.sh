#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

OUT="persona_runs/dnd_good_scale/sae/fid3333_scale_to_incoherence_l16_extend.json"
mkdir -p "$(dirname "$OUT")"

# Seed prior alphas from crashed runs so we don't redo 7-12
python3 << 'PYEOF'
import json
from pathlib import Path
out = Path("persona_runs/dnd_good_scale/sae/fid3333_scale_to_incoherence_l16_extend.json")
prior = {
    7.0: [0]*20, 8.0: [0]*20, 9.0: [0]*20, 10.0: [0]*20, 12.0: [0]*20,
}
sweep = []
for a, scores in prior.items():
    sweep.append({
        "alpha_equiv": a, "scale": round(a * 1108.9, 2), "effective_scale": round(a * 1108.9, 2),
        "ratio_to_dense_alpha": round(a / 2.0, 3), "incoherent_count": 0, "n_questions": 20,
        "mean_tes": 0.0, "scores": scores, "replies": [],
    })
payload = {
    "trait": "good", "layer": 16, "fid": 3333, "cos_to_v": 0.6951,
    "method": "residual_add", "sign": 1.0, "alpha_dense": 2.0, "v_norm": 1108.9,
    "alpha_grid": [7,8,9,10,12,15,18,20],
    "scale_schedule": "scale = alpha_equiv * ||v||",
    "reference": {"baseline_mean": 0.2, "dense_mean": 64.5, "baseline_scores": [], "dense_scores": []},
    "sweep": sweep,
}
out.write_text(json.dumps(payload, indent=2))
print("Seeded", out, "with alphas", sorted(prior))
PYEOF

echo "=== GOOD fid 3333 RESUME alpha 15-20 started $(date -Is) ==="
.venv/bin/python3 -u scripts/fid_scale_to_incoherence.py \
  --trait good --layer 16 --fid 3333 --sign 1.0 \
  --alphas 15,18,20 \
  --n-questions 20 --judge-workers 4 \
  --skip-reference \
  --out "$OUT" \
  2>&1
echo "=== DONE $(date -Is) ==="
