#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"

OUT="persona_runs/dnd_good_scale/sae/fid53982_scale_to_incoherence_l16.json"
mkdir -p "$(dirname "$OUT")"

python3 << 'PYEOF'
import json
from pathlib import Path
out = Path("persona_runs/dnd_good_scale/sae/fid53982_scale_to_incoherence_l16.json")
prior = {
    1.0: {"mean": 5.0, "scores": [0,0,5,90,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0]},
    2.0: {"mean": 14.0, "scores": [0,5,85,5,0,85,5,0,0,0,5,0,0,0,0,0,0,85,0,5]},
    3.0: {"mean": 23.8, "scores": [0,5,25,0,0,0,85,15,65,0,5,85,0,0,90,10,0,5,85,0]},
    4.0: {"mean": 28.5, "scores": [0,10,95,90,5,0,85,0,90,0,0,90,85,0,5,0,5,5,0,5]},
}
sweep = []
for a, row in prior.items():
    sweep.append({
        "alpha_equiv": a, "scale": round(a * 1108.9, 2), "effective_scale": round(a * 1108.9, 2),
        "ratio_to_dense_alpha": round(a / 2.0, 3), "incoherent_count": 0, "n_questions": 20,
        "mean_tes": row["mean"], "scores": row["scores"], "replies": [],
    })
payload = {
    "trait": "good", "layer": 16, "fid": 53982, "cos_to_v": 0.0187,
    "method": "residual_add", "sign": 1.0, "alpha_dense": 2.0, "v_norm": 1108.9,
    "alpha_grid": [1,2,3,4,5,6,7,8,9,10,12,15],
    "scale_schedule": "scale = alpha_equiv * ||v||",
    "reference": {"baseline_mean": 0.0, "dense_mean": 66.4, "baseline_scores": [], "dense_scores": []},
    "sweep": sweep,
}
out.write_text(json.dumps(payload, indent=2))
print("Seeded", out, "with alphas", sorted(prior))
PYEOF

echo "=== GOOD fid 53982 RESUME alpha 5+ started $(date -Is) ==="
.venv/bin/python3 -u scripts/fid_scale_to_incoherence.py \
  --trait good --layer 16 --fid 53982 --sign 1.0 \
  --alphas 5,6,7,8,9,10,12,15 \
  --n-questions 20 --judge-workers 4 \
  --skip-reference \
  --out "$OUT" \
  2>&1
echo "=== DONE $(date -Is) ==="
