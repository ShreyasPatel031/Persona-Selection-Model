#!/usr/bin/env bash
# Pull Chen M.3.2 top-50 results from gemma-mvp and print formal proof summary.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$REPO/persona_runs/dnd_good_scale/sae/chen_m32_top50_20q_l16.json"
REMOTE="gemma-mvp:~/gemma-chat/persona_runs/dnd_good_scale/sae/chen_m32_top50_20q_l16.json"

mkdir -p "$(dirname "$LOCAL")"
gcloud compute scp "$REMOTE" "$LOCAL" \
  --project=applied-ai-practice00 --zone=us-central1-a --tunnel-through-iap

python3 << 'PYEOF'
import json
from pathlib import Path
p = Path("persona_runs/dnd_good_scale/sae/chen_m32_top50_20q_l16.json")
d = json.load(p.open())
features = d.get("features") or []
fp = d.get("formal_proof") or {}
print(f"Features done: {len(features)}/50")
print(f"Dense CAA: {d.get('reference', {}).get('dense_mean')}")
print(f"Max feature TES: {fp.get('max_feature_mean_tes')}")
print(f"T_pass: {fp.get('t_pass')}")
print(f"Conclusion: {fp.get('conclusion', 'in progress')}")
if features:
    top = sorted(features, key=lambda r: r.get('best_mean_tes') or 0, reverse=True)[:5]
    print("\nTop 5 so far:")
    for r in top:
        print(f"  rank={r['cos_rank']} fid={r['feature_id']} cos={r['cos_to_v']} mean={r['best_mean_tes']}")
PYEOF
