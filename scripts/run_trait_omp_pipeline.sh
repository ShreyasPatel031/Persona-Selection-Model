#!/usr/bin/env bash
# OMP decompose + steer sweep for D&D traits (evil, lawful, chaotic by default).
#
# Usage (on gemma-mvp):
#   cd ~/gemma-chat && bash scripts/run_trait_omp_pipeline.sh
#   TRAITS=evil bash scripts/run_trait_omp_pipeline.sh
#   SKIP_STEER=1 TRAITS=lawful,chaotic bash scripts/run_trait_omp_pipeline.sh
set -euo pipefail

cd "${HOME}/gemma-chat"
export PYTHONPATH="${HOME}/gemma-chat"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"

TRAITS="${TRAITS:-evil,lawful,chaotic}"
K_MAX="${K_MAX:-1000}"
SKIP_GEOMETRY="${SKIP_GEOMETRY:-0}"
SKIP_STEER="${SKIP_STEER:-0}"
N_QUESTIONS="${N_QUESTIONS:-5}"

LOG="logs/trait_omp_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs persona_runs
exec > >(tee -a "$LOG") 2>&1

PY="${HOME}/gemma-chat/.venv/bin/python3"

echo "=== trait OMP pipeline started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "TRAITS=$TRAITS K_MAX=$K_MAX SKIP_STEER=$SKIP_STEER"

IFS=',' read -ra TRAIT_ARR <<< "$TRAITS"
SUMMARY_ROWS=()

for trait in "${TRAIT_ARR[@]}"; do
  trait="$(echo "$trait" | xargs)"
  [ -z "$trait" ] && continue

  echo ""
  echo "========== TRAIT: $trait =========="

  echo "--- Phase 1: OMP decompose (k_max=$K_MAX) ---"
  "$PY" scripts/omp_decompose.py --trait "$trait" --k-max "$K_MAX"

  if [ "$SKIP_GEOMETRY" != "1" ]; then
    echo "--- Phase 1b: 16k vs 262k geometry ---"
    "$PY" scripts/omp_compare_16k_262k.py --trait "$trait"
  fi

  if [ "$SKIP_STEER" != "1" ]; then
    echo "--- Phase 2: OMP steer sweep ---"
    "$PY" scripts/omp_steer_test.py --trait "$trait" --n-questions "$N_QUESTIONS"
  fi

  echo "--- Write manifest ---"
  "$PY" - <<PY
import json
from pathlib import Path
from scripts.trait_sae_config import resolve_trait

cfg = resolve_trait("$trait")
decomp = json.loads(cfg["decomp"].read_text())
steer = None
if cfg["steer"].exists():
    steer = json.loads(cfg["steer"].read_text())
geom = None
if cfg["geometry"].exists():
    geom = json.loads(cfg["geometry"].read_text())

manifest = {
    "trait": cfg["trait"],
    "run_id": cfg["run_id"],
    "layer": cfg["layer"],
    "sae_id": cfg["sae_id"],
    "alpha": 1.5,
    "k_max": decomp.get("k_max"),
    "target_norm": decomp.get("target_norm"),
    "k_at_cos_99": decomp.get("k_at_cos_99"),
    "final_cosine": decomp.get("final_cosine"),
    "artifacts": {
        "decomp": str(cfg["decomp"]),
        "steer": str(cfg["steer"]) if cfg["steer"].exists() else None,
        "geometry": str(cfg["geometry"]) if cfg["geometry"].exists() else None,
    },
}
if steer:
    manifest["dense_caa_mean"] = next(
        (r["mean"] for r in steer.get("results", steer) if r.get("label") == "DENSE_CAA"),
        None,
    )
    manifest["omp_steer_summary"] = [
        {"label": r["label"], "mean": r.get("mean"), "cosine": r.get("cosine")}
        for r in steer.get("results", steer)
        if str(r.get("label", "")).startswith("OMP_K")
    ]
cfg["manifest"].parent.mkdir(parents=True, exist_ok=True)
cfg["manifest"].write_text(json.dumps(manifest, indent=2))
print("Wrote", cfg["manifest"])
PY
done

echo ""
echo "--- Cross-trait summary ---"
"$PY" - <<'PY'
import json
from pathlib import Path
from scripts.trait_sae_config import TRAIT_REGISTRY, resolve_trait

rows = []
for trait in TRAIT_REGISTRY:
    cfg = resolve_trait(trait)
    if not cfg["decomp"].exists():
        continue
    decomp = json.loads(cfg["decomp"].read_text())
    row = {
        "trait": trait,
        "layer": cfg["layer"],
        "target_norm": decomp.get("target_norm"),
        "k_at_cos_99": decomp.get("k_at_cos_99"),
        "final_cosine": decomp.get("final_cosine"),
    }
    cps = {c["k"]: c for c in decomp.get("checkpoints", [])}
    for k in (50, 100, 450, 750):
        if k in cps:
            row[f"cos@{k}"] = cps[k]["cosine"]
    if cfg["steer"].exists():
        steer = json.loads(cfg["steer"].read_text())
        for r in steer.get("results", steer):
            lab = r.get("label", "")
            if lab == "DENSE_CAA":
                row["dense_trait"] = r.get("mean")
            elif lab.startswith("OMP_K"):
                row[f"trait_{lab}"] = r.get("mean")
    rows.append(row)

out = Path("persona_runs/cross_trait_omp_summary.json")
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
print("Saved", out)
PY

echo ""
echo "=== DONE log=$LOG ==="
