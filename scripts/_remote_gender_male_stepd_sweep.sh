#!/usr/bin/env bash
set -eo pipefail
cd ~/gemma-chat
set -a; [ -f .hf.env ] && . .hf.env; set +a
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}"
export PYTHONPATH="$HOME/gemma-chat"
PY=.venv/bin/python3

echo "=== STEP D gender_male (max-per-arm 200) $(date -Is) ==="
$PY -m app.persona.run step-d --run-id gender_male --max-per-arm 200

echo "=== SPLIT-HALF ==="
$PY -c '
import json
from pathlib import Path
s = json.loads(Path("persona_runs/gender_male/vectors/summary.json").read_text())
sh = s.get("split_half_cosine", {})
print("kept_pos", s.get("kept_pos"), "kept_neg", s.get("kept_neg"))
print("split_half_cosine", sh.get("mean_cosine_at_argmax_norm"))
print("interpretation", sh.get("interpretation"))
print("argmax_norm_layer", sh.get("argmax_norm_layer"))
cpl = sh.get("mean_cosine_per_layer", [])
for i, v in enumerate(cpl):
    print(f"  L{i}: {v:.4f}")
'

echo "=== LAYER SWEEP gender_male $(date -Is) ==="
$PY -c '
import json, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"gemma-chat"))
from app.persona.activations import load_model_and_tokenizer
from app.persona.layer_heuristics import _mid_range_candidate_layers
from app.persona.steering_demo import _language_model_layers
from app.persona.quality_gates import _generate_steered
from app.persona.schemas import PersonaTraitArtifact
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.response_style import with_paragraph_cap

persona_runs = Path.home()/"gemma-chat/persona_runs"
bundle_path = persona_runs/"gender_male/artifacts/trait_bundle.json"
vectors_path = persona_runs/"gender_male/vectors/persona_vectors.pt"

artifact = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
ck = torch.load(vectors_path, map_location="cpu", weights_only=False)
v_all = ck["v"].float()

model, tok, dev = load_model_and_tokenizer(None, device=None)
n_layers = len(_language_model_layers(model))
layers = sorted({li for li in _mid_range_candidate_layers(n_layers, n_candidates=6) if 12 <= li <= 22})
dtype = next(model.parameters()).dtype

neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)
questions = list(artifact.eval_questions[:10])
jkw = {"project_id": os.environ.get("GOOGLE_CLOUD_PROJECT")}

results = {}
for li in layers:
    direction = v_all[li].to(device=dev, dtype=dtype).view(1,1,-1)
    scores = []
    for q in questions:
        reply = _generate_steered(model, tok, dev, neg_sys, q,
            layer_idx=li, direction=direction, alpha=1.5, max_new_tokens=150)
        try:
            js = score_transcript(judge_instr, neg_sys, q, reply, **jkw)
            scores.append(int(js.score))
        except Exception as e:
            print(f"  judge error L{li}: {e}")
            scores.append(0)
    mean = sum(scores)/len(scores) if scores else 0
    results[li] = round(mean, 1)
    print(f"  L{li}: mean_trait={mean:.1f}  scores={scores}")

best = max(results, key=lambda l: results[l])
print(f"BEST LAYER: L{best} mean_trait={results[best]}")
out = persona_runs/"gender_male/eval/layer_sweep.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"layers": results, "best_layer": best, "best_mean": results[best]}, indent=2) + "\n")
'

echo "=== VALIDATE gender_male $(date -Is) ==="
$PY -m app.persona.run validate \
  --run-id gender_male \
  --project applied-ai-practice00 \
  --n-questions 20 \
  --sweep-coherence-stop 20

echo "=== VALIDATE SUMMARY ==="
$PY -c '
import json
from pathlib import Path
r = json.loads(Path("persona_runs/gender_male/eval/validation_report.json").read_text())
print("overall_pass", r.get("overall_pass"))
for g in r.get("gates", []):
    d = g.get("details", {})
    extra = d.get("best_layer") or d.get("interpretation") or ""
    print(f"  {g[\"gate\"]}: {\"PASS\" if g[\"passed\"] else \"FAIL\"}  score={g.get(\"score\")}  {extra}")
se = r.get("steering_effectiveness") or {}
if se:
    print("steering: best_layer", se.get("best_layer"), "best_alpha", se.get("best_alpha"))
'

echo "=== DONE $(date -Is) ==="
