"""Generate baseline + steered replies on eval_questions for manual review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.persona.activations import load_model_and_tokenizer
from app.persona.quality_gates import _generate_steered
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact

RUN_ID = "dnd_good_scale"
LAYER = 31
ALPHAS = [0.0, 1.5, 2.0]
MAX_NEW = 180
N_QUESTIONS = 20


def main() -> None:
    root = Path.home() / "gemma-chat"
    run_dir = root / "persona_runs" / RUN_ID
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    vectors_pt = run_dir / "vectors" / "persona_vectors.pt"
    out_path = run_dir / "eval" / "manual_review_scenario_vector.json"

    artifact = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    questions = artifact.eval_questions[:N_QUESTIONS]

    dev = torch.device("cuda")
    model, tokenizer, dev = load_model_and_tokenizer("google/gemma-3-4b-it", device=dev)
    ck = torch.load(vectors_pt, map_location=dev, weights_only=False)
    direction = ck["v"].float()[LAYER].to(device=dev, dtype=next(model.parameters()).dtype)

    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q[:70]}...", flush=True)
        row = {"index": i, "question": q, "replies": {}}
        for alpha in ALPHAS:
            if alpha == 0.0:
                text = _generate_steered(
                    model,
                    tokenizer,
                    dev,
                    neg_sys,
                    q,
                    layer_idx=LAYER,
                    direction=direction,
                    alpha=0.0,
                    max_new_tokens=MAX_NEW,
                )
            else:
                text = _generate_steered(
                    model,
                    tokenizer,
                    dev,
                    neg_sys,
                    q,
                    layer_idx=LAYER,
                    direction=direction,
                    alpha=alpha,
                    max_new_tokens=MAX_NEW,
                )
            row["replies"][str(alpha)] = text
        results.append(row)

    doc = {
        "run_id": RUN_ID,
        "layer": LAYER,
        "alphas": ALPHAS,
        "vector_path": str(vectors_pt),
        "questions_source": "eval_questions",
        "n_questions": len(results),
        "items": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
