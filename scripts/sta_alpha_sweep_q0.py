#!/usr/bin/env python3
"""Quick STA sparse alpha sweep on Q0 until coherence breaks."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.quality_gates import score_coherence
from app.persona.response_style import with_paragraph_cap
from app.persona.sae_common import build_sta_steering_vector
from app.persona.sae_experiment import _generate_steered_reply
from app.persona.schemas import PersonaTraitArtifact
from app.phase2 import load_sae_for_layer

PERSONA_RUNS = Path(os.environ.get("PERSONA_RUNS", Path.home() / "gemma-chat/persona_runs"))


def main() -> int:
    run_dir = PERSONA_RUNS / "dnd_good_scale"
    sta_doc = json.loads((run_dir / "sae/sta_validation_l16_exp1a.json").read_text())
    artifact = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts/trait_bundle.json").read_text()
    )
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    question = artifact.eval_questions[0]
    judge_instr = judge_rubric_to_instructions(
        artifact.judge_rubric, trait_label=artifact.trait_label
    )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")

    ckpt = torch.load(
        run_dir / "vectors/persona_vectors.pt", map_location="cpu", weights_only=False
    )
    u_dense = ckpt["v"].float()[16]
    layer = 16
    post_ids = set(sta_doc["sta_positive_atom_ids"])
    atoms = [
        a
        for a in sta_doc["sta_attribution"]["sta_positive_atoms"]
        if a["feature_id"] in post_ids
    ]

    model, tok, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        dev, release=sta_doc["sae_release"], sae_id=sta_doc["sae_id"]
    )
    dtype = next(model.parameters()).dtype
    u_sta = build_sta_steering_vector(sae, atoms, dev, dtype)
    sta_norm = float(u_sta.float().norm())
    dense_norm = float(u_dense.norm())
    matched = 1.5 * dense_norm / sta_norm

    alphas = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
        12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0,
    ]
    print(f"Q0 lifeboat | sta_norm={sta_norm:.1f} matched_alpha={matched:.2f}")
    print(f"{'alpha':>6} {'inj':>8} {'trait':>6} {'coh':>6}  preview")
    print("-" * 90)

    results: list[dict] = []
    for a in alphas:
        reply = _generate_steered_reply(
            model, tok, dev, neg_sys, question, layer, u_sta, a
        )
        inj = a * sta_norm
        try:
            trait = int(
                score_transcript(
                    judge_instr, neg_sys, question, reply, project_id=project
                ).score
            )
        except Exception as e:
            trait = -1
            print(f"  judge error @ {a}: {e}", file=sys.stderr)
        try:
            coh = int(score_coherence(reply, project_id=project))
        except Exception as e:
            coh = -1
            print(f"  coh error @ {a}: {e}", file=sys.stderr)

        preview = reply.replace("\n", " ")[:80]
        print(f"{a:6.1f} {inj:8.0f} {trait:6} {coh:6}  {preview}...")
        results.append(
            {"alpha": a, "inj": inj, "trait": trait, "coherence": coh, "reply": reply}
        )
        if coh >= 0 and coh < 60:
            print("STOP: coherence below 60")
            break
        if coh >= 0 and coh < 80 and a >= 8.0:
            print("STOP: coherence below 80 at high alpha")
            break

    out = run_dir / "sae/sta_alpha_sweep_q0.json"
    out.write_text(
        json.dumps(
            {
                "question": question,
                "matched_alpha": matched,
                "sta_norm": sta_norm,
                "dense_norm": dense_norm,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
