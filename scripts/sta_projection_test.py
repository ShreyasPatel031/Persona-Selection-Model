#!/usr/bin/env python3
"""
Project v_STA onto v_dense and compare per-question trait scores.

Conditions per question:
  BASELINE, DENSE_CAA, STA (raw), STA_PROJECTED (v_STA projected onto v_dense axis)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

RUN_DIR = Path("persona_runs/dnd_good_scale")
LAYER = 16
STEER_ALPHA = 1.5
STA_JSON = RUN_DIR / "sae/sta_validation_l16_v2.json"
LATENTS_PT = RUN_DIR / "sae/sae_latents_l16_v2.pt"


def rebuild_sta_vector(dev, dtype):
    from app.phase2 import load_sae_for_layer
    from app.persona.sae_common import (
        build_sta_steering_vector,
        compute_sta_attribution,
        filter_atoms_by_decoder_alignment,
    )

    ckpt_latents = torch.load(LATENTS_PT, map_location="cpu", weights_only=False)
    questions_latents = ckpt_latents.get("questions") or []
    sta_doc = json.loads(STA_JSON.read_text(encoding="utf-8"))

    sta_result = compute_sta_attribution(
        questions_latents,
        steered_alpha_key="1.5",
        amplitude_threshold=sta_doc.get("amplitude_threshold", 0.3),
        frequency_threshold=sta_doc.get("frequency_threshold", 0.4),
        top_k=sta_doc.get("top_k", 50),
    )
    u_dense = torch.load(
        RUN_DIR / "vectors/persona_vectors.pt", map_location="cpu", weights_only=False
    )["v"].float()[LAYER]

    sae, _ = load_sae_for_layer(
        dev,
        release=sta_doc.get("sae_release", "gemma-scope-2-4b-it-res-all"),
        sae_id=sta_doc.get("sae_id", "layer_16_width_16k_l0_small"),
    )
    pos_atoms, _ = filter_atoms_by_decoder_alignment(
        sae, sta_result["sta_positive_atoms"], u_dense, min_cosine=0.0
    )
    u_sta = build_sta_steering_vector(sae, pos_atoms, dev, dtype)
    return u_dense.to(dev), u_sta, pos_atoms


def project_onto_dense(u_sta: torch.Tensor, u_dense: torch.Tensor) -> torch.Tensor:
    u = u_dense.float() / (u_dense.float().norm() + 1e-8)
    coef = torch.dot(u_sta.float(), u)
    return (coef * u).to(dtype=u_sta.dtype, device=u_sta.device)


def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.sae_experiment import _generate_steered_reply
    from app.persona.schemas import PersonaTraitArtifact

    artifact = PersonaTraitArtifact.model_validate_json(
        (RUN_DIR / "artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = artifact.neg_system_prompt
    eval_qs = artifact.eval_questions[:5]
    judge_instr = judge_rubric_to_instructions(
        artifact.judge_rubric, trait_label=artifact.trait_label
    )

    model, tok, dev = load_model_and_tokenizer()
    dtype = next(model.parameters()).dtype
    u_dense, u_sta, atoms = rebuild_sta_vector(dev, dtype)
    u_proj = project_onto_dense(u_sta, u_dense)

    dense_norm = float(u_dense.float().norm().item())
    sta_norm = float(u_sta.float().norm().item())
    proj_norm = float(u_proj.float().norm().item())
    cos_sta = float(
        torch.dot(
            u_dense.float().cpu() / (dense_norm + 1e-8),
            u_sta.float().cpu() / (sta_norm + 1e-8),
        ).item()
    )
    cos_proj = float(
        torch.dot(
            u_dense.float().cpu() / (dense_norm + 1e-8),
            u_proj.float().cpu() / (proj_norm + 1e-8),
        ).item()
    )

    sta_alpha = STEER_ALPHA * dense_norm / max(sta_norm, 1e-8)
    proj_alpha = STEER_ALPHA * dense_norm / max(proj_norm, 1e-8)

    print("\n=== VECTOR STATS ===")
    print(f"n_atoms={len(atoms)}  cos(v_STA,v_dense)={cos_sta:.4f}  cos(v_proj,v_dense)={cos_proj:.4f}")
    print(f"||v_dense||={dense_norm:.1f}  ||v_STA||={sta_norm:.1f}  ||v_proj||={proj_norm:.1f}")
    print(f"alpha dense={STEER_ALPHA}  sta={sta_alpha:.4f}  proj={proj_alpha:.4f}")
    print(f"inject dense={STEER_ALPHA * dense_norm:.1f}  sta={sta_alpha * sta_norm:.1f}  proj={proj_alpha * proj_norm:.1f}")

    conditions = [
        ("BASELINE", u_dense, 0.0),
        ("DENSE_CAA", u_dense, STEER_ALPHA),
        ("STA", u_sta, sta_alpha),
        ("STA_PROJECTED", u_proj, proj_alpha),
    ]

    rows = []
    print("\n=== PER-QUESTION SCORES ===")
    print(f"{'Q':>3}  {'BASE':>5}  {'DENSE':>5}  {'STA':>5}  {'PROJ':>5}")
    print("-" * 32)

    for qi, q in enumerate(eval_qs):
        q_row = {"q_idx": qi, "question": q[:120], "conditions": {}}
        scores_line = [f"Q{qi + 1}"]

        for label, direction, alpha in conditions:
            reply = _generate_steered_reply(
                model, tok, dev, neg_sys, q, LAYER, direction, alpha
            )
            sc = None
            try:
                sc = int(score_transcript(judge_instr, neg_sys, q, reply).score)
            except Exception as e:
                logger.warning("Judge failed %s Q%d: %s", label, qi + 1, e)
            q_row["conditions"][label] = {"score": sc, "reply": reply[:400], "alpha": alpha}
            scores_line.append(f"{sc if sc is not None else '?':>5}")
            print(f"[Q{qi + 1}] {label:15s} score={sc}")

        print("  ".join(scores_line))
        rows.append(q_row)

    print("\n=== MEANS ===")
    for label, _, _ in conditions:
        vals = [
            r["conditions"][label]["score"]
            for r in rows
            if r["conditions"][label]["score"] is not None
        ]
        if vals:
            print(f"  {label:15s}  mean={sum(vals) / len(vals):5.1f}  scores={vals}")

    out = {
        "layer": LAYER,
        "n_atoms": len(atoms),
        "cos_sta_dense": cos_sta,
        "cos_proj_dense": cos_proj,
        "sta_alpha": sta_alpha,
        "proj_alpha": proj_alpha,
        "questions": rows,
    }
    out_path = Path("logs/sta_projection_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
