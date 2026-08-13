#!/usr/bin/env python3
"""
Phase 1 STA sweep: diagnostic + direct projection experiment.

1. Diagnostic: max cos(v_dense, W_dec[i]) at layers 16, 22, 29 for 16k and 262k SAEs
2. Exp 4:  direct projection of v_dense onto SAE dictionary (top-k by cosine)
           then generate/judge comparison vs dense steering
3. Exp 1B: encode latents with 262k SAE at layer 16, run validate-sta

Usage (on VM):
  cd ~/gemma-chat && PYTHONPATH=$HOME/gemma-chat GOOGLE_CLOUD_PROJECT=applied-ai-practice00 \\
    .venv/bin/python3 -u scripts/sta_phase1_sweep.py \\
    --run-id dnd_good_scale --steer-alpha 1.5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.persona.schemas import PersonaTraitArtifact

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

PERSONA_RUNS = Path(os.environ.get("PERSONA_RUNS", Path.home() / "gemma-chat/persona_runs"))

LAYER_SAE_CONFIGS = [
    # (layer, release, sae_id, label)
    (16, "gemma-scope-2-4b-it-res-all", "layer_16_width_16k_l0_small", "L16/16k"),
    (16, "gemma-scope-2-4b-it-res-all", "layer_16_width_262k_l0_small", "L16/262k"),
    (22, "gemma-scope-2-4b-it-res-all", "layer_22_width_16k_l0_small", "L22/16k"),
    (22, "gemma-scope-2-4b-it-res-all", "layer_22_width_262k_l0_small", "L22/262k"),
    (29, "gemma-scope-2-4b-it-res-all", "layer_29_width_16k_l0_small", "L29/16k"),
    (29, "gemma-scope-2-4b-it-res-all", "layer_29_width_262k_l0_small", "L29/262k"),
]


def load_dense_vectors(run_dir: Path) -> dict:
    vpt = run_dir / "vectors" / "persona_vectors.pt"
    return torch.load(vpt, map_location="cpu", weights_only=False)


def get_decoder_columns(sae) -> torch.Tensor:
    """Get W_dec (d_sae, d_in) directly from SAE parameters — no bias."""
    with torch.no_grad():
        return sae.W_dec.detach().float().cpu()


def run_diagnostic(run_dir: Path, device: torch.device) -> list[dict]:
    """Compute max cos(v_dense, W_dec[i]) for each layer/SAE config."""
    from app.phase2 import load_sae_for_layer

    ckpt = load_dense_vectors(run_dir)
    v_all = ckpt["v"].float()  # (n_layers, d_model)

    results = []
    for layer, release, sae_id, label in LAYER_SAE_CONFIGS:
        logger.info("=== Diagnostic: %s ===", label)
        t0 = time.time()

        if layer >= v_all.shape[0]:
            logger.warning("Layer %d > available vectors (%d), skipping", layer, v_all.shape[0])
            continue

        v_dense = v_all[layer]
        v_norm = F.normalize(v_dense.unsqueeze(0), dim=-1)[0]

        try:
            sae, sae_info = load_sae_for_layer(device, release=release, sae_id=sae_id)
        except Exception as e:
            logger.warning("Failed to load %s: %s", label, e)
            continue

        W_dec = get_decoder_columns(sae)  # (d_sae, d_in)
        W_norm = F.normalize(W_dec, dim=-1)

        cosines = W_norm @ v_norm  # (d_sae,)
        max_cos, max_idx = cosines.max(dim=0)
        min_cos, min_idx = cosines.min(dim=0)

        top_k_vals, top_k_ids = cosines.topk(20)
        top20_pos = [(int(top_k_ids[i]), float(top_k_vals[i])) for i in range(20)]

        best_sparse_50 = cosines.topk(50)
        best_sparse_200 = cosines.topk(min(200, cosines.shape[0]))

        # Sparse reconstruction quality: if we take the top-k columns weighted
        # by their projection, what cosine do we get with v_dense?
        for k_label, k in [("k=50", 50), ("k=200", 200), ("k=500", 500)]:
            k = min(k, cosines.shape[0])
            top_ids = cosines.topk(k).indices
            projections = (W_dec[top_ids] * v_dense.unsqueeze(0)).sum(dim=-1)  # dot products
            v_recon = (projections.unsqueeze(-1) * W_dec[top_ids]).sum(dim=0)
            recon_cos = float(F.cosine_similarity(v_recon.unsqueeze(0), v_dense.unsqueeze(0)).item())
            logger.info("  %s reconstruction cos = %.4f", k_label, recon_cos)

        elapsed = time.time() - t0
        row = {
            "label": label,
            "layer": layer,
            "sae_id": sae_id,
            "d_sae": int(sae.cfg.d_sae),
            "max_cosine": float(max_cos),
            "max_cosine_feature_id": int(max_idx),
            "min_cosine": float(min_cos),
            "min_cosine_feature_id": int(min_idx),
            "mean_abs_cosine": float(cosines.abs().mean()),
            "top20_positive": top20_pos,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        logger.info(
            "  max_cos=%.4f (feat %d)  min_cos=%.4f  mean_abs=%.4f  [%.1fs]",
            float(max_cos), int(max_idx), float(min_cos),
            float(cosines.abs().mean()), elapsed,
        )

        del sae, W_dec, W_norm, cosines
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return results


def run_direct_projection(
    run_dir: Path,
    device: torch.device,
    *,
    layer: int = 16,
    sae_release: str = "gemma-scope-2-4b-it-res-all",
    sae_id: str = "layer_16_width_262k_l0_small",
    steer_alpha: float = 1.5,
    top_k: int = 200,
    skip_judge: bool = False,
    project_id: str | None = None,
    limit: int = 0,
) -> dict:
    """
    Exp 4: Project v_dense directly onto SAE decoder columns.

    Instead of selecting atoms by activation frequency/amplitude (which finds
    INPUT features), directly find the k decoder columns most aligned with
    v_dense and reconstruct a steering vector from their projections.
    """
    from app.phase2 import load_sae_for_layer
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.quality_gates import score_coherence

    logger.info("=== Exp 4: Direct projection L%d/%s top_k=%d ===", layer, sae_id, top_k)

    ckpt = load_dense_vectors(run_dir)
    v_dense = ckpt["v"].float()[layer]
    v_norm = F.normalize(v_dense.unsqueeze(0), dim=-1)[0]

    bundle = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts" / "trait_bundle.json").read_text()
    )
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys_default = bundle.neg_system_prompt

    model, tokenizer, dev = load_model_and_tokenizer(None, device=device)
    sae, sae_info = load_sae_for_layer(dev, release=sae_release, sae_id=sae_id)
    dtype = next(model.parameters()).dtype

    W_dec = get_decoder_columns(sae)
    W_norm = F.normalize(W_dec, dim=-1)
    cosines = W_norm @ v_norm

    # Select top-k most aligned columns (positive cosine only)
    pos_mask = cosines > 0
    pos_cosines = cosines.clone()
    pos_cosines[~pos_mask] = -float("inf")
    top_vals, top_ids = pos_cosines.topk(min(top_k, int(pos_mask.sum())))

    # Build projection-based steering vector: v_proj = sum_i proj_i * W_dec[i]
    projections = (W_dec[top_ids] * v_dense.unsqueeze(0)).sum(dim=-1)
    v_proj = (projections.unsqueeze(-1) * W_dec[top_ids]).sum(dim=0)
    v_proj = v_proj.to(device=dev, dtype=dtype)

    dense_norm = float(v_dense.norm())
    proj_norm = float(v_proj.float().norm())
    cosine_proj = float(F.cosine_similarity(
        v_proj.float().cpu().unsqueeze(0), v_dense.unsqueeze(0)
    ).item())
    proj_alpha = steer_alpha * (dense_norm / max(proj_norm, 1e-8))

    logger.info(
        "Projection: %d atoms, cos=%.4f, dense_norm=%.1f, proj_norm=%.1f, proj_alpha=%.4f",
        len(top_ids), cosine_proj, dense_norm, proj_norm, proj_alpha,
    )

    top_atoms_info = [
        {"feature_id": int(top_ids[i]), "cosine": float(top_vals[i]),
         "projection": float(projections[i])}
        for i in range(min(30, len(top_ids)))
    ]

    # Free SAE memory before generation
    del sae, W_dec, W_norm, cosines
    torch.cuda.empty_cache() if dev.type == "cuda" else None

    # Load generation questions
    gen_path = run_dir / "sae" / "generations_l16_v2.json"
    if not gen_path.exists():
        gen_path = run_dir / "sae" / "generations.json"
    gen = json.loads(gen_path.read_text())
    questions = gen.get("questions", [])
    if limit and limit < len(questions):
        questions = questions[:limit]

    from app.persona.sae_experiment import _generate_steered_reply

    rows = []
    for i, qrow in enumerate(questions):
        q = qrow["question"]
        neg_sys = qrow.get("neg_system") or neg_sys_default
        logger.info("Projection validate %d/%d", i + 1, len(questions))

        baseline = _generate_steered_reply(model, tokenizer, dev, neg_sys, q, layer, v_dense.to(dev, dtype=dtype), 0.0)
        dense_reply = _generate_steered_reply(model, tokenizer, dev, neg_sys, q, layer, v_dense.to(dev, dtype=dtype), steer_alpha)
        proj_reply = _generate_steered_reply(model, tokenizer, dev, neg_sys, q, layer, v_proj, proj_alpha)

        row = {
            "question_index": i,
            "question": q,
            "dense_alpha": steer_alpha,
            "proj_alpha": proj_alpha,
            "baseline_reply": baseline,
            "dense_steered_reply": dense_reply,
            "proj_steered_reply": proj_reply,
        }

        if not skip_judge:
            pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            for key, reply in [("baseline", baseline), ("dense", dense_reply), ("proj", proj_reply)]:
                try:
                    js = score_transcript(judge_instr, neg_sys, q, reply, project_id=pid)
                    row[f"{key}_trait_score"] = int(js.score)
                except Exception as e:
                    row[f"{key}_trait_score"] = -1
                try:
                    row[f"{key}_coherence"] = score_coherence(reply, project_id=pid)
                except Exception as e:
                    row[f"{key}_coherence"] = -1
        rows.append(row)

    def _mean(key):
        vals = [r[key] for r in rows if r.get(key, -1) >= 0]
        return round(sum(vals) / len(vals), 2) if vals else None

    doc = {
        "method": "direct_projection",
        "layer": layer,
        "sae_id": sae_id,
        "top_k": top_k,
        "n_atoms_used": len(top_ids),
        "cosine_dense_proj": cosine_proj,
        "dense_norm": dense_norm,
        "proj_norm": proj_norm,
        "dense_alpha": steer_alpha,
        "proj_alpha": proj_alpha,
        "top_atoms": top_atoms_info,
        "mean_baseline_trait": _mean("baseline_trait_score"),
        "mean_dense_trait": _mean("dense_trait_score"),
        "mean_proj_trait": _mean("proj_trait_score"),
        "mean_baseline_coherence": _mean("baseline_coherence"),
        "mean_dense_coherence": _mean("dense_coherence"),
        "mean_proj_coherence": _mean("proj_coherence"),
        "comparisons": rows,
    }
    return doc


def main():
    parser = argparse.ArgumentParser(description="Phase 1 STA sweep")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steer-alpha", type=float, default=1.5)
    parser.add_argument("--only-diagnostic", action="store_true",
                        help="Run only the layer/SAE diagnostic")
    parser.add_argument("--only-projection", action="store_true",
                        help="Run only the direct projection experiment")
    parser.add_argument("--proj-layer", type=int, default=16)
    parser.add_argument("--proj-sae-id", default="layer_16_width_262k_l0_small")
    parser.add_argument("--proj-sae-release", default="gemma-scope-2-4b-it-res-all")
    parser.add_argument("--proj-top-k", type=int, default=200)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    run_dir = PERSONA_RUNS / args.run_id
    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Diagnostic
    if not args.only_projection:
        logger.info("========== DIAGNOSTIC: Layer/SAE cosine sweep ==========")
        diag = run_diagnostic(run_dir, device)
        diag_path = out_dir / "phase1_diagnostic.json"
        diag_path.write_text(json.dumps(diag, indent=2) + "\n")
        logger.info("Wrote %s", diag_path)

        logger.info("\n=== DIAGNOSTIC SUMMARY ===")
        for r in diag:
            logger.info(
                "  %-12s  max_cos=%.4f (feat %5d)  mean_abs=%.4f",
                r["label"], r["max_cosine"], r["max_cosine_feature_id"],
                r["mean_abs_cosine"],
            )

    # Step 2: Direct projection experiment
    if not args.only_diagnostic:
        logger.info("\n========== EXP 4: Direct Projection ==========")
        proj_doc = run_direct_projection(
            run_dir, device,
            layer=args.proj_layer,
            sae_release=args.proj_sae_release,
            sae_id=args.proj_sae_id,
            steer_alpha=args.steer_alpha,
            top_k=args.proj_top_k,
            skip_judge=args.skip_judge,
            limit=args.limit,
        )
        proj_path = out_dir / "exp4_direct_projection.json"
        proj_path.write_text(json.dumps(proj_doc, indent=2) + "\n")
        logger.info("Wrote %s", proj_path)

        logger.info("\n=== EXP 4 RESULTS ===")
        logger.info("  cos(v_dense, v_proj) = %.4f", proj_doc["cosine_dense_proj"])
        logger.info("  Baseline  trait=%.1f  coh=%.1f",
                     proj_doc["mean_baseline_trait"] or 0, proj_doc["mean_baseline_coherence"] or 0)
        logger.info("  Dense     trait=%.1f  coh=%.1f",
                     proj_doc["mean_dense_trait"] or 0, proj_doc["mean_dense_coherence"] or 0)
        logger.info("  Projection trait=%.1f  coh=%.1f",
                     proj_doc["mean_proj_trait"] or 0, proj_doc["mean_proj_coherence"] or 0)

    logger.info("\nPhase 1 sweep complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
