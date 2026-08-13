#!/usr/bin/env python3
"""Trait-directional Arad-style output scores for all SAE features.

For each feature f:
  score(f) = cosine(W_U @ W_dec[f], W_U @ v_dense_layer)

Ranks features by output score and writes a K-sweep feature file compatible with
ssv_omp_k_sweep.py --feature-file.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.phase2 import load_sae_for_layer
from scripts.ssv_feature_logit_lens import effective_lm_head
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, resolve_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("output_score")

DEFAULT_KS = (5, 10, 20, 50, 100, 128, 200, 256, 512, 750, 1000)
CHUNK = 256


def compute_output_scores(
    W_dec: torch.Tensor,
    W_U: torch.Tensor,
    v_layer: torch.Tensor,
    *,
    chunk: int = CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scores, logit_norms) for each feature."""
    d_sae = W_dec.shape[0]
    logit_dense = W_U @ v_layer.float()
    logit_dense_norm = logit_dense.norm()
    if logit_dense_norm < 1e-8:
        raise RuntimeError("Dense logit signature has zero norm")

    scores = torch.zeros(d_sae, dtype=torch.float32)
    logit_norms = torch.zeros(d_sae, dtype=torch.float32)

    for start in range(0, d_sae, chunk):
        end = min(start + chunk, d_sae)
        block = W_dec[start:end]  # (chunk, d_model)
        logit_block = W_U @ block.T  # (vocab, chunk)
        norms = logit_block.norm(dim=0).clamp_min(1e-8)
        dots = (logit_block * logit_dense.unsqueeze(1)).sum(dim=0)
        cos = dots / (norms * logit_dense_norm)
        scores[start:end] = cos.cpu()
        logit_norms[start:end] = norms.cpu()
        del logit_block, block
        if (start // chunk) % 200 == 0:
            logger.info("  scored %d / %d features", end, d_sae)

    return scores, logit_norms


def build_k_sweep_rows(
    ranked_fids: list[int],
    ranked_scores: list[float],
    ks: list[int],
) -> list[dict]:
    """Build results rows with feature_ids and feature_weights (output scores)."""
    rows = []
    for k in sorted(ks):
        k = min(k, len(ranked_fids))
        fids = ranked_fids[:k]
        weights = ranked_scores[:k]
        rows.append({
            "k": k,
            "method": "output_score_arad",
            "n_active_features": len(fids),
            "feature_ids": fids,
            "feature_weights": [round(w, 6) for w in weights],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Arad trait-directional output score ranking")
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--top-n", type=int, default=1000, help="Save top-N ranked features")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--out", default=None, help="K-sweep feature file path")
    ap.add_argument("--ranking-out", default=None, help="Full ranking JSON path")
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(args.layer or cfg["layer"])
    sae_dir = Path(cfg["sae_dir"])
    alpha = float(cfg["alpha"])
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})

    out_path = Path(args.out or sae_dir / f"sae_output_score_l{layer}.json")
    ranking_path = Path(args.ranking_out or sae_dir / f"output_score_ranking_l{layer}.json")

    logger.info("Loading model on CPU for W_U...")
    model, _, _ = load_model_and_tokenizer(None, device=torch.device("cpu"))
    W_U = effective_lm_head(model)
    logger.info("W_U shape: %s", tuple(W_U.shape))

    logger.info("Loading SAE L%d on CPU...", layer)
    sae, sae_info = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=hidden_state_index(layer),
    )
    W_dec = sae.W_dec.detach().float()
    logger.info("W_dec shape: %s", tuple(W_dec.shape))

    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    v_layer = (alpha * v_full[layer].float())
    logger.info("Dense vector L%d norm (alpha=%s): %.3f", layer, alpha, float(v_layer.norm()))

    logger.info("Computing trait-directional output scores for %d features...", W_dec.shape[0])
    scores, logit_norms = compute_output_scores(W_dec, W_U, v_layer, chunk=args.chunk)

    sorted_idx = torch.argsort(scores, descending=True)
    top_n = min(args.top_n, len(sorted_idx))
    top_idx = sorted_idx[:top_n].tolist()
    ranked_fids = [int(i) for i in top_idx]
    ranked_scores = [float(scores[i]) for i in top_idx]

    logger.info("Top-10 output scores:")
    for rank, (fid, sc) in enumerate(zip(ranked_fids[:10], ranked_scores[:10]), start=1):
        logger.info("  #%d fid=%d score=%.4f logit_norm=%.3f", rank, fid, sc, float(logit_norms[fid]))

    # Overlap with OMP top-5 for diagnostics
    omp_path = sae_dir / f"omp_decomposition_262k_l{layer}.json"
    if omp_path.is_file():
        omp_doc = json.loads(omp_path.read_text(encoding="utf-8"))
        omp_rows = sorted(
            omp_doc.get("decomposition") or [],
            key=lambda r: abs(float(r.get("coefficient", 0))),
            reverse=True,
        )
        omp_top5 = {int(r["feature_id"]) for r in omp_rows[:5]}
        arad_top5 = set(ranked_fids[:5])
        logger.info("OMP top-5: %s", sorted(omp_top5))
        logger.info("Arad top-5: %s", sorted(arad_top5))
        logger.info("Overlap: %s", sorted(omp_top5 & arad_top5))

    ranking_payload = {
        "method": "output_score_arad_trait_directional",
        "trait": cfg["trait"],
        "layer": layer,
        "alpha_dense": alpha,
        "sae_id": sae_info.get("sae_id") or cfg["sae_id"],
        "formula": "cosine(W_U @ W_dec[f], W_U @ (alpha * v_layer))",
        "n_features": int(W_dec.shape[0]),
        "top_n": top_n,
        "ranking": [
            {
                "rank": i + 1,
                "feature_id": fid,
                "output_score": round(sc, 6),
                "logit_norm": round(float(logit_norms[fid]), 4),
            }
            for i, (fid, sc) in enumerate(zip(ranked_fids, ranked_scores))
        ],
    }
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")
    logger.info("Wrote ranking %s (%d features)", ranking_path, top_n)

    sweep_payload = {
        "method": "output_score_arad",
        "trait": cfg["trait"],
        "layer": layer,
        "alpha_dense": alpha,
        "weight_mode": "output_score",
        "note": "Top-K by trait-directional output score; weights = scores",
        "ks_planned": ks,
        "results": build_k_sweep_rows(ranked_fids, ranked_scores, ks),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")
    logger.info("Wrote K-sweep feature file %s (K=%s)", out_path, ks)


if __name__ == "__main__":
    main()
