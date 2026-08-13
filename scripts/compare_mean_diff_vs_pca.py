#!/usr/bin/env python3
"""Compare mean-diff (CAA) vs PCA-PC1 (RepE LAT) as trait directions on the same rollouts.

For each layer:
  v_md  = mean(h_pos) - mean(h_neg)           # Chen Step D / CAA
  v_pca = PC1 of paired diffs (h_pos - h_neg) # RepE LAT first axis

Metrics (which finds trait better on held-in rollouts):
  - cosine(v_md, v_pca)
  - sep_md  = mean(h_pos·û_md) - mean(h_neg·û_md)
  - sep_pca = mean(h_pos·û_pca) - mean(h_neg·û_pca)
  - corr(projection, judge_score) per direction

Output: app/static/good_mean_diff_vs_pca.json (and logs copy)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.persona.activations import (  # noqa: E402
    iter_kept_rollouts,
    load_model_and_tokenizer,
    mean_residuals_over_assistant,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LAYERS = [8, 12, 15, 16, 20, 28]


def _pc1_direction(diffs: np.ndarray) -> tuple[np.ndarray, float, float]:
    """diffs (n, d) -> unit PC1, pc1_var ratio, pc2_var ratio."""
    Xc = diffs - diffs.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(diffs) - 1, 1)
    total = float(var.sum()) or 1.0
    pc1 = Vt[0].astype(np.float64)
    n = np.linalg.norm(pc1)
    if n < 1e-12:
        pc1 = pc1
    else:
        pc1 = pc1 / n
    return pc1, float(var[0] / total), float(var[1] / total) if len(var) > 1 else 0.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _sep(pos: np.ndarray, neg: np.ndarray, u: np.ndarray) -> float:
    """Mean margin: avg(h_pos·û) - avg(h_neg·û)."""
    return float((pos @ u).mean() - (neg @ u).mean())


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="dnd_good")
    ap.add_argument("--max-per-arm", type=int, default=80)
    ap.add_argument("--layers", default=",".join(str(l) for l in DEFAULT_LAYERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=REPO / "app/static/good_mean_diff_vs_pca.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    run_dir = REPO / "persona_runs" / args.run_id
    rollouts = run_dir / "rollouts" / "rollouts.jsonl"
    if not rollouts.is_file():
        raise SystemExit(f"missing {rollouts}")

    pos_rows, neg_rows = [], []
    for o in iter_kept_rollouts(rollouts):
        if o.get("arm") == "pos":
            pos_rows.append(o)
        elif o.get("arm") == "neg":
            neg_rows.append(o)

    rng = random.Random(args.seed)
    if len(pos_rows) > args.max_per_arm:
        pos_rows = rng.sample(pos_rows, args.max_per_arm)
    if len(neg_rows) > args.max_per_arm:
        neg_rows = rng.sample(neg_rows, args.max_per_arm)
    n = min(len(pos_rows), len(neg_rows))
    logger.info("Using %d pos + %d neg (paired n=%d)", len(pos_rows), len(neg_rows), n)

    model, tok, dev = load_model_and_tokenizer()

    def collect(rows: list[dict]) -> tuple[np.ndarray, list[float | None]]:
        mats = []
        scores = []
        for i, o in enumerate(rows):
            logger.info("Forward %d/%d (%s)", i + 1, len(rows), o.get("arm"))
            m = mean_residuals_over_assistant(
                model, tok, dev, o["system"], o["question"], o["assistant_a"]
            )
            mats.append(m.float().cpu().numpy())
            sc = o.get("score")
            scores.append(float(sc) if sc is not None else None)
            if dev.type == "cuda" and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        return np.stack(mats, axis=0), scores

    pos_all, pos_scores = collect(pos_rows)
    neg_all, neg_scores = collect(neg_rows)
    pos = pos_all[:n]
    neg = neg_all[:n]
    pos_sc = pos_scores[:n]
    neg_sc = neg_scores[:n]

    # Saved Step D vector if present
    v_saved = None
    vec_pt = run_dir / "vectors" / "persona_vectors.pt"
    if vec_pt.is_file():
        ck = torch.load(vec_pt, map_location="cpu", weights_only=False)
        v_saved = ck["v"].float().numpy()

    layer_docs = []
    for l in layers:
        h_pos = pos[:, l, :]
        h_neg = neg[:, l, :]
        diffs = h_pos - h_neg

        v_md = h_pos.mean(axis=0) - h_neg.mean(axis=0)
        u_md = v_md / (np.linalg.norm(v_md) + 1e-12)

        v_pca, pc1_var, pc2_var = _pc1_direction(diffs)

        # Signed PC1: orient same hemisphere as mean-diff
        if np.dot(v_pca, u_md) < 0:
            v_pca = -v_pca

        sep_md = _sep(h_pos, h_neg, u_md)
        sep_pca = _sep(h_pos, h_neg, v_pca)

        proj_md = h_pos @ u_md
        proj_neg_md = h_neg @ u_md
        proj_pca = h_pos @ v_pca
        proj_neg_pca = h_neg @ v_pca

        all_proj_md = np.concatenate([proj_md, proj_neg_md])
        all_proj_pca = np.concatenate([proj_pca, proj_neg_pca])
        all_scores = np.array(
            [s if s is not None else np.nan for s in pos_sc + neg_sc], dtype=np.float64
        )
        valid = ~np.isnan(all_scores)
        corr_md = _corr(all_proj_md[valid], all_scores[valid]) if valid.sum() >= 3 else None
        corr_pca = _corr(all_proj_pca[valid], all_scores[valid]) if valid.sum() >= 3 else None

        saved_cos = None
        sep_saved = None
        if v_saved is not None:
            u_saved = v_saved[l] / (np.linalg.norm(v_saved[l]) + 1e-12)
            saved_cos = {
                "vs_mean_diff": round(_cos(u_saved, u_md), 4),
                "vs_pca_pc1": round(_cos(u_saved, v_pca), 4),
            }
            sep_saved = round(_sep(h_pos, h_neg, u_saved), 4)

        layer_docs.append({
            "layer": l,
            "n_pairs": n,
            "norm_mean_diff": round(float(np.linalg.norm(v_md)), 4),
            "cosine_mean_diff_vs_pca_pc1": round(_cos(v_md, v_pca), 4),
            "sep_mean_diff": round(sep_md, 4),
            "sep_pca_pc1": round(sep_pca, 4),
            "sep_ratio_pca_over_md": round(sep_pca / (sep_md + 1e-12), 4),
            "pc1_var": round(pc1_var, 4),
            "pc2_var": round(pc2_var, 4),
            "corr_proj_mean_diff_vs_judge": round(corr_md, 4) if corr_md is not None else None,
            "corr_proj_pca_pc1_vs_judge": round(corr_pca, 4) if corr_pca is not None else None,
            "mean_pos_score": round(float(np.nanmean(pos_sc)), 2) if any(s is not None for s in pos_sc) else None,
            "mean_neg_score": round(float(np.nanmean(neg_sc)), 2) if any(s is not None for s in neg_sc) else None,
            "saved_step_d": {
                "cosine_vs_directions": saved_cos,
                "sep_along_saved_v": sep_saved,
            } if v_saved is not None else None,
            "winner_sep": "pca" if sep_pca > sep_md else ("mean_diff" if sep_md > sep_pca else "tie"),
            "winner_judge_corr": (
                "pca" if (corr_pca or -1) > (corr_md or -1)
                else ("mean_diff" if (corr_md or -1) > (corr_pca or -1) else "tie")
            ) if corr_md is not None and corr_pca is not None else None,
        })
        logger.info(
            "L%02d cos=%.3f sep_md=%.1f sep_pca=%.1f ratio=%.3f winner=%s",
            l, _cos(v_md, v_pca), sep_md, sep_pca, sep_pca / (sep_md + 1e-12),
            layer_docs[-1]["winner_sep"],
        )

    best_sep_md = max(layer_docs, key=lambda x: x["sep_mean_diff"])
    best_sep_pca = max(layer_docs, key=lambda x: x["sep_pca_pc1"])

    doc = {
        "trait": "good",
        "run_id": args.run_id,
        "method": (
            "Same pos/neg rollouts: compare CAA mean-diff vs RepE PC1 on paired diffs"
        ),
        "n_pos": len(pos_rows),
        "n_neg": len(neg_rows),
        "n_paired": n,
        "summary": {
            "best_sep_mean_diff_layer": best_sep_md["layer"],
            "best_sep_mean_diff": best_sep_md["sep_mean_diff"],
            "best_sep_pca_layer": best_sep_pca["layer"],
            "best_sep_pca": best_sep_pca["sep_pca_pc1"],
            "layers_pca_beats_md_sep": sum(1 for x in layer_docs if x["winner_sep"] == "pca"),
            "layers_md_beats_pca_sep": sum(1 for x in layer_docs if x["winner_sep"] == "mean_diff"),
        },
        "layers": layer_docs,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    logs_copy = REPO / "logs" / "good_mean_diff_vs_pca.json"
    logs_copy.parent.mkdir(parents=True, exist_ok=True)
    logs_copy.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", logs_copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
