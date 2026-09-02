#!/usr/bin/env python3
"""Export t-SNE (and PCA) coords for Big Five ladder activations in embedding space.

Reads cached ``centroids_*.pt`` / ``ladder_vectors_*.pt`` from a ladder run dir.
Points are mean assistant-token residual activations at the steer layer (default L15).

Usage:
    python3 scripts/build_big_five_tsne_data.py \\
        --vectors-dir results/final_cycle/ladder \\
        --out app/static/big_five_tsne.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def _load_prompt_meta(vectors_dir: Path, trait: str) -> dict[tuple[int, int], dict]:
    path = vectors_dir / f"prompt_ladder_{trait}.json"
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text())
    out: dict[tuple[int, int], dict] = {}
    for row in blob.get("administrations", []):
        key = (int(row["level"]), int(row["variant"]))
        out[key] = {
            "target_ev": row.get("target_ev"),
            "ev_scores": row.get("ev_scores", {}),
            "usable": row.get("usable", True),
        }
    return out


def _mean_ev_scores(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for row in rows:
        for trait, val in row.items():
            if val is None:
                continue
            acc.setdefault(trait, []).append(float(val))
    return {trait: float(np.mean(vals)) for trait, vals in acc.items() if vals}


def _collect_points(vectors_dir: Path, layer: int) -> tuple[list[dict], np.ndarray, dict]:
    rows: list[dict] = []
    vectors: list[np.ndarray] = []

    for trait in TRAITS:
        centroids_path = vectors_dir / f"centroids_{trait}.pt"
        if not centroids_path.is_file():
            raise FileNotFoundError(f"missing {centroids_path}")
        blob = torch.load(centroids_path, map_location="cpu", weights_only=False)
        acts = blob["activations"].float().numpy()  # (levels, variants, layers, d)
        levels = [int(x) for x in blob["levels"]]
        meta = _load_prompt_meta(vectors_dir, trait)

        for li, level in enumerate(levels):
            variant_stack = acts[li, :, layer, :]
            mean_vec = variant_stack.mean(axis=0)
            mean_meta = meta.get((level, 0), {})
            variant_ev_rows = [
                meta.get((level, vi), {}).get("ev_scores") or {}
                for vi in range(variant_stack.shape[0])
            ]
            mean_ev_scores = _mean_ev_scores(variant_ev_rows)
            rows.append(
                {
                    "id": f"{trait}:L{level}:mean",
                    "trait": trait,
                    "level": level,
                    "variant": None,
                    "kind": "level_mean",
                    "target_ev": mean_meta.get("target_ev"),
                    "trait_ev": mean_ev_scores.get(trait),
                    "ev_scores": mean_ev_scores,
                }
            )
            vectors.append(mean_vec)

            for vi in range(variant_stack.shape[0]):
                m = meta.get((level, vi), {})
                ev_scores = m.get("ev_scores") or {}
                rows.append(
                    {
                        "id": f"{trait}:L{level}:v{vi}",
                        "trait": trait,
                        "level": level,
                        "variant": vi,
                        "kind": "variant",
                        "target_ev": m.get("target_ev"),
                        "trait_ev": (ev_scores or {}).get(trait),
                        "ev_scores": ev_scores,
                        "usable": m.get("usable", True),
                    }
                )
                vectors.append(variant_stack[vi])

        vec_path = vectors_dir / f"ladder_vectors_{trait}.pt"
        if vec_path.is_file():
            vb = torch.load(vec_path, map_location="cpu", weights_only=False)
            for kind in ("v_pc1", "v_endpoint", "v_ordinal", "v_probe"):
                if kind not in vb:
                    continue
                arr = vb[kind].float().numpy()
                rows.append(
                    {
                        "id": f"{trait}:{kind}",
                        "trait": trait,
                        "level": None,
                        "variant": None,
                        "kind": kind,
                        "target_ev": None,
                        "trait_ev": None,
                    }
                )
                vectors.append(arr[layer])

    model_id = None
    sample = vectors_dir / "centroids_openness.pt"
    if sample.is_file():
        model_id = torch.load(sample, map_location="cpu", weights_only=False).get("model_id")

    return rows, np.stack(vectors, axis=0), {"model_id": model_id}


def _attach_coords(rows: list[dict], matrix: np.ndarray, *, prefix: str, dims: int) -> None:
    for i, row in enumerate(rows):
        for d in range(dims):
            row[f"{prefix}{d + 1}"] = float(matrix[i, d])


def build(
    vectors_dir: Path,
    *,
    layer: int = 15,
    perplexity: float = 25.0,
    seed: int = 42,
) -> dict:
    rows, X, meta = _collect_points(vectors_dir, layer)
    n = X.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 points, got {n}")

    p = min(perplexity, max(5.0, (n - 1) / 3.0))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=3, random_state=seed)
    pca_coords = pca.fit_transform(Xs)

    tsne = TSNE(
        n_components=3,
        perplexity=p,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
    )
    tsne_coords = tsne.fit_transform(Xs)

    _attach_coords(rows, pca_coords, prefix="pca", dims=3)
    _attach_coords(rows, tsne_coords, prefix="tsne", dims=3)

    def extent(prefix: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for d in range(3):
            col = [r[f"{prefix}{d + 1}"] for r in rows]
            out[f"{prefix}{d + 1}"] = [min(col), max(col)]
        return out

    return {
        "title": "Big Five — embedding t-SNE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": meta.get("model_id"),
        "steer_layer": layer,
        "hidden_dim": int(X.shape[1]),
        "n_points": n,
        "perplexity": p,
        "method": (
            f"t-SNE (perplexity={p:.1f}, init=pca) on L{layer} residual activations "
            f"({X.shape[1]}-dim), standardized. Includes variant centroids, level means, "
            "and ladder direction vectors."
        ),
        "pca_var": [float(v) for v in pca.explained_variance_ratio_[:3]],
        "extent": {
            "pca": extent("pca"),
            "tsne": extent("tsne"),
        },
        "points": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vectors-dir",
        type=Path,
        default=REPO_ROOT / "results/final_cycle/ladder",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "app/static/big_five_tsne.json",
    )
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--perplexity", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payload = build(
        args.vectors_dir.resolve(),
        layer=args.layer,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} ({payload['n_points']} points, layer {payload['steer_layer']})")


if __name__ == "__main__":
    main()
