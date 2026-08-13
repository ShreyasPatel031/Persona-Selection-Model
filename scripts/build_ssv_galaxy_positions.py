#!/usr/bin/env python3
"""
Precompute 3D UMAP positions for all 262k SAE features from their decoder columns.

Usage (run on GPU VM):
  python scripts/build_ssv_galaxy_positions.py \
      --out app/static/ssv_galaxy_positions.json \
      [--layer 16] [--n-neighbors 15] [--min-dist 0.1] [--sample-fstats]

The output JSON is loaded by ssv_galaxy.html to position the 262k feature dots.
Spatial proximity = cosine similarity of decoder columns = semantic overlap of features.

If --z-cache-{trait} paths are given, also computes per-trait F-statistics so the
background galaxy can be tinted by F-stat relevance per trait.

Approximate runtimes:
  UMAP on 262k x 2560: ~8-15 min on CPU, ~4-8 min on GPU (umap-learn uses CPU by default)
  cuML UMAP on GPU: ~30-90 sec

Reduce dimensionality first with PCA(50) before UMAP to speed things up dramatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait, sae_id_for_layer


def pca_reduce(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Fast PCA via SVD. Returns (n, n_components) float32."""
    print(f"  PCA {X.shape} -> ({X.shape[0]}, {n_components})...", flush=True)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)


def compute_f_stats_from_cache(z_cache_path: Path) -> np.ndarray:
    """Load a probe z-cache and compute F-statistics between pos and neg samples."""
    cached = np.load(z_cache_path)
    z_all, y_all = cached["z"].astype(np.float64), cached["y"]
    mask_pos = y_all > 0.5
    mask_neg = ~mask_pos
    n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())
    n = n_pos + n_neg
    grand = z_all.mean(axis=0)
    mean_pos = z_all[mask_pos].mean(axis=0)
    mean_neg = z_all[mask_neg].mean(axis=0)
    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((z_all[mask_pos] - mean_pos) ** 2).sum(axis=0) + (
        (z_all[mask_neg] - mean_neg) ** 2
    ).sum(axis=0)
    ms_within = ss_within / max(n - 2, 1)
    f = np.divide(ss_between, ms_within, out=np.zeros_like(ss_between), where=ms_within > 1e-12)
    return f.astype(np.float32)


def normalize_f_stats(f: np.ndarray, top_k: int = 1000) -> np.ndarray:
    """Normalize F-stats to [0, 1] using the top-K threshold as the max."""
    threshold = np.sort(f)[::-1][min(top_k, len(f) - 1)]
    return np.clip(f / max(threshold, 1e-8), 0.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default="app/static/ssv_galaxy_positions.json")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--pca-components", type=int, default=50, help="PCA dims before UMAP (0=skip PCA)")
    ap.add_argument("--z-cache-good", default=None, help="Path to probe_z_cache_l{layer}.npz for good trait")
    ap.add_argument("--z-cache-evil", default=None, help="Path to probe_z_cache for evil trait")
    ap.add_argument("--z-cache-lawful", default=None, help="Path to probe_z_cache for lawful (usually l15)")
    ap.add_argument("--z-cache-chaotic", default=None, help="Path to probe_z_cache for chaotic trait")
    ap.add_argument("--use-cuml", action="store_true", help="Use cuML GPU UMAP (requires cuml installed)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load SAE decoder ───────────────────────────────────────────────────
    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    sae_id = sae_id_for_layer(layer)
    print(f"Loading SAE: layer={layer} sae_id={sae_id}", flush=True)
    sae, _ = load_sae_for_layer(torch.device("cpu"), release=SAE_RELEASE, sae_id=sae_id, hidden_state_index=layer)
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # (d_sae, d_model)
    d_sae, d_model = W_dec.shape
    print(f"W_dec shape: {W_dec.shape}", flush=True)

    # ── Optionally normalize decoder rows to unit sphere ───────────────────
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_dec_normed = W_dec / np.maximum(norms, 1e-8)

    # ── PCA reduction ──────────────────────────────────────────────────────
    if args.pca_components > 0:
        X_reduced = pca_reduce(W_dec_normed, n_components=args.pca_components)
    else:
        X_reduced = W_dec_normed

    # ── UMAP ───────────────────────────────────────────────────────────────
    print(f"Running UMAP(n_components=3, n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...", flush=True)
    if args.use_cuml:
        from cuml import UMAP as cuUMAP
        reducer = cuUMAP(n_components=3, n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42)
    else:
        import umap
        reducer = umap.UMAP(n_components=3, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                            random_state=42, low_memory=True, verbose=True)

    embedding = reducer.fit_transform(X_reduced)  # (d_sae, 3)
    print(f"UMAP done. Embedding shape: {embedding.shape}", flush=True)

    # Normalize embedding to [-1, 1] per axis
    for dim in range(3):
        lo, hi = embedding[:, dim].min(), embedding[:, dim].max()
        embedding[:, dim] = 2 * (embedding[:, dim] - lo) / max(hi - lo, 1e-8) - 1

    positions = embedding.astype(np.float16).tolist()  # reduce size

    # ── Per-trait F-statistics ─────────────────────────────────────────────
    f_stats_out: dict[str, list] = {}
    z_cache_map = {
        "good": args.z_cache_good,
        "evil": args.z_cache_evil,
        "lawful": args.z_cache_lawful,
        "chaotic": args.z_cache_chaotic,
    }
    for trait, cache_path in z_cache_map.items():
        if cache_path and Path(cache_path).exists():
            print(f"Computing F-stats for {trait} from {cache_path}...", flush=True)
            f = compute_f_stats_from_cache(Path(cache_path))
            f_norm = normalize_f_stats(f, top_k=1000)
            # Store as uint8 (0-255) to minimize JSON size — decode as /255 in JS
            f_stats_out[trait] = [int(round(v * 255)) for v in f_norm.tolist()]
            print(f"  {trait}: top F-stat={f.max():.1f}, nnz={int((f > 0.01).sum())}", flush=True)
        else:
            print(f"  No z-cache for {trait}, skipping F-stats.", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    payload = {
        "meta": {
            "layer": layer,
            "sae_id": sae_id,
            "n_features": d_sae,
            "d_model": d_model,
            "umap": {
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "pca_components": args.pca_components,
            },
        },
        "positions": positions,   # list of [x, y, z], length d_sae
        "f_stats": f_stats_out,   # trait -> uint8 list of length d_sae
    }

    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {out_path} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
