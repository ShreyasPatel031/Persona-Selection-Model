#!/usr/bin/env python3
"""SSV optimization with feature mask = OMP top-K (Plan E).

Runs optimize_v_steer constrained to OMP decomposition features at each K,
saving weights to sae_ssv_omp_mask_l{layer}.json for use with ssv_omp_k_sweep.py --feature-file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import f_statistic_per_feature, optimize_v_steer
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, resolve_trait


def load_omp_top_k(sae_dir: Path, layer: int, k: int) -> list[int]:
    path = sae_dir / f"omp_decomposition_262k_l{layer}.json"
    rows = sorted(
        json.loads(path.read_text())["decomposition"],
        key=lambda r: abs(float(r["coefficient"])),
        reverse=True,
    )[:k]
    return [int(r["feature_id"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--ks", default="5,10,20,50,100,128,200,256,512,750,1000")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lambda-lm", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--z-cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    sae_dir = Path(cfg["sae_dir"])
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    z_cache = Path(args.z_cache or sae_dir / f"probe_z_cache_l{layer}.npz")
    out_path = Path(args.out or sae_dir / f"sae_ssv_omp_mask_l{layer}.json")

    if not z_cache.is_file():
        raise FileNotFoundError(f"Missing z cache: {z_cache}")

    cached = np.load(z_cache)
    z_all, y_all = cached["z"], cached["y"]
    print(f"Loaded z cache: {z_all.shape[0]} samples from {z_cache}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=hidden_state_index(layer),
    )
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()

    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()
    steer_norm = float(v_layer.norm())

    optim_kw = dict(
        n_iter=args.n_iter,
        lr=args.lr,
        lambda_lm=args.lambda_lm,
        beta=args.beta,
        opt_device=dev,
    )

    results = []
    for k in ks:
        k = min(k, d_sae)
        omp_fids = load_omp_top_k(sae_dir, layer, k)
        feature_mask = torch.zeros(d_sae, dtype=torch.float32)
        feature_mask[omp_fids] = 1.0

        print(f"\n=== SSV-on-OMP-mask K={k} ({len(omp_fids)} features) ===", flush=True)
        v_opt, _ = optimize_v_steer(
            z_neg_means, W_dec, mu_pos, mu_neg, feature_mask, **optim_kw,
        )
        v_residual = (W_dec.T @ v_opt.cpu().float()).float()
        raw_norm = float(v_residual.norm())
        if raw_norm > 1e-8:
            v_residual = v_residual * (steer_norm / raw_norm)

        active_mask = v_opt.abs() > 1e-8
        top_active = torch.argsort(v_opt.abs(), descending=True)
        top_fids = [int(f) for f in top_active if active_mask[f]]
        top_weights = [round(float(v_opt[f]), 6) for f in top_active if active_mask[f]]
        cos = F.cosine_similarity(v_layer.unsqueeze(0), v_residual.unsqueeze(0)).item()

        row = {
            "k": k,
            "method": "ssv_omp_mask",
            "omp_mask_fids": omp_fids,
            "n_active_features": int(active_mask.sum()),
            "feature_ids": top_fids,
            "feature_weights": top_weights,
            "cosine_vs_dense": round(cos, 4),
        }
        results.append(row)
        print(f"  active={row['n_active_features']} cos_dense={cos:.4f}", flush=True)

    payload = {
        "method": "ssv_omp_mask",
        "trait": cfg["trait"],
        "layer": layer,
        "note": "SSV optimize_v_steer with OMP top-K feature mask",
        "optim": {"n_iter": args.n_iter, "lr": args.lr, "lambda_lm": args.lambda_lm, "beta": args.beta},
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} ({len(results)} K values)", flush=True)


if __name__ == "__main__":
    main()
