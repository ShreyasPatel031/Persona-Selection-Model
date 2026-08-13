#!/usr/bin/env python3
"""Compare OMP reconstruction ceiling for 16k vs 262k SAE at a trait layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait, sae_id_for_layer


def run_omp_ceiling(W: torch.Tensor, target: torch.Tensor, k_max: int) -> list[dict]:
    Wu = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ck = {10, 50, 100, 200, 500, 750, 1000, 2000, 5000, 7500, 10000, k_max}
    ck = sorted(c for c in ck if c <= k_max)

    r = target.clone()
    sel, coef = [], []
    rows: list[dict] = []
    for k in range(k_max):
        b = int((Wu @ r).abs().argmax())
        c = float((W[b] @ r) / (W[b] @ W[b]))
        sel.append(b)
        coef.append(c)
        r = r - c * W[b]
        if (k + 1) in ck:
            recon = sum(coef[i] * W[sel[i]] for i in range(len(sel)))
            cos = float(
                torch.nn.functional.cosine_similarity(
                    target.unsqueeze(0), recon.unsqueeze(0)
                ).item()
            )
            nr = float(recon.norm() / target.norm())
            rows.append({"k": k + 1, "cosine": round(cos, 4), "norm_ratio": round(nr, 4)})
            print(f"  K={k+1:5d}  cos={cos:.4f}  norm_ratio={nr:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.trait:
        cfg = resolve_trait(args.trait)
    else:
        cfg = resolve_trait("good")

    layer = int(args.layer if args.layer is not None else cfg["layer"])
    vectors_path = Path(args.vectors or cfg["vectors"])
    out_path = Path(args.out or cfg["geometry"])

    target = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"][
        layer
    ].float()

    report = {
        "trait": cfg.get("trait"),
        "run_id": cfg["run_id"],
        "layer": layer,
        "target_norm": round(float(target.norm()), 2),
        "widths": {},
    }

    for label, sid in [
        ("16k", sae_id_for_layer(layer, "16k")),
        ("262k", sae_id_for_layer(layer, "262k")),
    ]:
        sae, _ = load_sae_for_layer(
            torch.device("cpu"),
            release=SAE_RELEASE,
            sae_id=sid,
            hidden_state_index=layer + 1,
        )
        W = sae.W_dec.detach().float()
        d_sae = W.shape[0]
        k_max = d_sae if d_sae <= 16384 else 10000
        print(f"=== {label} d_sae={d_sae} k_max={k_max} ===")
        rows = run_omp_ceiling(W, target, k_max)
        report["widths"][label] = {
            "sae_id": sid,
            "d_sae": d_sae,
            "k_max": k_max,
            "checkpoints": rows,
        }
        print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
