#!/usr/bin/env python3
"""
Orthogonal Matching Pursuit decomposition of v_dense into SAE decoder columns.

Pure linear algebra — no model inference, no judging, no gradients.
Finds the sparsest set of decoder columns whose weighted sum approximates v_dense.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.trait_sae_config import (
    DEFAULT_K_MAX,
    SAE_RELEASE,
    resolve_trait,
)


def omp_decompose(
    W: torch.Tensor,
    target: torch.Tensor,
    k_max: int,
    report_ks: set[int] | None = None,
) -> tuple[list[int], list[float], list[dict]]:
    W_unit = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    residual = target.clone()
    selected: list[int] = []
    coefs: list[float] = []
    checkpoints: list[dict] = []

    if report_ks is None:
        report_ks = {
            1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100,
            150, 200, 450, 500, 750, 1000, k_max,
        }
    report_ks = {k for k in report_ks if k <= k_max}

    for k in range(k_max):
        best = int((W_unit @ residual).abs().argmax().item())
        c = float((W[best] @ residual) / (W[best] @ W[best]))
        selected.append(best)
        coefs.append(c)
        residual = residual - c * W[best]

        if (k + 1) in report_ks:
            recon = sum(coefs[i] * W[selected[i]] for i in range(len(selected)))
            cos_sim = float(
                torch.nn.functional.cosine_similarity(
                    target.unsqueeze(0), recon.unsqueeze(0)
                ).item()
            )
            res_frac = float(residual.norm().item() / target.norm().item())
            nr = float(recon.norm().item() / target.norm().item())
            checkpoints.append(
                {
                    "k": k + 1,
                    "feature_id": int(best),
                    "coefficient": round(c, 4),
                    "cosine": round(cos_sim, 4),
                    "norm_ratio": round(nr, 4),
                    "residual_frac": round(res_frac, 4),
                }
            )
            print(
                f"k={k+1:>4d}  feat={best:>6d}  coef={c:+.2f}  "
                f"cos={cos_sim:.4f}  norm_ratio={nr:.3f}  residual={res_frac:.4f}"
            )

    return selected, coefs, checkpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default=None, help="good|evil|lawful|chaotic")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--sae-id", default=None)
    ap.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON (default: persona_runs/<run_id>/sae/omp_decomposition_262k_l<N>.json)",
    )
    args = ap.parse_args()

    if args.trait:
        cfg = resolve_trait(args.trait)
        run_id = cfg["run_id"]
        layer = cfg["layer"]
        vectors_path = Path(args.vectors or cfg["vectors"])
        sae_id = args.sae_id or cfg["sae_id"]
        hs_index = cfg["hs_index"]
        out_path = Path(args.out or cfg["decomp"])
        trait_label = cfg["trait"]
    else:
        run_id = args.run_id or "dnd_good_scale"
        layer = args.layer if args.layer is not None else 16
        paths = resolve_trait("good") if run_id == "dnd_good_scale" else None
        if paths and paths["run_id"] == run_id:
            vectors_path = Path(args.vectors or paths["vectors"])
            sae_id = args.sae_id or paths["sae_id"]
            hs_index = paths["hs_index"]
            out_path = Path(args.out or paths["decomp"])
        else:
            from scripts.trait_sae_config import hidden_state_index, run_paths, sae_id_for_layer

            p = run_paths(run_id, layer)
            vectors_path = Path(args.vectors or p["vectors"])
            sae_id = args.sae_id or sae_id_for_layer(layer)
            hs_index = hidden_state_index(layer)
            out_path = Path(args.out or p["decomp"])
        trait_label = run_id

    v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
    target = v_full[layer].float()

    from app.phase2 import load_sae_for_layer

    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=sae_id,
        hidden_state_index=hs_index,
    )
    W = sae.W_dec.detach().float()

    print(f"=== OMP decompose trait={trait_label} run_id={run_id} layer={layer} sae={sae_id} ===")
    print(f"target_norm={float(target.norm()):.2f}  k_max={args.k_max}")

    selected, coefs, checkpoints = omp_decompose(W, target, args.k_max)

    recon = sum(coefs[i] * W[selected[i]] for i in range(len(selected)))
    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            target.unsqueeze(0), recon.unsqueeze(0)
        ).item()
    )
    res_frac = float((target - recon).norm().item() / target.norm().item())

    print(f"\nFinal decomposition ({len(selected)} features): cos={cos_sim:.4f}")
    pairs = sorted(zip(selected, coefs), key=lambda x: abs(x[1]), reverse=True)
    for fid, c in pairs[:25]:
        print(f"  feat {fid:>6d}  weight={c:+.4f}")

    k_at_99 = next((c["k"] for c in checkpoints if c["cosine"] >= 0.99), None)

    result = {
        "method": "orthogonal_matching_pursuit",
        "trait": trait_label,
        "run_id": run_id,
        "layer": layer,
        "sae_id": sae_id,
        "k_max": args.k_max,
        "n_features": len(selected),
        "target_norm": round(float(target.norm()), 2),
        "checkpoints": checkpoints,
        "k_at_cos_99": k_at_99,
        "decomposition": [
            {"feature_id": int(s), "coefficient": round(float(c), 4)}
            for s, c in zip(selected, coefs)
        ],
        "final_cosine": round(cos_sim, 4),
        "final_residual_frac": round(res_frac, 4),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
