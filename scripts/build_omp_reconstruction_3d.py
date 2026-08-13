#!/usr/bin/env python3
"""
Build 3D PCA coordinates for OMP reconstruction viz.

Run on gemma-mvp (needs SAE + persona vectors). Pure geometry — no generation.

  .venv/bin/python scripts/build_omp_reconstruction_3d.py --trait good
  .venv/bin/python scripts/build_omp_reconstruction_3d.py --trait evil

Writes app/static/omp_reconstruction_3d_{trait}.json

Projection:
  1. Fit PCA on the OMP feature decoder columns W_dec[f] only (not the target).
  2. Whiten PC1–3 so each axis has unit variance — spreads the feature cloud.
  3. Project v_dense and cumulative reconstructions into that same frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.trait_sae_config import SAE_RELEASE, resolve_trait


def build_frame(dirs: np.ndarray, target: np.ndarray):
    """Build a 3D frame that actually shows structure.

    Axis Y  = direction of v_dense (progress toward the target).
    Axes X,Z = top-2 PCA directions of the feature decoder columns AFTER
               removing their component along Y (the off-axis scatter).

    This keeps the target visible (it lives along Y) while letting the
    feature cloud spread in the X-Z plane instead of collapsing onto the
    target direction. Returns a closure project(vec) -> (x, y, z) plus
    diagnostics.
    """
    u = target / (np.linalg.norm(target) + 1e-12)  # (d,) target axis

    # Off-axis residual of each decoder column (remove along-target part).
    along = dirs @ u  # (k,)
    resid = dirs - np.outer(along, u)  # (k, d)
    resid_c = resid - resid.mean(axis=0)

    _, svals, vt = np.linalg.svd(resid_c, full_matrices=False)
    e1 = vt[0]
    e2 = vt[1] if vt.shape[0] > 1 else np.zeros_like(e1)
    # Orthonormalize e1,e2 against u for a clean orthogonal frame.
    e1 = e1 - (e1 @ u) * u
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = e2 - (e2 @ u) * u - (e2 @ e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    var_total = float((svals ** 2).sum()) or 1.0
    offaxis_explained = [float((svals[i] ** 2) / var_total) for i in range(min(2, len(svals)))]

    def project(vec: np.ndarray) -> np.ndarray:
        return np.array([vec @ e1, vec @ u, vec @ e2], dtype=np.float64)

    return project, offaxis_explained, svals.astype(np.float64)


def build_trait(trait: str, k_max: int, out_path: Path) -> dict:
    cfg = resolve_trait(trait)
    layer = int(cfg["layer"])
    vectors_path = Path(cfg["vectors"])
    decomp_path = Path(cfg["decomp"])
    sae_id = cfg["sae_id"]
    hs_index = cfg["hs_index"]

    decomp = json.loads(decomp_path.read_text(encoding="utf-8"))
    rows = (decomp.get("decomposition") or [])[:k_max]
    fids = [int(r["feature_id"]) for r in rows]
    coefs = [float(r["coefficient"]) for r in rows]

    label_by_fid: dict[int, dict] = {}
    bubble = Path(f"app/static/ssv_bubble_viz_omp_data_{trait}.json")
    if bubble.is_file():
        bdoc = json.loads(bubble.read_text(encoding="utf-8"))
        for lvl in bdoc.get("k_levels") or []:
            for f in lvl.get("features") or []:
                fid = int(f["fid"])
                if fid not in label_by_fid:
                    label_by_fid[fid] = {
                        "title": f.get("title") or f.get("label") or "",
                        "sign": f.get("sign") or ("pos" if float(f.get("weight", 0)) >= 0 else "neg"),
                        "label_source": f.get("label_source"),
                    }

    print(f"Loading vectors {vectors_path} …")
    v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
    target = v_full[layer].float().numpy().astype(np.float64)

    print(f"Loading SAE {sae_id} …")
    from app.phase2 import load_sae_for_layer

    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=sae_id,
        hidden_state_index=hs_index,
    )
    W = sae.W_dec.detach().float().numpy().astype(np.float64)
    dirs = W[fids]  # (k, d) raw decoder columns

    # --- Frame: Y = target axis, X/Z = off-axis PCA of feature columns ---
    project, offaxis_explained, svals = build_frame(dirs, target)
    print(
        "Off-axis PCA explained (X,Z): "
        + ", ".join(f"{e:.1%}" for e in offaxis_explained)
    )

    # Feature points: unit decoder directions, then amplify off-axis (X,Z) so the
    # cloud isn't a pancake along the target axis.
    unit_dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    feat_raw = np.array([project(u) for u in unit_dirs])
    target_pos = project(target)
    ambient_origin = project(np.zeros_like(target))
    contrib_raw = np.array([project(coefs[i] * dirs[i]) for i in range(len(fids))])

    # Whiten X and Z on the feature cloud, then stretch so XZ RMS ≈ 1.35 × |target|.
    # Y stays proportional to alignment with v_dense (honest trait progress).
    feat_c = feat_raw - ambient_origin
    sx = float(feat_c[:, 0].std() or 1e-8)
    sz = float(feat_c[:, 2].std() or 1e-8)
    tlen = float(abs(target_pos[1] - ambient_origin[1]) or 1.0)
    xz_target_rms = 2.4 * tlen

    def stretch(p: np.ndarray) -> np.ndarray:
        q = p - ambient_origin
        return ambient_origin + np.array(
            [q[0] / sx * xz_target_rms, q[1], q[2] / sz * xz_target_rms],
            dtype=np.float64,
        )

    feat_pos = np.array([stretch(p) for p in feat_raw])
    target_pos = stretch(target_pos)
    contrib_pos = np.array([stretch(p) for p in contrib_raw])
    # Keep ambient_origin as-is (already ~0)

    features = []
    for i, (fid, c) in enumerate(zip(fids, coefs)):
        meta = label_by_fid.get(fid, {})
        sign = meta.get("sign") or ("pos" if c >= 0 else "neg")
        features.append(
            {
                "fid": fid,
                "order": i + 1,
                "coefficient": round(c, 4),
                "sign": sign,
                "title": meta.get("title") or f"F{fid}",
                "pos": [round(float(x), 5) for x in feat_pos[i]],
                "contrib": [round(float(x), 5) for x in contrib_pos[i]],
            }
        )

    recon = np.zeros_like(target)
    steps = []
    t_norm = float(np.linalg.norm(target) or 1.0)
    for k in range(1, len(fids) + 1):
        recon = recon + coefs[k - 1] * dirs[k - 1]
        tip = stretch(project(recon))
        cos = float(np.dot(target, recon) / (t_norm * (float(np.linalg.norm(recon)) + 1e-8)))
        res_frac = float(np.linalg.norm(target - recon) / t_norm)
        steps.append(
            {
                "k": k,
                "fid": fids[k - 1],
                "coefficient": round(coefs[k - 1], 4),
                "recon": [round(float(x), 5) for x in tip],
                "cosine": round(cos, 4),
                "residual_frac": round(res_frac, 4),
            }
        )

    # Axis interpretation: top features by |loading| on each display axis
    def top_on_axis(axis: int, n: int = 5) -> list[dict]:
        scored = sorted(
            (
                {
                    "fid": features[i]["fid"],
                    "title": features[i]["title"],
                    "loading": round(float(feat_pos[i, axis]), 3),
                }
                for i in range(len(features))
            ),
            key=lambda r: abs(r["loading"]),
            reverse=True,
        )
        return scored[:n]

    axes = {
        "x": {
            "name": "Off-axis PC1",
            "meaning": "Largest residual direction after removing alignment with v_dense. Features far on ±X differ from the trait vector sideways (not more/less of the trait).",
            "explained_offaxis": round(offaxis_explained[0], 4) if offaxis_explained else None,
            "color": "#c45c4a",
            "top_features": top_on_axis(0),
        },
        "y": {
            "name": "Trait / v_dense axis",
            "meaning": "Direction of the dense persona vector. Moving along +Y is progress toward reconstructing v_dense; sphere Y reflects how aligned that decoder column is with the trait.",
            "explained_offaxis": None,
            "color": "#3d8b6e",
            "top_features": top_on_axis(1),
        },
        "z": {
            "name": "Off-axis PC2",
            "meaning": "Second residual direction orthogonal to both v_dense and PC1. Captures remaining sideways diversity among OMP features.",
            "explained_offaxis": round(offaxis_explained[1], 4) if len(offaxis_explained) > 1 else None,
            "color": "#2b6cb0",
            "top_features": top_on_axis(2),
        },
    }

    # Diagnostics for spread
    feat_std = feat_pos.std(axis=0)
    print(f"Feature cloud std (x,y,z): {feat_std}")
    print(f"Target: {target_pos}   Origin: {ambient_origin}")

    out = {
        "meta": {
            "trait": trait,
            "layer": layer,
            "sae_id": sae_id,
            "method": "omp_target_axis_offaxis_pca_xz_stretched",
            "k_max": len(fids),
            "target_norm": round(t_norm, 2),
            "final_cosine": steps[-1]["cosine"] if steps else None,
            "offaxis_explained": [round(e, 4) for e in offaxis_explained],
            "xz_stretch_rms": round(xz_target_rms, 2),
            "note": (
                "Y = v_dense (trait progress). X/Z = off-axis PCA of decoder columns, "
                "whitened and stretched so XZ RMS ≈ 2.4×|target| for visibility. "
                "Off-axis variance is small in residual space (~5%); stretch is display-only."
            ),
        },
        "axes": axes,
        "origin": [round(float(x), 5) for x in ambient_origin],
        "target": [round(float(x), 5) for x in target_pos],
        "features": features,
        "steps": steps,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote {out_path}  features={len(features)}  final_cos={out['meta']['final_cosine']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good", choices=["good", "evil", "lawful", "chaotic", "male", "female"])
    ap.add_argument("--k-max", type=int, default=100, help="OMP features to include (greedy order)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"app/static/omp_reconstruction_3d_{args.trait}.json")
    build_trait(args.trait, args.k_max, out)


if __name__ == "__main__":
    main()
