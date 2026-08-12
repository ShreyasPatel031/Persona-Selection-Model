#!/usr/bin/env python3
"""Gram-Schmidt orthogonalize chaotic vector against good at each layer.

Writes a new vectors .pt with the same schema as persona_vectors.pt:
  {"v": Tensor[n_layers, hidden_dim], ...}

Usage:
  python scripts/ortho_vectors.py \\
    --chaotic-vectors persona_runs/dnd_chaotic/vectors/persona_vectors.pt \\
    --good-vectors persona_runs/dnd_good/vectors/persona_vectors.pt \\
    --out-vectors persona_runs/dnd_chaotic/vectors/persona_vectors_ortho_vs_good.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_v(path: Path) -> tuple[torch.Tensor, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "v" not in ck:
        raise KeyError(f"No 'v' key in {path}; keys={list(ck.keys())}")
    return ck["v"].float().clone(), ck


def orthogonalize_against(
    v_target: torch.Tensor,
    v_remove: torch.Tensor,
) -> tuple[torch.Tensor, list[dict]]:
    """Per-layer: v_target -= proj(v_target onto v_remove)."""
    if v_target.shape != v_remove.shape:
        raise ValueError(f"shape mismatch {tuple(v_target.shape)} vs {tuple(v_remove.shape)}")
    out = v_target.clone()
    rows: list[dict] = []
    for layer in range(v_target.shape[0]):
        a = v_target[layer]
        b = v_remove[layer]
        na = float(a.norm().item())
        nb = float(b.norm().item())
        if na < 1e-8 or nb < 1e-8:
            cos_before = float("nan")
            cos_after = float("nan")
            ratio = float("nan")
            out[layer] = a
        else:
            cos_before = float((a @ b) / (a.norm() * b.norm()))
            proj = ((a @ b) / (b @ b)) * b
            ortho = a - proj
            out[layer] = ortho
            cos_after = float((ortho @ b) / (ortho.norm() * b.norm() + 1e-12))
            ratio = float(ortho.norm() / a.norm())
        rows.append(
            {
                "layer": layer,
                "cos_before": cos_before,
                "cos_after": cos_after,
                "norm_ratio": ratio,
                "norm_before": na,
                "norm_after": float(out[layer].norm().item()),
            }
        )
    return out, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Orthogonalize chaotic vs good vectors")
    parser.add_argument(
        "--chaotic-vectors",
        type=Path,
        default=Path("persona_runs/dnd_chaotic/vectors/persona_vectors.pt"),
    )
    parser.add_argument(
        "--good-vectors",
        type=Path,
        default=Path("persona_runs/dnd_good/vectors/persona_vectors.pt"),
    )
    parser.add_argument(
        "--out-vectors",
        type=Path,
        default=Path("persona_runs/dnd_chaotic/vectors/persona_vectors_ortho_vs_good.pt"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("persona_runs/ortho_chaotic_vs_good_diagnostics.json"),
    )
    parser.add_argument("--focus-layer", type=int, default=16)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    chaotic_path = args.chaotic_vectors if args.chaotic_vectors.is_absolute() else root / args.chaotic_vectors
    good_path = args.good_vectors if args.good_vectors.is_absolute() else root / args.good_vectors
    out_v = args.out_vectors if args.out_vectors.is_absolute() else root / args.out_vectors
    out_j = args.out_json if args.out_json.is_absolute() else root / args.out_json

    v_c, ck_c = _load_v(chaotic_path)
    v_g, _ = _load_v(good_path)
    v_ortho, rows = orthogonalize_against(v_c, v_g)

    focus = rows[args.focus_layer] if 0 <= args.focus_layer < len(rows) else None
    print(f"Loaded chaotic {tuple(v_c.shape)} from {chaotic_path}", file=sys.stderr)
    print(f"Loaded good    {tuple(v_g.shape)} from {good_path}", file=sys.stderr)
    if focus:
        print(
            f"L{args.focus_layer}: cos {focus['cos_before']:.4f} → {focus['cos_after']:.4f} "
            f"| ||v_ortho||/||v|| = {focus['norm_ratio']:.4f}",
            file=sys.stderr,
        )

    out_ck = dict(ck_c)
    out_ck["v"] = v_ortho
    out_ck["ortho_meta"] = {
        "method": "gram_schmidt",
        "target": "chaotic",
        "removed": "good",
        "source_chaotic": str(chaotic_path),
        "source_good": str(good_path),
        "focus_layer": args.focus_layer,
        "focus": focus,
    }
    out_v.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, out_v)
    print(f"Wrote {out_v}", file=sys.stderr)

    doc = {
        "method": "gram_schmidt",
        "target": "chaotic",
        "removed": "good",
        "source_chaotic": str(chaotic_path),
        "source_good": str(good_path),
        "out_vectors": str(out_v),
        "focus_layer": args.focus_layer,
        "focus": focus,
        "per_layer": rows,
    }
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_j}", file=sys.stderr)
    print(json.dumps(focus, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
