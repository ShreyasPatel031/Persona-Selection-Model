#!/usr/bin/env python3
"""Orthogonalize D&D persona vectors for independent axis composition.

Strategy (matches the working chaotic⊥good composition test):
  - Order axis (lawful, chaotic): remove projection onto RAW good.
    This zeros cos(order_ortho, good) so order steering does not pull moral.
  - Good: left UNCHANGED (raw). Mutually ortho'ing good against lawful
    reintroduces cross-axis correlation between the vectors we actually add.
  - Evil: remove projection onto RAW lawful (small cleanup; cos≈0.14).

Writes:
  persona_runs/dnd_<trait>/vectors/persona_vectors_ortho.pt
  persona_runs/ortho_dnd_grid_diagnostics.json

Usage:
  python scripts/ortho_vectors_grid.py --config-json persona_runs/dnd_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAITS = ("lawful", "chaotic", "good", "evil")


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


def _load_v(path: Path) -> tuple[torch.Tensor, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "v" not in ck:
        raise KeyError(f"No 'v' key in {path}; keys={list(ck.keys())}")
    return ck["v"].float().clone(), ck


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float((a @ b) / (a.norm() * b.norm()))


def _corr_matrix(vecs: dict[str, torch.Tensor], layer: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for a in TRAITS:
        out[a] = {}
        for b in TRAITS:
            out[a][b] = _cosine(vecs[a][layer], vecs[b][layer])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ortho all 4 D&D vectors for 9-grid")
    parser.add_argument(
        "--config-json",
        type=Path,
        default=Path("persona_runs/dnd_config.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write all ortho .pt files under this dir as <trait>_ortho.pt",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("persona_runs/ortho_dnd_grid_diagnostics.json"),
    )
    parser.add_argument("--focus-layer", type=int, default=16)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = args.config_json if args.config_json.is_absolute() else root / args.config_json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    traits_cfg = cfg.get("traits", cfg)

    raw: dict[str, torch.Tensor] = {}
    cks: dict[str, dict] = {}
    src_paths: dict[str, str] = {}
    for name in TRAITS:
        p = Path(traits_cfg[name]["vectors"])
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            # Fallbacks for good_scale → good
            alt = root / f"persona_runs/dnd_{name}/vectors/persona_vectors.pt"
            if alt.is_file():
                print(f"WARN: {p} missing; using {alt}", file=sys.stderr)
                p = alt
            else:
                raise FileNotFoundError(f"Missing vectors for {name}: {p}")
        v, ck = _load_v(p)
        raw[name] = v
        cks[name] = ck
        src_paths[name] = str(p)
        print(f"Loaded {name:8s} {tuple(v.shape)} from {p}", file=sys.stderr)

    layer = int(args.focus_layer)
    before = _corr_matrix(raw, layer)
    print(f"\nCosine BEFORE ortho @ L{layer}:", file=sys.stderr)
    for a in TRAITS:
        row = "  ".join(f"{before[a][b]:+.4f}" for b in TRAITS)
        print(f"  {a:8s} {row}", file=sys.stderr)

    # Order ⊥ raw good; good stays raw; evil ⊥ raw lawful.
    ortho: dict[str, torch.Tensor] = {}
    per_layer: dict[str, list[dict]] = {}

    plans: dict[str, str | None] = {
        "lawful": "good",
        "chaotic": "good",
        "good": None,  # keep raw — composition partner for order_ortho
        "evil": "lawful",
    }
    for target, remove in plans.items():
        if remove is None:
            ortho[target] = raw[target].clone()
            per_layer[target] = [
                {
                    "layer": i,
                    "cos_before": 1.0 if i == layer else float("nan"),
                    "cos_after": 1.0 if i == layer else float("nan"),
                    "norm_ratio": 1.0,
                    "norm_before": float(raw[target][i].norm().item()),
                    "norm_after": float(raw[target][i].norm().item()),
                    "skipped": True,
                }
                for i in range(raw[target].shape[0])
            ]
            print(f"  {target}: UNCHANGED (raw)", file=sys.stderr)
            continue
        v_out, rows = orthogonalize_against(raw[target], raw[remove])
        ortho[target] = v_out
        per_layer[target] = rows
        focus = rows[layer]
        print(
            f"  {target} ⊥ {remove} @ L{layer}: "
            f"cos {focus['cos_before']:+.4f} → {focus['cos_after']:+.4f} "
            f"| norm_ratio={focus['norm_ratio']:.4f}",
            file=sys.stderr,
        )

    after = _corr_matrix(ortho, layer)
    print(f"\nCosine AFTER ortho @ L{layer}:", file=sys.stderr)
    for a in TRAITS:
        row = "  ".join(f"{after[a][b]:+.4f}" for b in TRAITS)
        print(f"  {a:8s} {row}", file=sys.stderr)

    out_paths: dict[str, str] = {}
    for name in TRAITS:
        if args.out_dir is not None:
            out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
            out_v = out_dir / f"{name}_ortho.pt"
        else:
            src = Path(src_paths[name])
            out_v = src.parent / "persona_vectors_ortho.pt"

        out_ck = dict(cks[name])
        out_ck["v"] = ortho[name]
        removed = plans[name]
        out_ck["ortho_meta"] = {
            "method": "gram_schmidt" if removed else "identity",
            "target": name,
            "removed": removed,
            "source": src_paths[name],
            "source_remove": src_paths[removed] if removed else None,
            "focus_layer": layer,
            "focus": per_layer[name][layer],
        }
        out_v.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_ck, out_v)
        out_paths[name] = str(out_v)
        print(f"Wrote {out_v}", file=sys.stderr)

    out_j = args.out_json if args.out_json.is_absolute() else root / args.out_json
    doc = {
        "method": "gram_schmidt_dnd_grid",
        "focus_layer": layer,
        "plans": plans,
        "sources": src_paths,
        "out_vectors": out_paths,
        "cosine_before": before,
        "cosine_after": after,
        "per_layer": {
            k: {
                "focus": per_layer[k][layer],
                "rows": per_layer[k],
            }
            for k in TRAITS
        },
    }
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_j}", file=sys.stderr)
    print(json.dumps({"cosine_before": before, "cosine_after": after, "out_vectors": out_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
