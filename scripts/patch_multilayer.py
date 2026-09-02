#!/usr/bin/env python3
"""Multi-layer residual patch: is the write-failure a single-site limit?

``patch_upper_bound.py`` showed that editing one layer recovers ~0% of the
prompted inventory separation. A prompt is present at every layer of every
forward pass; this applies the prompted displacement at many layers at once.

Bands tested (under a persona-free prompt, both poles):

    single          the ladder's chosen best layer (reproduces the prior ceiling)
    mid_band        layers in [0.3, 0.8] of depth
    every_2         every second layer in mid_band
    all             every layer

If a multi-layer patch recovers the prompted score, residual steering is still
viable and the one-layer restriction (CAA / SAE) is the bottleneck. If it does
not, personality is not an additive residual property at all in this model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("patch_multilayer")


class _MultiLayerEdit:
    """Add a per-layer delta at every (or last) position of each hooked layer."""

    def __init__(
        self,
        model,
        deltas: dict[int, torch.Tensor],
        *,
        position: str = "all",
    ) -> None:
        from app.persona.intensity_ladder import language_model_layers

        self.layers = language_model_layers(model)
        self.position = position
        self.handles = []
        param = next(model.parameters())
        self.deltas = {
            int(i): d.to(device=param.device, dtype=param.dtype) for i, d in deltas.items()
        }

    def __enter__(self):
        for idx, delta in self.deltas.items():
            layer = self.layers[idx]

            def make_hook(d: torch.Tensor):
                def hook(_m, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if isinstance(h, torch.Tensor) and h.dim() == 3:
                        if self.position == "last":
                            h[:, -1, :].add_(d)
                        else:
                            h.add_(d)
                    return output

                return hook

            self.handles.append(layer.register_forward_hook(make_hook(delta)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _band_layers(n_layers: int, name: str, best: int) -> list[int]:
    lo = int(0.3 * n_layers)
    hi = int(0.8 * n_layers)
    mid = list(range(lo, max(lo + 1, hi)))
    if name == "single":
        return [best]
    if name == "mid_band":
        return mid
    if name == "every_2":
        return mid[::2]
    if name == "all":
        return list(range(n_layers))
    raise ValueError(f"unknown band {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default="")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--traits", default="")
    p.add_argument("--levels", default="9,1")
    p.add_argument(
        "--bands",
        default="single,mid_band,every_2,all",
        help="Comma-separated band names.",
    )
    p.add_argument("--position", default="all", choices=("all", "last"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.intensity_ladder import (
        _load_model,
        _option_token_ids,
        administer_inventory,
        ladder_system_prompt,
        resolve_steering_layer,
    )
    from app.persona.intensity_prompts import persona_free_system_prompt
    from app.persona.inventory_ipip import (
        TRAITS,
        items_from_csv,
        option_lock,
        score_traits_ev,
    )

    vdir = Path(args.vectors_dir)
    traits = [t.strip() for t in args.traits.split(",") if t.strip()] or list(TRAITS)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    items = items_from_csv(Path(args.items_csv))
    model, tokenizer, device = _load_model(args.model_id or None, None)
    option_ids = _option_token_ids(tokenizer)
    free = persona_free_system_prompt()

    def score_under(system: str, deltas: dict[int, torch.Tensor] | None) -> dict:
        if not deltas:
            responses, centroid = administer_inventory(
                model, tokenizer, device, system, items,
                option_ids=option_ids, collect_activations=True,
            )
        else:
            with _MultiLayerEdit(model, deltas, position=args.position):
                responses, centroid = administer_inventory(
                    model, tokenizer, device, system, items,
                    option_ids=option_ids, collect_activations=True,
                )
        return {
            "ev": score_traits_ev(responses),
            "lock": option_lock(responses),
            "centroid": centroid,
        }

    out: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "patch_multilayer",
        "position": args.position,
        "bands": bands,
        "traits": {},
    }

    for trait in traits:
        vec_pt = vdir / f"ladder_vectors_{trait}.pt"
        if not vec_pt.is_file():
            logger.warning("[%s] missing %s", trait, vec_pt.name)
            continue
        blob = torch.load(vec_pt, map_location="cpu")
        cents = blob["level_centroids"].float()  # (levels, layers, d)
        n_layers = int(cents.shape[1])
        best, note = resolve_steering_layer(blob["geometry"], n_layers)
        logger.info("[%s] best layer %s (%s)", trait, best, note)

        base = score_under(free, None)
        h_base = base["centroid"].float()  # (layers, d)
        row = {
            "best_layer": best,
            "layer_note": note,
            "baseline_ev": round(base["ev"][trait], 4),
            "levels": {},
        }

        for level in levels:
            target = cents[level - 1]  # (layers, d)
            disp = {li: (target[li] - h_base[li]) for li in range(n_layers)}
            prompted = score_under(
                ladder_system_prompt(trait, level, n_markers=3), None
            )
            entry = {
                "prompted_ev": round(prompted["ev"][trait], 4),
                "prompted_locked": prompted["lock"]["locked"],
                "bands": {},
            }
            logger.info(
                "[%s] L%s prompted=%.3f baseline=%.3f",
                trait, level, entry["prompted_ev"], row["baseline_ev"],
            )
            for band in bands:
                idxs = _band_layers(n_layers, band, best)
                deltas = {i: disp[i] for i in idxs}
                res = score_under(free, deltas)
                mean_norm = sum(float(disp[i].norm()) for i in idxs) / max(1, len(idxs))
                entry["bands"][band] = {
                    "layers": idxs,
                    "n_layers": len(idxs),
                    "ev": round(res["ev"][trait], 4),
                    "locked": res["lock"]["locked"],
                    "mean_layer_disp_norm": round(mean_norm, 2),
                }
                logger.info(
                    "[%s] L%s band=%-10s n=%2d ev=%.3f",
                    trait, level, band, len(idxs), res["ev"][trait],
                )
            row["levels"][str(level)] = entry

        out["traits"][trait] = row
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # Separation table.
    print("\n" + "=" * 100)
    print("MULTI-LAYER PATCH: prompted hi−lo vs patched hi−lo (fraction recovered)")
    print("=" * 100)
    header = f"{'trait':17}{'prompted':>10}" + "".join(f"{b:>12}" for b in bands)
    print(header)
    for trait, r in out["traits"].items():
        L = r["levels"]
        if "9" not in L or "1" not in L:
            continue
        pd = L["9"]["prompted_ev"] - L["1"]["prompted_ev"]
        cells = [f"{pd:>+10.3f}"]
        for band in bands:
            fd = (
                L["9"]["bands"][band]["ev"] - L["1"]["bands"][band]["ev"]
            )
            frac = (100 * fd / pd) if abs(pd) > 1e-9 else 0.0
            cells.append(f"{fd:>+5.2f}/{frac:>3.0f}%")
        print(f"{trait:17}" + "".join(f"{c:>12}" for c in cells))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
