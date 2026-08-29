#!/usr/bin/env python3
"""Upper bound: patch the activation to the prompted point, not along a direction.

Adding ``alpha * v`` reproduces one coordinate and leaves the other d-1 at their
baseline value. The fitted (probe) direction covers only 21-66% of the actual
low->high displacement, so even a perfectly dosed 1-D push lands nowhere near
the prompted state. This asks what is recoverable if we stop restricting the
intervention to one direction.

Per trait, under the persona-free prompt, at the ladder's chosen layer:

    baseline        no intervention
    prompted L9/L1  the ladder prompt itself — the target to reproduce
    full patch      add (c_level - h_baseline), all d dimensions. The ceiling on
                    what any layer-local residual edit can achieve.
    rank-k          add only that displacement's projection onto the top k PCs
                    of the ladder, for k = 1, 2, 4, 8. Says how many dimensions
                    the effect actually needs.
    probe 1x        the fitted direction at its own span (not PC1's span, the
                    mis-dosing bug this script exists to sidestep)

If the full patch reaches the prompted score, the residual mean at this layer is
sufficient and 1-D was simply too small a slice. If it does not, no vector at
this layer will work, whatever its direction.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("patch_upper_bound")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors-dir", required=True, help="Holds ladder_vectors_*.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default="")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--traits", default="")
    p.add_argument("--ranks", default="1,2,4,8")
    p.add_argument("--levels", default="9,1", help="Prompted levels to reproduce.")
    p.add_argument(
        "--position",
        default="all",
        choices=("all", "last"),
        help="Where to apply the edit. The centroid is an answer-position "
        "statistic, so 'last' is the faithful 'put the activation at that point' "
        "test; 'all' is what additive steering normally does.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.intensity_ladder import (
        _load_model,
        _option_token_ids,
        _Steering,
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
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    items = items_from_csv(Path(args.items_csv))
    model, tokenizer, dev = _load_model(args.model_id or None, None)
    option_ids = _option_token_ids(tokenizer)
    free = persona_free_system_prompt()

    from app.persona.intensity_ladder import language_model_layers

    class _LastPositionEdit:
        """Add the delta only at the final (answer) position of each sequence."""

        def __init__(self, layer_idx: int, delta: torch.Tensor) -> None:
            self.layer = language_model_layers(model)[layer_idx]
            param = next(model.parameters())
            self.delta = delta.to(device=param.device, dtype=param.dtype)
            self.handle = None

        def __enter__(self):
            def hook(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if isinstance(h, torch.Tensor) and h.dim() == 3:
                    h[:, -1, :].add_(self.delta)
                return output

            self.handle = self.layer.register_forward_hook(hook)
            return self

        def __exit__(self, *exc) -> None:
            if self.handle is not None:
                self.handle.remove()

    def administer(system: str, steer: torch.Tensor | None, layer: int) -> dict:
        if steer is None:
            responses, centroid = administer_inventory(
                model, tokenizer, dev, system, items,
                option_ids=option_ids, collect_activations=True,
            )
        else:
            ctx = (
                _LastPositionEdit(layer, steer)
                if args.position == "last"
                else _Steering(model, layer, steer, 1.0)
            )
            with ctx:
                responses, centroid = administer_inventory(
                    model, tokenizer, dev, system, items,
                    option_ids=option_ids, collect_activations=True,
                )
        return {
            "ev": score_traits_ev(responses),
            "lock": option_lock(responses),
            "centroid": centroid,
        }

    out: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "patch_upper_bound",
        "model_id": args.model_id or None,
        "n_items": len(items),
        "traits": {},
    }

    for trait in traits:
        vec_pt = vdir / f"ladder_vectors_{trait}.pt"
        if not vec_pt.is_file():
            logger.warning("[%s] no %s, skipping", trait, vec_pt.name)
            continue
        blob = torch.load(vec_pt, map_location="cpu")
        cents = blob["level_centroids"].float()  # (levels, layers, d)
        layer, layer_note = resolve_steering_layer(
            blob["geometry"], int(cents.shape[1])
        )
        logger.info("[%s] layer %s (%s)", trait, layer, layer_note)

        base = administer(free, None, layer)
        h_base = base["centroid"].float()[layer]
        rows: dict = {
            "layer": layer,
            "layer_note": layer_note,
            "baseline_ev": round(base["ev"][trait], 4),
            "baseline_locked": base["lock"]["locked"],
            "levels": {},
        }
        logger.info("[%s] baseline ev=%.4f", trait, rows["baseline_ev"])

        # PCs of the ladder at this layer, for the rank-k truncation.
        at_layer = cents[:, layer, :]
        centered = at_layer - at_layer.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)

        for level in levels:
            target = at_layer[level - 1]
            disp = target - h_base
            prompted = administer(
                ladder_system_prompt(trait, level, n_markers=3), None, layer
            )
            entry = {
                "prompted_ev": round(prompted["ev"][trait], 4),
                "prompted_locked": prompted["lock"]["locked"],
                "displacement_norm": round(float(disp.norm()), 2),
                "interventions": {},
            }
            logger.info(
                "[%s] level %s prompted ev=%.4f (|disp|=%.1f)",
                trait, level, entry["prompted_ev"], entry["displacement_norm"],
            )

            full = administer(free, disp, layer)
            entry["interventions"]["full_patch"] = {
                "ev": round(full["ev"][trait], 4),
                "locked": full["lock"]["locked"],
                "norm": round(float(disp.norm()), 2),
            }
            logger.info(
                "[%s] level %s full patch ev=%.4f locked=%s",
                trait, level, full["ev"][trait], full["lock"]["locked"],
            )

            for k in ranks:
                basis = vh[: min(k, vh.shape[0])]
                approx = basis.T @ (basis @ disp)
                res = administer(free, approx, layer)
                entry["interventions"][f"rank{k}"] = {
                    "ev": round(res["ev"][trait], 4),
                    "locked": res["lock"]["locked"],
                    "norm": round(float(approx.norm()), 2),
                    "fraction_of_displacement": round(
                        float(approx.norm() / disp.norm().clamp_min(1e-9)), 4
                    ),
                }
                logger.info(
                    "[%s] level %s rank%-2d ev=%.4f (covers %.0f%% of |disp|)",
                    trait, level, k, res["ev"][trait],
                    100 * float(approx.norm() / disp.norm().clamp_min(1e-9)),
                )

            if "v_probe" in blob:
                v = blob["v_probe"][layer].float()
                v = v / v.norm().clamp_min(1e-9)
                # Dose at the probe direction's OWN span, not PC1's.
                span = float(torch.dot(at_layer[-1] - at_layer[0], v))
                sign = 1.0 if level > (cents.shape[0] + 1) / 2 else -1.0
                steer = v * abs(span) * sign
                res = administer(free, steer, layer)
                entry["interventions"]["probe_own_span"] = {
                    "ev": round(res["ev"][trait], 4),
                    "locked": res["lock"]["locked"],
                    "norm": round(float(steer.norm()), 2),
                }
                logger.info(
                    "[%s] level %s probe@own-span ev=%.4f (mag %.1f)",
                    trait, level, res["ev"][trait], float(steer.norm()),
                )

            rows["levels"][str(level)] = entry

        out["traits"][trait] = rows
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 100)
    print("CAN A LAYER-LOCAL EDIT REPRODUCE THE PROMPT?  (ev on the target trait)")
    print("=" * 100)
    print(f"{'trait':16}{'lvl':>4}{'base':>7}{'prompted':>10}{'full':>8}"
          f"{'rank1':>8}{'rank2':>8}{'rank4':>8}{'rank8':>8}{'probe':>8}")
    for trait, r in out["traits"].items():
        for level, e in r["levels"].items():
            iv = e["interventions"]
            g = lambda k: f"{iv[k]['ev']:.3f}" if k in iv else "-"  # noqa: E731
            print(f"{trait:16}{level:>4}{r['baseline_ev']:>7.3f}{e['prompted_ev']:>10.3f}"
                  f"{g('full_patch'):>8}{g('rank1'):>8}{g('rank2'):>8}"
                  f"{g('rank4'):>8}{g('rank8'):>8}{g('probe_own_span'):>8}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
