#!/usr/bin/env python3
"""E1 — is the difference the VECTOR, holding model / instrument / scope fixed?

E0 showed that narrowing the inject span destroys direction-controlled inventory
movement *inside our pipeline*. That does not show injection scope explains Blas
et al.'s inventory null, because at least five things still differ: vector
construction, the data the vector was fit on, the model, the instrument, and the
dose calibration. This script isolates the first two.

Their vector is a two-arm contrast: ``mean(high statements) - mean(low
statements)``, unit-normalised (their released tensors have norm 1.0), or a
logistic-regression weight vector. Nothing in either construction constrains the
*middle* of the trait scale to be ordered along the direction. Ours is PC1 across
nine prompted intensity levels, where gradedness is fit in. A two-arm contrast can
separate the endpoints perfectly while intermediate levels project
non-monotonically — and a graded inventory measures exactly the middle.

Stages
------
``ladder``    Administer our nine-level intensity ladder on Llama-3.1-8B and save
              level centroids + our PC1. (GPU, the expensive part.)

``geometry``  The decisive cheap test, no extra forwards: project our nine level
              centroids onto THEIR released vector and ask whether their direction
              orders the ladder (Spearman, monotone fraction, span) as well as ours
              does. Also cosine between the two directions per layer. If their
              vector does not order the ladder, the vector is a real difference; if
              it does, the vector is not the story and scope/dose is.

``steer``     Head-to-head inventory sweep: their vector vs our PC1, same model,
              same layer, same instrument, same ``full`` scope, same argmax
              readout, each dosed in units of its OWN ladder span so neither is
              mis-dosed.

    python3 scripts/e1_vector_headtohead.py ladder --out-dir results/e1_vector
    python3 scripts/e1_vector_headtohead.py geometry --out-dir results/e1_vector
    python3 scripts/e1_vector_headtohead.py steer --out-dir results/e1_vector
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

logger = logging.getLogger("e1_vector")

DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
THEIR_REPO = Path("/tmp/psych-steer/replication")
TRAIT = "conscientiousness"

# Their released variants, per replication/injection_utils.get_vector_path.
THEIR_VARIANTS = (
    ("meandiff", "statement"),
    ("meandiff", "binary_choice"),
    ("l2_zero_intercept", "statement"),
)


def their_vector_path(variant: str, mode: str, layer: int) -> Path:
    base = THEIR_REPO / "vectors" / "Llama-3.1-8B-Instruct" / TRAIT / variant / mode
    exact = base / f"layer_{layer}.pt"
    if exact.is_file():
        return exact
    for pat in (f"layer_{layer}_C_*.pt", f"layer_{layer}_*.pt"):
        for p in sorted(base.glob(pat)):
            if not p.name.endswith("_raw.pt"):
                return p
    raise FileNotFoundError(f"no vector for layer {layer} in {base}")


def load_their_stack(variant: str, mode: str, n_layers: int, dim: int) -> tuple[torch.Tensor, list[int]]:
    """Stack their per-layer vectors into (n_layers, d), zero where absent."""
    stack = torch.zeros(n_layers, dim, dtype=torch.float32)
    present: list[int] = []
    for li in range(n_layers):
        try:
            v = torch.load(their_vector_path(variant, mode, li), map_location="cpu")
        except FileNotFoundError:
            continue
        v = v.detach().float().reshape(-1)
        if v.numel() != dim:
            continue
        stack[li] = v
        present.append(li)
    return stack, present


def cmd_ladder(args: argparse.Namespace) -> int:
    from app.persona.intensity_ladder import build_ladder_vectors, run_prompt_ladder

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ladder_json = out / f"prompt_ladder_{TRAIT}.json"
    centroids = out / f"centroids_{TRAIT}.pt"
    run_prompt_ladder(
        ladder_json,
        centroids,
        trait=TRAIT,
        model_id=args.model_id,
        variants=args.variants,
        all_traits=True,
        items_csv=Path(args.items_csv),
    )
    build_ladder_vectors(centroids, out / f"ladder_vectors_{TRAIT}.pt", out / f"ladder_geometry_{TRAIT}.json")
    print(f"wrote {centroids} and ladder_vectors_{TRAIT}.pt")
    return 0


def cmd_geometry(args: argparse.Namespace) -> int:
    """Does their direction order our nine-level ladder as well as ours does?"""
    from app.persona.intensity_ladder import monotone_fraction, spearman_rho

    out = Path(args.out_dir)
    blob = torch.load(out / f"ladder_vectors_{TRAIT}.pt", map_location="cpu")
    centroids: torch.Tensor = blob["level_centroids"]  # (n_levels, n_layers, d)
    ours: torch.Tensor = blob["v_pc1"]
    n_levels, n_layers, dim = centroids.shape
    levels = [float(i + 1) for i in range(n_levels)]

    def ordering(unit: torch.Tensor, layer: int) -> dict:
        u = unit.float()
        nrm = float(u.norm())
        if nrm < 1e-9:
            return {"available": False}
        u = u / nrm
        proj = [float(torch.dot(c.float(), u)) for c in centroids[:, layer, :]]
        rho = spearman_rho(levels, proj)
        mono = monotone_fraction(proj)
        return {
            "available": True,
            "spearman_level_vs_projection": None if rho is None else round(rho, 4),
            "monotone_fraction": None if mono is None else round(mono, 4),
            "span": round(abs(proj[-1] - proj[0]), 4),
            "projections": [round(p, 3) for p in proj],
        }

    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "e1_vector_geometry",
        "question": (
            "Their vector is a two-arm contrast (or LR weights); ours is PC1 over nine "
            "prompted intensity levels. Does their direction order the nine levels? If "
            "not, the vector is a genuine difference and not merely injection scope."
        ),
        "model_id": blob.get("model_id"),
        "trait": TRAIT,
        "n_levels": n_levels,
        "n_layers": n_layers,
        "variants": {},
    }

    for variant, mode in THEIR_VARIANTS:
        theirs, present = load_their_stack(variant, mode, n_layers, dim)
        if not present:
            report["variants"][f"{variant}/{mode}"] = {"error": "no vectors found"}
            continue
        per_layer = []
        for li in present:
            ours_l = ordering(ours[li], li)
            theirs_l = ordering(theirs[li], li)
            uo = ours[li].float()
            ut = theirs[li].float()
            denom = float(uo.norm()) * float(ut.norm())
            cos = float(torch.dot(uo, ut)) / denom if denom > 1e-9 else None
            per_layer.append(
                {
                    "layer": li,
                    "cos_ours_theirs": None if cos is None else round(cos, 4),
                    "ours": ours_l,
                    "theirs": theirs_l,
                }
            )

        def best(key: str, which: str) -> dict | None:
            scored = [
                r
                for r in per_layer
                if r[which].get("available") and r[which].get(key) is not None
            ]
            return max(scored, key=lambda r: abs(r[which][key])) if scored else None

        bt = best("spearman_level_vs_projection", "theirs")
        bo = best("spearman_level_vs_projection", "ours")
        cosines = [abs(r["cos_ours_theirs"]) for r in per_layer if r["cos_ours_theirs"] is not None]
        report["variants"][f"{variant}/{mode}"] = {
            "n_layers_present": len(present),
            "max_abs_cos_ours_theirs": round(max(cosines), 4) if cosines else None,
            "mean_abs_cos_ours_theirs": round(sum(cosines) / len(cosines), 4) if cosines else None,
            "best_layer_theirs": None if bt is None else bt["layer"],
            "best_rho_theirs": None if bt is None else bt["theirs"]["spearman_level_vs_projection"],
            "best_mono_theirs": None if bt is None else bt["theirs"]["monotone_fraction"],
            "best_layer_ours": None if bo is None else bo["layer"],
            "best_rho_ours": None if bo is None else bo["ours"]["spearman_level_vs_projection"],
            "best_mono_ours": None if bo is None else bo["ours"]["monotone_fraction"],
            "per_layer": per_layer,
        }

    path = out / "vector_geometry.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("E1 geometry: does their direction order our nine-level ladder?\n")
    hdr = f"{'their variant':<28}{'|cos| max':>10}{'ρ theirs':>10}{'mono':>7}{'ρ ours':>9}{'mono':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, v in report["variants"].items():
        if "error" in v:
            print(f"{name:<28} {v['error']}")
            continue
        print(
            f"{name:<28}{str(v['max_abs_cos_ours_theirs']):>10}"
            f"{str(v['best_rho_theirs']):>10}{str(v['best_mono_theirs']):>7}"
            f"{str(v['best_rho_ours']):>9}{str(v['best_mono_ours']):>7}"
        )
    print(f"\nWrote {path}")
    return 0


def cmd_steer(args: argparse.Namespace) -> int:
    """Head-to-head inventory sweep: their vector vs ours, everything else fixed."""
    from app.persona.intensity_ladder import (
        direction_span_magnitude,
        run_validated_sweep,
    )

    out = Path(args.out_dir)
    ours_pt = out / f"ladder_vectors_{TRAIT}.pt"
    blob = torch.load(ours_pt, map_location="cpu")
    centroids: torch.Tensor = blob["level_centroids"]
    n_layers, dim = int(centroids.shape[1]), int(centroids.shape[2])
    layer = int(args.layer)

    # Three arms decompose estimator from data:
    #   ours_pc1      PCA over nine levels,  our ladder data
    #   ours_endpoint two-arm mean-difference, our ladder data   <- their estimator
    #   theirs        two-arm mean-difference, their statement corpus
    # ours_pc1 vs ours_endpoint isolates the estimator; ours_endpoint vs theirs
    # isolates the data the contrast was taken over.
    arms: list[tuple[str, torch.Tensor]] = [
        ("ours_pc1", blob["v_pc1"][layer]),
        ("ours_endpoint", blob["v_endpoint"][layer]),
    ]
    steer_variants = THEIR_VARIANTS if getattr(args, "all_variants", False) else (("meandiff", "statement"),)
    for variant, mode in steer_variants:
        theirs, present = load_their_stack(variant, mode, n_layers, dim)
        if layer in present:
            arms.append((f"theirs_{variant}_{mode}", theirs[layer]))

    ours_span = direction_span_magnitude(centroids, layer, blob["v_pc1"][layer])
    rows: list[dict] = []
    for name, direction in arms:
        own_span = direction_span_magnitude(centroids, layer, direction)
        # Matched-L2 dosing: both unit directions get the same residual magnitudes,
        # keyed to OUR ladder span. Own-span dosing of their unit vector is ~0.01
        # residual units (they are nearly orthogonal to the ladder) and would be a
        # trivial null.
        if ours_span <= 0:
            rows.append({"arm": name, "error": "zero ladder span on our PC1"})
            continue
        mags = [ours_span * m for m in (0.25, 0.5, 1.0, 1.5, 2.0)]
        # run_validated_sweep reads a vectors blob, so hand it this arm as v_pc1.
        arm_blob = dict(blob)
        stack = blob["v_pc1"].clone()
        stack[layer] = direction.to(stack.dtype)
        arm_blob["v_pc1"] = stack
        arm_pt = out / f"_arm_{name}.pt"
        torch.save(arm_blob, arm_pt)

        for pole in ("high", "low"):
            out_json = out / f"sweep_{TRAIT}_{name}_{pole}.json"
            logger.info("%s %s own_span=%.4f dose_span=%.1f grid=%s", name, pole, own_span, ours_span, [round(m) for m in mags])
            try:
                run_validated_sweep(
                    arm_pt,
                    out_json,
                    trait=TRAIT,
                    which="pc1",
                    layer_idx=layer,
                    magnitudes=mags,
                    auto_calibrate=False,
                    steer_toward=pole,
                    n_random_controls=args.random_controls,
                    alpha_units="raw",
                    model_id=args.model_id,
                    items_csv=Path(args.items_csv),
                    probe_questions=[],
                    baseline="persona_free",
                    injection_scope=args.scope,
                )
                d = json.loads(out_json.read_text())
                rows.append(
                    {
                        "arm": name,
                        "pole": pole,
                        "ladder_span_this_direction": round(own_span, 4),
                        "dose_span_ours_pc1": round(ours_span, 2),
                        "works": d["verdict"]["works"],
                        "trait_abs_delta": d["verdict"]["trait_abs_delta"],
                        "max_control_abs_delta": d["verdict"]["max_control_abs_delta"],
                        "report": str(out_json),
                    }
                )
            except Exception as exc:  # keep going; one arm failing is informative
                logger.exception("%s %s failed", name, pole)
                rows.append({"arm": name, "pole": pole, "error": str(exc)})

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "e1_vector_headtohead",
        "model_id": args.model_id,
        "layer": layer,
        "scope": args.scope,
        "table": rows,
    }
    path = out / "headtohead_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ladder", cmd_ladder), ("geometry", cmd_geometry), ("steer", cmd_steer)):
        sp = sub.add_parser(name)
        sp.add_argument("--out-dir", default="results/e1_vector")
        sp.add_argument("--model-id", default=DEFAULT_MODEL)
        sp.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
        sp.add_argument("--variants", type=int, default=2)
        sp.add_argument("--layer", type=int, default=14)
        sp.add_argument("--scope", default="full")
        sp.add_argument("--random-controls", type=int, default=2)
        sp.add_argument(
            "--all-variants",
            action="store_true",
            help="Steer all three of their vector families (slow). Default: meandiff/statement only.",
        )
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
