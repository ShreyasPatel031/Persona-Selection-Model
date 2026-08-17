#!/usr/bin/env python3
"""End-to-end CPU smoke test of the OCEAN vector pipeline on a small real model.

Runs all three stages — prompt ladder, ladder vectors, validated sweep — so a
larger run is a single command with known-good plumbing. Deliberately uses a
small *instruction-tuned* model rather than a randomly initialised one: a random
model has no personality and locks onto a single option, which exercises the
screening but tells you nothing about whether the stages produce signal.

Defaults are sized for a few minutes on CPU (one trait, three ladder levels, two
magnitudes), so the numbers are indicative rather than conclusive.

    python3 scripts/smoke_ocean_pipeline.py
    python3 scripts/smoke_ocean_pipeline.py --model-id Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Ungated, small, has a chat template with a system role, and tokenizes the Likert
# digits to distinct ids — all four are required by the inventory protocol.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--trait", default="conscientiousness")
    p.add_argument("--items-csv", type=Path, default=REPO_ROOT / "data" / "ipip_neo_120.csv")
    p.add_argument("--levels", default="1,5,9")
    p.add_argument("--magnitudes", default="0.5,1.0")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ["PERSONA_FORCE_CPU"] = "1"
    os.environ["GEMMA_MODEL_ID"] = args.model_id

    from app.persona.intensity_ladder import (
        build_ladder_vectors,
        run_prompt_ladder,
        run_validated_sweep,
    )

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="ocean_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    mags = tuple(float(x) for x in args.magnitudes.split(",") if x.strip())
    print(f"smoke output: {out_dir}")
    print(f"model: {args.model_id} (CPU)")

    failures: list[str] = []

    ladder_json = out_dir / "prompt_ladder.json"
    centroids = out_dir / "centroids.pt"
    print(f"\n[1/3] prompt ladder, levels {levels} ...", flush=True)
    run_prompt_ladder(
        ladder_json,
        centroids,
        trait=args.trait,
        model_id=args.model_id,
        variants=1,
        levels=levels,
        all_traits=False,
        items_csv=args.items_csv,
    )
    ladder = json.loads(ladder_json.read_text())
    print(
        f"      {ladder['n_items']} items | usable "
        f"{ladder['n_usable_administrations']}/{ladder['n_administrations']} | "
        f"rho={ladder['spearman_level_vs_target_score']} | "
        f"level means {ladder['level_mean_target_score']}"
    )
    for field in ("keying_balance", "locked_administrations", "level_mean_target_score"):
        if field not in ladder:
            failures.append(f"prompt ladder report missing {field}")
    if not centroids.is_file():
        failures.append("prompt ladder wrote no centroids")

    vec_pt = out_dir / "ladder_vectors.pt"
    geom_json = out_dir / "geometry.json"
    print("[2/3] ladder vectors ...", flush=True)
    build_ladder_vectors(centroids, vec_pt, geom_json)
    geom = json.loads(geom_json.read_text())
    best = geom["geometry"]["best_layer"]
    print(
        f"      {geom['n_layers']} layers | best layer {best} | "
        f"cos(endpoint,PC1)={geom['geometry']['per_layer'][best]['cos_endpoint_pc1']}"
    )
    if not vec_pt.is_file():
        failures.append("vector stage wrote no .pt")

    sweep_json = out_dir / "validated_sweep.json"
    print(f"[3/3] validated sweep, magnitudes {mags} ...", flush=True)
    run_validated_sweep(
        vec_pt,
        sweep_json,
        trait=args.trait,
        which="pc1",
        magnitudes=mags,
        steer_toward="auto",
        n_random_controls=1,
        model_id=args.model_id,
        items_csv=args.items_csv,
        probe_questions=("You have three deadlines next week. Walk me through your plan.",),
        max_new_tokens=args.max_new_tokens,
    )
    sweep = json.loads(sweep_json.read_text())
    verdict, curve = sweep["verdict"], sweep["trait_curve"]
    print(
        f"      toward {verdict['steered_toward']} | grid {sweep['magnitude_grid']} | "
        f"usable {curve['n_usable_rungs']}/{curve['n_rungs']} | rho="
        f"{curve['spearman_absalpha_vs_target_ev']}"
    )
    print(
        f"      trait delta={verdict['trait_abs_delta']} vs control "
        f"{verdict['max_control_abs_delta']} | ceiling {verdict['trait_coherence_ceiling']} | "
        f"works={verdict['works']}"
    )

    for field in (
        "steered_toward",
        "trait_abs_delta",
        "max_control_abs_delta",
        "beats_random_controls",
        "trait_coherence_ceiling",
        "works",
    ):
        if field not in verdict:
            failures.append(f"verdict missing {field}")
    if not sweep["control_curves"]:
        failures.append("no control curve was swept")
    if sweep["magnitude_grid"][0] != 0.0:
        failures.append("grid must start at zero for a baseline")
    if len(sweep["magnitude_grid"]) != len(mags) + 1:
        failures.append(f"expected zero plus {len(mags)} rungs, got {sweep['magnitude_grid']}")
    first = curve["rows"][0]
    for field in ("lock", "usable", "argmax_scores", "ev_scores", "probes"):
        if field not in first:
            failures.append(f"sweep rung missing {field}")
    if first["probes"] and "coherence" not in first["probes"][0]:
        failures.append("probe rows carry no coherence metrics")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all three stages ran and produced the expected fields")
    print(f"artefacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
