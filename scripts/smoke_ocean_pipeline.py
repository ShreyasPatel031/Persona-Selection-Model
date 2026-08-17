#!/usr/bin/env python3
"""End-to-end CPU smoke test of the OCEAN vector pipeline on a tiny random model.

Runs all three stages — prompt ladder, ladder vectors, validated sweep — so the
GPU run is a single command with known-good plumbing. A randomly initialised
2-layer model cannot produce a real personality signal, so this asserts only that
each stage executes, writes its artefact, and reports the fields downstream
stages consume. Numeric claims come from a real model.

    python3 scripts/smoke_ocean_pipeline.py
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

# Gemma merges the system turn into the first user turn; the tiny test checkpoints
# ship no chat template, so supply that shape explicitly.
GEMMA_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ message['content'] + '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)

TINY_MODEL = "bumblebee-testing/tiny-random-Gemma3ForCausalLM"


def patch_tokenizer_template(model_id: str) -> None:
    """Give the tiny tokenizer a chat template, cached where transformers looks."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if getattr(tok, "chat_template", None):
        return
    tok.chat_template = GEMMA_STYLE_TEMPLATE
    local = Path(tempfile.gettempdir()) / "tiny_gemma_with_template"
    tok.save_pretrained(local)
    os.environ["GEMMA_MODEL_ID"] = str(local)

    # The model weights still come from the hub id; copy the config across so the
    # patched directory is loadable as a whole.
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.from_pretrained(model_id).save_pretrained(local)
    AutoModelForCausalLM.from_pretrained(model_id).save_pretrained(local)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=TINY_MODEL)
    p.add_argument("--trait", default="conscientiousness")
    p.add_argument(
        "--items-csv",
        type=Path,
        default=REPO_ROOT / "data" / "ipip_neo_120.csv",
        help="Inventory to administer; trimmed to one trait to keep the smoke test quick.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    os.environ["PERSONA_FORCE_CPU"] = "1"

    patch_tokenizer_template(args.model_id)
    model_id = os.environ.get("GEMMA_MODEL_ID", args.model_id)

    from app.persona.intensity_ladder import (
        build_ladder_vectors,
        run_prompt_ladder,
        run_validated_sweep,
    )

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="ocean_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"smoke output: {out_dir}")
    print(f"model: {model_id} (CPU, randomly initialised)")

    failures: list[str] = []

    # Stage 1: three levels, one variant, single-trait items — enough to build a
    # centroid grid the vector stage can consume.
    ladder_json = out_dir / "prompt_ladder.json"
    centroids = out_dir / "centroids.pt"
    print("\n[1/3] prompt ladder ...", flush=True)
    run_prompt_ladder(
        ladder_json,
        centroids,
        trait=args.trait,
        model_id=model_id,
        variants=1,
        levels=(1, 5, 9),
        all_traits=False,
        items_csv=args.items_csv,
    )
    ladder = json.loads(ladder_json.read_text())
    print(
        f"      {ladder['n_items']} items, {ladder['n_administrations']} administrations, "
        f"{ladder['n_usable_administrations']} usable, "
        f"rho={ladder['spearman_level_vs_target_score']}, range={ladder['target_score_range']}"
    )
    for field in ("keying_balance", "locked_administrations", "level_mean_target_score"):
        if field not in ladder:
            failures.append(f"prompt ladder report missing {field}")
    if not centroids.is_file():
        failures.append("prompt ladder wrote no centroids")

    # Stage 2: directions per layer + geometry.
    vec_pt = out_dir / "ladder_vectors.pt"
    geom_json = out_dir / "geometry.json"
    print("[2/3] ladder vectors ...", flush=True)
    build_ladder_vectors(centroids, vec_pt, geom_json)
    geom = json.loads(geom_json.read_text())
    print(
        f"      {geom['n_layers']} layers, best layer {geom['geometry']['best_layer']}, "
        f"cos(endpoint,PC1) at best = "
        f"{geom['geometry']['per_layer'][geom['geometry']['best_layer']]['cos_endpoint_pc1']}"
    )
    if not vec_pt.is_file():
        failures.append("vector stage wrote no .pt")

    # Stage 3: the validated sweep, with one control and one probe to stay quick.
    sweep_json = out_dir / "validated_sweep.json"
    print("[3/3] validated sweep ...", flush=True)
    run_validated_sweep(
        vec_pt,
        sweep_json,
        trait=args.trait,
        which="pc1",
        magnitudes=(0.5, 1.0),
        steer_toward="auto",
        n_random_controls=1,
        model_id=model_id,
        items_csv=args.items_csv,
        probe_questions=("You have three deadlines next week. Walk me through your plan.",),
        max_new_tokens=16,
    )
    sweep = json.loads(sweep_json.read_text())
    verdict = sweep["verdict"]
    curve = sweep["trait_curve"]
    print(
        f"      steered toward {verdict['steered_toward']}, grid {sweep['magnitude_grid']}, "
        f"{curve['n_usable_rungs']}/{curve['n_rungs']} usable rungs"
    )
    print(
        f"      trait delta={verdict['trait_abs_delta']} vs control "
        f"{verdict['max_control_abs_delta']}, works={verdict['works']}"
    )

    required_verdict = (
        "steered_toward",
        "trait_abs_delta",
        "max_control_abs_delta",
        "beats_random_controls",
        "trait_coherence_ceiling",
        "works",
    )
    for field in required_verdict:
        if field not in verdict:
            failures.append(f"verdict missing {field}")
    if not sweep["control_curves"]:
        failures.append("no control curve was swept")
    if len(sweep["magnitude_grid"]) != 3:
        failures.append(f"expected zero plus two rungs, got {sweep['magnitude_grid']}")
    if sweep["magnitude_grid"][0] != 0.0:
        failures.append("grid must start at zero for a baseline")
    first_rung = curve["rows"][0]
    for field in ("lock", "usable", "argmax_scores", "ev_scores", "probes"):
        if field not in first_rung:
            failures.append(f"sweep rung missing {field}")
    if first_rung["probes"] and "coherence" not in first_rung["probes"][0]:
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
