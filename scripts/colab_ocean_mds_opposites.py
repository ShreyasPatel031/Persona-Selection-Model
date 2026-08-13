#!/usr/bin/env python3
"""OCEAN MDS opposite-pole demo: same vectors/layers as up-steering, negative α."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse statement banks + helpers from the up-demo module when present.
sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from colab_ocean_mds_demo import (  # noqa: E402
    MODEL_ID as DEFAULT_MODEL,
    N_STMT,
    OCEAN_STATEMENTS,
    PROBE_Q,
    SYSTEM,
    extract_mds_for_trait,
    generate,
    language_model_layers,
    pick_dtype,
)

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", DEFAULT_MODEL)
OUT = Path(os.environ.get("PERSONA_OUT", "/content/ocean_mds_opposites.json"))

# From successful up-run defaults
UP_CFGS = {
    "openness": {"layer": 15, "alpha": 8.0},
    "conscientiousness": {"layer": 15, "alpha": 2.0},
    "extraversion": {"layer": 15, "alpha": 2.0},
    "agreeableness": {"layer": 15, "alpha": 8.0},
    "neuroticism": {"layer": 15, "alpha": 8.0},
}


def main() -> int:
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)

    dtype = pick_dtype()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=token,
    )
    model.eval()
    layers = language_model_layers(model)
    device = next(model.parameters()).device
    print("layers", len(layers), "alloc_GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    vec_path = Path("/content/ocean_mds_vectors.pt")
    if vec_path.is_file():
        vectors = torch.load(vec_path, map_location="cpu", weights_only=False)
        print("loaded vectors", vec_path, flush=True)
    else:
        vectors = {}
        for trait in UP_CFGS:
            print(f"extract {trait}", flush=True)
            vectors[trait] = extract_mds_for_trait(
                model, tokenizer, layers, device, trait, N_STMT
            )
        torch.save(vectors, vec_path)

    baseline = generate(model, tokenizer, layers, device, PROBE_Q)
    print("\nBASELINE\n", baseline, flush=True)

    traits = {}
    for trait, cfg in UP_CFGS.items():
        layer = int(cfg["layer"])
        alpha_down = -float(cfg["alpha"])
        direction = vectors[trait][layer]
        reply = generate(
            model, tokenizer, layers, device, PROBE_Q, direction, alpha_down, layer
        )
        print(f"\n[{trait} DOWN] L{layer} α={alpha_down}\n{reply}\n", flush=True)
        traits[trait] = {
            "layer": layer,
            "alpha": alpha_down,
            "polarity": "down",
            "reply": reply,
        }

    report = {
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "question": PROBE_Q,
        "system": SYSTEM,
        "baseline": baseline,
        "traits_down": traits,
        "note": "Opposite pole via negative α on same MDS v=(μ↑-μ↓)/2 and layers as up-run.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print("wrote", OUT, flush=True)
    print("OCEAN_OPPOSITES_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
