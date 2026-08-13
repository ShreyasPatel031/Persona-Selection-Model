#!/usr/bin/env python3
"""Finish MPI-120 retune after timeout: tune N quickly, then full baseline+5ups.

Uses configs already selected from the interrupted sweep for O/C/E/A.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from colab_ocean_mds_demo import (  # noqa: E402
    MODEL_ID as DEFAULT_MODEL,
    language_model_layers,
    pick_dtype,
)
from colab_ocean_mpi120_eval import load_mpi  # noqa: E402
from colab_ocean_mpi120_retune import TRAITS, run_inventory  # noqa: E402

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", DEFAULT_MODEL)
OUT = Path(os.environ.get("PERSONA_OUT", "/content/ocean_mpi120_retune.json"))
MPI_CSV = Path(os.environ.get("MPI_CSV", "/content/mpi_120.csv"))

# From interrupted sweep (40-item tune subset)
PRESET = {
    "openness": {"layer": 15, "alpha": 1.0, "ocean": "O"},
    "conscientiousness": {"layer": 15, "alpha": 2.0, "ocean": "C"},
    "extraversion": {"layer": 15, "alpha": 2.0, "ocean": "E"},
    "agreeableness": {"layer": 15, "alpha": 2.0, "ocean": "A"},
}


def main() -> int:
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)

    items = load_mpi(MPI_CSV)
    tune_items = []
    per = {k: 0 for k in "OCEAN"}
    for row in items:
        d = row["label_ocean"]
        if per[d] < 8:
            tune_items.append(row)
            per[d] += 1

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
    print("alloc_GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    vectors = torch.load("/content/ocean_mds_vectors.pt", map_location="cpu", weights_only=False)
    print("loaded vectors", flush=True)

    # Quick N tune on L15 only
    print("=== tune neuroticism L15 ===", flush=True)
    base_t = run_inventory(model, tokenizer, layers, device, tune_items)
    n_base = base_t["means"]["N"]
    best_n = None
    tune_log = []
    for alpha in (0.5, 1.0, 2.0, 4.0, 8.0):
        print(f"  N L15 α={alpha}", flush=True)
        res = run_inventory(
            model, tokenizer, layers, device, tune_items, vectors["neuroticism"][15], alpha, 15
        )
        delta = res["means"]["N"] - n_base
        entry = {
            "alpha": alpha,
            "means": res["means"],
            "delta": round(delta, 3),
            "collapse": res["collapse"],
            "letter_hist": res["letter_hist"],
        }
        tune_log.append(entry)
        print("   ", entry, flush=True)
        if not res["collapse"]:
            key = (delta, res["means"]["N"], -abs(alpha - 2.0))
            if best_n is None or key > best_n[0]:
                best_n = (key, alpha)

    n_alpha = best_n[1] if best_n else 2.0
    best_cfg = dict(PRESET)
    best_cfg["neuroticism"] = {
        "layer": 15,
        "alpha": n_alpha,
        "ocean": "N",
        "delta_target": next(e["delta"] for e in tune_log if e["alpha"] == n_alpha),
        "collapse": next(e["collapse"] for e in tune_log if e["alpha"] == n_alpha),
    }
    print("best_cfg", best_cfg, flush=True)

    print("\n=== FULL MPI-120 ===", flush=True)
    baseline = run_inventory(model, tokenizer, layers, device, items)
    print("baseline", baseline["means"], baseline["letter_hist"], flush=True)

    final_means = {"baseline": baseline["means"]}
    final_meta = {
        "baseline": {"means": baseline["means"], "letter_hist": baseline["letter_hist"]}
    }
    target_lift = {}
    final_items = {"baseline": baseline["items"]}

    for trait, cfg in best_cfg.items():
        print(f"final {trait} L{cfg['layer']} α={cfg['alpha']}", flush=True)
        res = run_inventory(
            model,
            tokenizer,
            layers,
            device,
            items,
            vectors[trait][int(cfg["layer"])],
            float(cfg["alpha"]),
            int(cfg["layer"]),
        )
        key = f"{trait}_up"
        final_means[key] = res["means"]
        final_meta[key] = {
            "layer": cfg["layer"],
            "alpha": cfg["alpha"],
            "means": res["means"],
            "letter_hist": res["letter_hist"],
            "collapse": res["collapse"],
        }
        final_items[key] = res["items"]
        dim = TRAITS[trait]
        target_lift[trait] = {
            "target_dim": dim,
            "baseline": baseline["means"][dim],
            "steered": res["means"][dim],
            "delta": round(res["means"][dim] - baseline["means"][dim], 3),
            "all_deltas": {
                k: round(res["means"][k] - baseline["means"][k], 3) for k in "OCEAN"
            },
            "layer": cfg["layer"],
            "alpha": cfg["alpha"],
        }
        print("  means", res["means"], "lift", target_lift[trait], flush=True)

    report = {
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "inventory": "MPI-120",
        "note": "Finished after timeout; O/C/E/A from prior sweep, N retuned on L15.",
        "best_cfg": best_cfg,
        "neuroticism_tune_log": tune_log,
        "baseline_means": baseline["means"],
        "final_means": final_means,
        "target_lift": target_lift,
        "final": final_meta,
        "final_items": final_items,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(final_means, indent=2), flush=True)
    print(json.dumps(target_lift, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    n_pos = sum(1 for v in target_lift.values() if v["delta"] > 0)
    print("MPI120_RETUNE_OK" if n_pos >= 3 else "MPI120_RETUNE_PARTIAL", f"pos={n_pos}/5", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
