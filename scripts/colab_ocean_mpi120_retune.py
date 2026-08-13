#!/usr/bin/env python3
"""Retune MDS α on MPI-120 for each OCEAN-up trait, then final baseline+5ups eval.

Search: layer 15 first, α ∈ {0.5,1,2,4,8}; reject collapsed answer distributions;
pick α maximizing target-dim score (tie-break: larger target lift, less collapse).
If L15 yields no non-collapsed lift, try mid-layers {12,15,18,21}.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from colab_ocean_mds_demo import (  # noqa: E402
    MODEL_ID as DEFAULT_MODEL,
    N_STMT,
    extract_mds_for_trait,
    language_model_layers,
    make_hook,
    pick_dtype,
)
from colab_ocean_mpi120_eval import (  # noqa: E402
    LETTER_TO_RAW,
    choose_letter,
    item_prompt,
    load_mpi,
    ocean_means,
    score_item,
)

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", DEFAULT_MODEL)
OUT = Path(os.environ.get("PERSONA_OUT", "/content/ocean_mpi120_retune.json"))
MPI_CSV = Path(os.environ.get("MPI_CSV", "/content/mpi_120.csv"))

TRAITS = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]
LAYERS_TRY = [15, 12, 18, 21]


def run_inventory(model, tokenizer, layers, device, items, direction=None, alpha=0.0, layer=None):
    item_scores = []
    letters = []
    for i, row in enumerate(items):
        letter = choose_letter(
            model, tokenizer, layers, device, item_prompt(row["text"]), direction, alpha, layer
        )
        if letter not in LETTER_TO_RAW:
            letter = "C"
        item_scores.append(
            {
                "i": i,
                "text": row["text"],
                "label_ocean": row["label_ocean"],
                "key": int(row["key"]),
                "letter": letter,
                "score": score_item(letter, int(row["key"])),
            }
        )
        letters.append(letter)
        if (i + 1) % 40 == 0:
            print(f"    items {i+1}/{len(items)}", flush=True)
    means = ocean_means(item_scores)
    hist = dict(Counter(letters))
    n = len(letters)
    collapse = (hist.get("E", 0) + hist.get("D", 0)) / n >= 0.55 or hist.get("A", 0) / n >= 0.92
    return {
        "means": means,
        "letter_hist": hist,
        "collapse": collapse,
        "items": item_scores,
    }


def main() -> int:
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)

    items = load_mpi(MPI_CSV)
    # Stratified 40-item subset for α search (8 per OCEAN dim); full 120 for finals.
    tune_items = []
    per = {k: 0 for k in "OCEAN"}
    for row in items:
        d = row["label_ocean"]
        if per[d] < 8:
            tune_items.append(row)
            per[d] += 1
    print("tune_items", len(tune_items), "per", per, "full", len(items), flush=True)
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
    n_layers = len(layers)
    print("layers", n_layers, "alloc_GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    vec_path = Path("/content/ocean_mds_vectors.pt")
    if vec_path.is_file():
        vectors = torch.load(vec_path, map_location="cpu", weights_only=False)
        print("loaded", vec_path, flush=True)
    else:
        vectors = {}
        for trait in TRAITS:
            print("extract", trait, flush=True)
            vectors[trait] = extract_mds_for_trait(
                model, tokenizer, layers, device, trait, N_STMT
            )
        torch.save(vectors, vec_path)

    print("\n=== baseline (tune subset) ===", flush=True)
    baseline_tune = run_inventory(model, tokenizer, layers, device, tune_items)
    print(
        "baseline_tune means",
        baseline_tune["means"],
        "hist",
        baseline_tune["letter_hist"],
        flush=True,
    )

    tune_log = []
    best_cfg = {}

    for trait, dim in TRAITS.items():
        print(f"\n===== tune {trait} (target {dim}) =====", flush=True)
        candidates = []
        for layer in LAYERS_TRY:
            if layer >= n_layers:
                continue
            # After a good non-collapsed find on L15, still finish L15 alphas then stop expanding layers
            for alpha in ALPHAS:
                print(f"  try L{layer} α={alpha}", flush=True)
                direction = vectors[trait][layer]
                res = run_inventory(
                    model, tokenizer, layers, device, tune_items, direction, alpha, layer
                )
                target = res["means"][dim]
                delta = target - baseline_tune["means"][dim]
                entry = {
                    "trait": trait,
                    "layer": layer,
                    "alpha": alpha,
                    "means": res["means"],
                    "delta_target": round(delta, 3),
                    "collapse": res["collapse"],
                    "letter_hist": res["letter_hist"],
                }
                tune_log.append(entry)
                print(
                    f"    target={target:.3f} Δ={delta:+.3f} collapse={res['collapse']} hist={res['letter_hist']}",
                    flush=True,
                )
                if not res["collapse"]:
                    candidates.append((delta, target, -abs(alpha - 2.0), entry))
            # If L15 already has a positive non-collapsed lift, skip other layers for speed
            if layer == 15 and any(c[0] > 0 for c in candidates):
                print(f"  L15 has positive lift for {trait}; skip further layers", flush=True)
                break

        if candidates:
            # Sort only on numeric keys (dicts are not orderable).
            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            best = candidates[0][3]
        else:
            # fallback: best target among all even if collapsed / negative
            pool = [e for e in tune_log if e["trait"] == trait]
            best = max(
                pool,
                key=lambda e: (e["delta_target"], -int(e["collapse"]), e["means"][dim]),
            )
            best = dict(best)
            best["fallback"] = True
        # If best lift is ~0 (ceiling), keep a gentle non-collapsed α for final full-120 check
        if float(best["delta_target"]) <= 0 and candidates:
            gentle = [
                c[3]
                for c in candidates
                if c[3]["alpha"] in (0.5, 1.0, 2.0) and not c[3]["collapse"]
            ]
            if gentle:
                best = max(gentle, key=lambda e: (e["delta_target"], e["means"][dim]))
        best_cfg[trait] = {
            "layer": best["layer"],
            "alpha": best["alpha"],
            "ocean": dim,
            "delta_target": best["delta_target"],
            "collapse": best["collapse"],
            "fallback": best.get("fallback", False),
        }
        print("BEST", trait, best_cfg[trait], flush=True)

    # Final eval with tuned configs on FULL MPI-120
    print("\n===== FINAL full MPI-120 baseline + 5 ups =====", flush=True)
    baseline = run_inventory(model, tokenizer, layers, device, items)
    print("baseline_full", baseline["means"], baseline["letter_hist"], flush=True)
    final = {
        "baseline": {
            "means": baseline["means"],
            "letter_hist": baseline["letter_hist"],
            "items": baseline["items"],
        }
    }
    target_lift = {}
    for trait, cfg in best_cfg.items():
        print(f"final {trait} L{cfg['layer']} α={cfg['alpha']}", flush=True)
        res = run_inventory(
            model,
            tokenizer,
            layers,
            device,
            items,
            vectors[trait][cfg["layer"]],
            float(cfg["alpha"]),
            int(cfg["layer"]),
        )
        final[f"{trait}_up"] = {
            "layer": cfg["layer"],
            "alpha": cfg["alpha"],
            "means": res["means"],
            "letter_hist": res["letter_hist"],
            "collapse": res["collapse"],
            "items": res["items"],
        }
        dim = cfg["ocean"]
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
        "tune_subset_n": len(tune_items),
        "alphas_swept": ALPHAS,
        "layers_policy": LAYERS_TRY,
        "baseline_tune_means": baseline_tune["means"],
        "baseline_means": baseline["means"],
        "best_cfg": best_cfg,
        "tune_log": [{k: e[k] for k in e if k != "items"} for e in tune_log],
        "final_means": {k: v["means"] for k, v in final.items()},
        "target_lift": target_lift,
        "final": {
            k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in final.items()
        },
        "final_items": {k: v.get("items") for k, v in final.items()},
    }
    OUT.write_text(json.dumps(report, indent=2))
    print("\nSUMMARY means", json.dumps(report["final_means"], indent=2), flush=True)
    print("target_lift", json.dumps(target_lift, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    n_pos = sum(1 for v in target_lift.values() if v["delta"] > 0)
    print("MPI120_RETUNE_OK" if n_pos >= 4 else "MPI120_RETUNE_PARTIAL", f"pos={n_pos}/5", flush=True)
    return 0 if n_pos >= 3 else 2


if __name__ == "__main__":
    sys.exit(main())
