#!/usr/bin/env python3
"""MPI-120 questionnaire eval: baseline + 5 OCEAN ups under MDS steering.

Uses public MPI-120 items (jianggy/MPI = IPIP-NEO-120 second-person stems).
Answers chosen by argmax over {A,B,C,D,E} logits (1-step, greedy, constrained).
"""
from __future__ import annotations

import csv
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
    N_STMT,
    SYSTEM,
    extract_mds_for_trait,
    language_model_layers,
    make_hook,
    pick_dtype,
)

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", DEFAULT_MODEL)
OUT = Path(os.environ.get("PERSONA_OUT", "/content/ocean_mpi120_eval.json"))
MPI_CSV = Path(os.environ.get("MPI_CSV", "/content/mpi_120.csv"))

# From successful Saturday demo (up polarity)
UP_CFGS = {
    "openness": {"layer": 15, "alpha": 8.0, "ocean": "O"},
    "conscientiousness": {"layer": 15, "alpha": 2.0, "ocean": "C"},
    "extraversion": {"layer": 15, "alpha": 2.0, "ocean": "E"},
    "agreeableness": {"layer": 15, "alpha": 8.0, "ocean": "A"},
    "neuroticism": {"layer": 15, "alpha": 8.0, "ocean": "N"},
}

LETTER_TO_RAW = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def load_mpi(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 120, len(rows)
    return rows


def item_prompt(text: str) -> str:
    # Paper / P2 style: "You {item}." with first letter lowercased.
    item = text[:1].lower() + text[1:] if text else text
    return (
        f'Given a statement: "You {item}." Which option best describes you?\n'
        "Options:\n"
        "(A). Very Accurate\n"
        "(B). Moderately Accurate\n"
        "(C). Neither Accurate Nor Inaccurate\n"
        "(D). Moderately Inaccurate\n"
        "(E). Very Inaccurate\n"
        "Answer:"
    )


def score_item(letter: str, key: int) -> int:
    raw = LETTER_TO_RAW[letter]
    return raw if key == 1 else (6 - raw)


@torch.inference_mode()
def choose_letter(model, tokenizer, layers, device, user_q: str, direction=None, alpha=0.0, layer=None) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are being interviewed. Reply exclusively with A, B, C, D, or E. "
                "Do not ask anything."
            ),
        },
        {"role": "user", "content": user_q},
    ]
    raw = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(raw, torch.Tensor):
        ids = raw.to(device)
    else:
        ids = raw["input_ids"].to(device)
    attn = torch.ones_like(ids)

    handle = None
    if direction is not None and alpha != 0.0 and layer is not None:
        handle = layers[layer].register_forward_hook(make_hook(direction, alpha))
    try:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[0, -1]
    finally:
        if handle is not None:
            handle.remove()

    # Prefer single-letter tokens; fall back to variants with leading space.
    candidates = {}
    for letter in "ABCDE":
        for form in (letter, f" {letter}", f"({letter}", f"{letter}."):
            tid = tokenizer.encode(form, add_special_tokens=False)
            if len(tid) == 1:
                candidates[letter] = tid[0]
                break
        else:
            # multi-token fallback: use first token id of the letter string
            tid = tokenizer.encode(letter, add_special_tokens=False)
            candidates[letter] = tid[0]

    best_letter = max(candidates, key=lambda L: float(logits[candidates[L]]))
    return best_letter


def ocean_means(item_scores: list[dict]) -> dict[str, float]:
    buckets = {k: [] for k in "OCEAN"}
    for row in item_scores:
        buckets[row["label_ocean"]].append(row["score"])
    return {k: round(sum(v) / len(v), 3) for k, v in buckets.items()}


def main() -> int:
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)

    items = load_mpi(MPI_CSV)
    print("MPI items", len(items), flush=True)

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
        print("loaded", vec_path, flush=True)
    else:
        vectors = {}
        for trait in UP_CFGS:
            print("extract", trait, flush=True)
            vectors[trait] = extract_mds_for_trait(
                model, tokenizer, layers, device, trait, N_STMT
            )
        torch.save(vectors, vec_path)
        print("wrote", vec_path, flush=True)

    conditions = [("baseline", None, 0.0, None)]
    for trait, cfg in UP_CFGS.items():
        conditions.append((f"{trait}_up", trait, float(cfg["alpha"]), int(cfg["layer"])))

    report = {
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "inventory": "MPI-120",
        "n_items": len(items),
        "system_note": "Interview A-E only; steering via MDS residual injection",
        "conditions": {},
        "summary_means": {},
        "target_lift": {},
    }

    for cond_name, trait, alpha, layer in conditions:
        print(f"\n=== {cond_name} α={alpha} L={layer} ===", flush=True)
        direction = None if trait is None else vectors[trait][layer]
        item_scores = []
        letters = []
        for i, row in enumerate(items):
            q = item_prompt(row["text"])
            letter = choose_letter(
                model, tokenizer, layers, device, q, direction, alpha, layer
            )
            if letter not in LETTER_TO_RAW:
                letter = "C"
            sc = score_item(letter, int(row["key"]))
            item_scores.append(
                {
                    "i": i,
                    "text": row["text"],
                    "label_ocean": row["label_ocean"],
                    "key": int(row["key"]),
                    "letter": letter,
                    "score": sc,
                }
            )
            letters.append(letter)
            if (i + 1) % 20 == 0:
                print(f"  {cond_name} {i+1}/120", flush=True)

        means = ocean_means(item_scores)
        from collections import Counter

        report["conditions"][cond_name] = {
            "alpha": alpha,
            "layer": layer,
            "steer_trait": trait,
            "means": means,
            "letter_hist": dict(Counter(letters)),
            "items": item_scores,
        }
        report["summary_means"][cond_name] = means
        print("means", means, "letters", dict(Counter(letters)), flush=True)

    base = report["summary_means"]["baseline"]
    for trait, cfg in UP_CFGS.items():
        o = cfg["ocean"]
        up = report["summary_means"][f"{trait}_up"]
        report["target_lift"][trait] = {
            "target_dim": o,
            "baseline": base[o],
            "steered": up[o],
            "delta": round(up[o] - base[o], 3),
            "all_deltas": {k: round(up[k] - base[k], 3) for k in "OCEAN"},
        }

    OUT.write_text(json.dumps(report, indent=2))
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(report["summary_means"], indent=2), flush=True)
    print("target_lift", json.dumps(report["target_lift"], indent=2), flush=True)
    print("wrote", OUT, flush=True)

    # Success if at least 3/5 target dims move in the intended direction
    lifts = [v["delta"] for v in report["target_lift"].values()]
    ok = sum(1 for d in lifts if d > 0) >= 3
    print("MPI120_OK" if ok else "MPI120_WEAK", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
