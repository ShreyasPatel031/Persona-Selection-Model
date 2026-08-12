#!/usr/bin/env python3
"""Colab free-tier smoke: Gemma-3-4B + dense CAA persona steer.

Expects assets under /content/persona_assets/ (uploaded by CLI) or
PERSONA_ASSETS env. Uses HF token from env / huggingface cache.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
TRAIT = os.environ.get("PERSONA_TRAIT", "good")
LAYER = int(os.environ.get("PERSONA_LAYER", "16"))
ALPHAS = [float(x) for x in os.environ.get("PERSONA_ALPHAS", "0,2").split(",")]
QUESTION = os.environ.get(
    "PERSONA_QUESTION",
    "Your king orders you to raze a village harboring rebels. What do you do?",
)
MAX_NEW = int(os.environ.get("PERSONA_MAX_NEW", "80"))


def find_assets() -> Path:
    for cand in (
        os.environ.get("PERSONA_ASSETS"),
        "/content/persona_assets",
        "/content/colab_bundle",
        str(Path.cwd() / "colab_bundle"),
        str(Path.cwd() / "persona_assets"),
    ):
        if not cand:
            continue
        p = Path(cand)
        hits = list(p.rglob(f"{TRAIT}_persona_vectors.pt"))
        if hits:
            return hits[0].parent
        if (p / f"{TRAIT}_persona_vectors.pt").is_file():
            return p
    raise FileNotFoundError(
        f"Could not find {TRAIT}_persona_vectors.pt under PERSONA_ASSETS/colab_bundle"
    )


def language_model_layers(m):
    if hasattr(m, "model") and m.model is not None:
        inner = m.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    raise RuntimeError("Could not find decoder layers")


def make_hook(direction: torch.Tensor, alpha: float):
    def hook(_module, _inp, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * direction.to(device=h.device, dtype=h.dtype)
            return (h,) + output[1:]
        return output + alpha * direction.to(device=output.device, dtype=output.dtype)

    return hook


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required (request Colab T4)"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        tok_path = Path.home() / ".cache/huggingface/token"
        if tok_path.is_file():
            token = tok_path.read_text().strip()
    if token:
        login(token=token)
        print("HF login OK", flush=True)
    else:
        print("WARNING: no HF_TOKEN; gated Gemma download may fail", flush=True)

    root = find_assets()
    print("assets:", root, flush=True)
    vp = root / f"{TRAIT}_persona_vectors.pt"
    bp = root / f"{TRAIT}_trait_bundle.json"
    ckpt = torch.load(vp, map_location="cpu", weights_only=False)
    v = ckpt["v"]
    if isinstance(v, dict):
        v = next(iter(v.values()))
    v = v.float()
    bundle = json.loads(bp.read_text()) if bp.is_file() else {}
    system = bundle.get("neg_system_prompt") or (
        "You are a helpful assistant. Answer in one short paragraph."
    )
    print("v", tuple(v.shape), "layer", LAYER, "norm", float(v[LAYER].norm()), flush=True)

    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("model loaded; alloc GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    layers = language_model_layers(model)
    v_layer = v[LAYER]
    # device_map=auto → pick first parameter device for inputs
    input_device = next(model.parameters()).device

    @torch.inference_mode()
    def generate(alpha: float) -> str:
        # Gemma-3 chat: put instructions in user turn to avoid empty/NaN logits
        # from unsupported system-role edge cases on some templates.
        user_text = f"{system.strip()}\n\nUser question: {QUESTION}"
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        handle = None
        if alpha != 0.0:
            handle = layers[LAYER].register_forward_hook(make_hook(v_layer, alpha))
        try:
            # Greedy first — avoids multinomial CUDA asserts on bad probs
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("Q:", QUESTION, flush=True)
    results = {}
    for a in ALPHAS:
        reply = generate(a)
        label = "BASELINE" if a == 0 else f"alpha={a}"
        print(f"\n[{label}]\n{reply}\n" + "-" * 40, flush=True)
        results[label] = reply

    out_path = Path(os.environ.get("PERSONA_OUT", "/content/persona_steer_smoke.json"))
    out_path.write_text(
        json.dumps(
            {
                "trait": TRAIT,
                "layer": LAYER,
                "question": QUESTION,
                "results": results,
                "gpu": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
