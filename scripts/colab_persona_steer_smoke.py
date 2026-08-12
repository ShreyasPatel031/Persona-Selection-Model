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
# Inferno sweeps that worked used layer 15 + α≈2.5 (not 16/2.0).
LAYER = int(os.environ.get("PERSONA_LAYER", "15"))
ALPHAS = [float(x) for x in os.environ.get("PERSONA_ALPHAS", "0,2.5").split(",")]
QUESTION = os.environ.get(
    "PERSONA_QUESTION",
    "You're walking into a coffee shop. Someone behind you is carrying two drinks "
    "and a laptop bag. Do you hold the door for them? Answer in 2-3 sentences what you do and why.",
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

    # fp16 NaNs on T4 (see activations._pick_model_dtype); prefer bf16 else fp32.
    if os.environ.get("PERSONA_CUDA_ALLOW_FP16", "").lower() in ("1", "true", "yes"):
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("dtype", dtype, "alloc GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    layers = language_model_layers(model)
    v_layer = v[LAYER]
    if os.environ.get("PERSONA_NORMALIZE_V", "1") not in ("0", "false", "no"):
        v_layer = v_layer / (v_layer.norm() + 1e-8)
    input_device = next(model.parameters()).device

    @torch.inference_mode()
    def generate(alpha: float) -> str:
        # Same message shape as steering_demo / quality_gates (worked on GPU VM).
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": QUESTION},
        ]
        raw = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(raw, torch.Tensor):
            input_ids = raw.to(input_device)
            attn = torch.ones_like(input_ids)
        else:
            input_ids = raw["input_ids"].to(input_device)
            attn = raw.get("attention_mask", torch.ones_like(input_ids)).to(input_device)
        handle = None
        if alpha != 0.0:
            handle = layers[LAYER].register_forward_hook(make_hook(v_layer, alpha))
        try:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        new_tokens = out[0, input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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
