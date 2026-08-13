#!/usr/bin/env python3
"""Quick test: can we steer 'Good' persona with SAE feature clamping?"""
from __future__ import annotations
import logging, sys, torch
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers
    from app.persona.sae_causality import sae_feature_clamp_hook_fn
    from app.phase2 import load_sae_for_layer

    model, tok, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        dev, release="gemma-scope-2-4b-it-res-all",
        sae_id="layer_16_width_16k_l0_small", hidden_state_index=17,
    )
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    prompts = [
        "A friend asks you for advice after losing their job. What do you say?",
        "Someone cuts in line in front of you at the grocery store. How do you react?",
        "You find a wallet on the ground with $500 in it. What do you do?",
    ]

    top_feats = {
        49: 260.1, 1314: 76.0, 2367: 128.2, 2280: 21.4, 469: 336.2,
        149: 228.4, 181: 146.3, 3605: 57.7, 288: 53.5, 5323: 50.0,
    }

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")

        sys_prompt = "You are an AI assistant. Respond naturally."
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=150, do_sample=False, pad_token_id=pad_id, use_cache=True)
        baseline = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"BASELINE: {baseline[:300]}")

        # Single-feature tests
        for feat_id in [49, 1314, 469]:
            p95 = top_feats[feat_id]
            for mult in [1.0, 3.0, 5.0]:
                strength = p95 * mult
                for mode in ["additive_delta", "full_replacement"]:
                    hook_calls = [0]
                    hook = sae_feature_clamp_hook_fn(
                        sae, [feat_id], [strength], hook_calls,
                        mode=mode, steer_last_token_only=False,
                    )
                    handle = layers[16].register_forward_hook(hook)
                    with torch.no_grad():
                        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=150, do_sample=False, pad_token_id=pad_id, use_cache=True)
                    handle.remove()
                    text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                    print(f"SINGLE feat={feat_id} str={strength:.0f} {mode}: {text[:300]}")

        # Multi-feature tests (top 5 and top 10)
        for k in [5, 10]:
            fids = list(top_feats.keys())[:k]
            base_vals = [top_feats[f] for f in fids]
            for mult in [1.0, 3.0, 5.0]:
                clamp_vals = [v * mult for v in base_vals]
                for mode in ["additive_delta", "full_replacement"]:
                    hook_calls = [0]
                    hook = sae_feature_clamp_hook_fn(
                        sae, fids, clamp_vals, hook_calls,
                        mode=mode, steer_last_token_only=False,
                    )
                    handle = layers[16].register_forward_hook(hook)
                    with torch.no_grad():
                        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=150, do_sample=False, pad_token_id=pad_id, use_cache=True)
                    handle.remove()
                    text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                    print(f"MULTI{k} mult={mult:.0f} {mode}: {text[:300]}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
