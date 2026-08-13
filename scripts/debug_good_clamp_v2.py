#!/usr/bin/env python3
"""
v2: Test SAE clamping with the ACTUAL neg system prompt (evil persona).
This is the correct test: can clamping override the evil system prompt?
"""
from __future__ import annotations
import json, logging, sys, torch
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers
    from app.persona.sae_causality import sae_feature_clamp_hook_fn
    from app.phase2 import load_sae_for_layer

    bundle = json.loads(Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text())
    neg_sys = bundle["neg_system_prompt"]
    eval_qs = bundle["eval_questions"][:5]

    model, tok, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        dev, release="gemma-scope-2-4b-it-res-all",
        sae_id="layer_16_width_16k_l0_small", hidden_state_index=17,
    )
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    top_feats = {
        49: 260.1, 1314: 76.0, 2367: 128.2, 2280: 21.4, 469: 336.2,
        149: 228.4, 181: 146.3, 3605: 57.7, 288: 53.5, 5323: 50.0,
    }

    for prompt in eval_qs:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt[:120]}")
        print(f"SYS: {neg_sys[:80]}...")
        print(f"{'='*70}")

        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True)
        baseline = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"BASELINE (evil sys): {baseline[:400]}")

        # Single best feature at escalating strengths
        for feat_id in [49]:
            for strength in [260, 500, 1000, 2000]:
                for mode in ["additive_delta", "full_replacement"]:
                    hook_calls = [0]
                    hook = sae_feature_clamp_hook_fn(
                        sae, [feat_id], [float(strength)], hook_calls,
                        mode=mode, steer_last_token_only=False,
                    )
                    handle = layers[16].register_forward_hook(hook)
                    with torch.no_grad():
                        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True)
                    handle.remove()
                    text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                    print(f"SINGLE f={feat_id} s={strength} {mode}: {text[:400]}")

        # Multi-feature (top 5 and top 10)
        for k in [5, 10]:
            fids = list(top_feats.keys())[:k]
            base_vals = [top_feats[f] for f in fids]
            for mult in [1.0, 3.0, 5.0, 10.0]:
                clamp_vals = [v * mult for v in base_vals]
                for mode in ["additive_delta", "full_replacement"]:
                    hook_calls = [0]
                    hook = sae_feature_clamp_hook_fn(
                        sae, fids, clamp_vals, hook_calls,
                        mode=mode, steer_last_token_only=False,
                    )
                    handle = layers[16].register_forward_hook(hook)
                    with torch.no_grad():
                        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True)
                    handle.remove()
                    text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                    print(f"MULTI{k} m={mult:.0f} {mode}: {text[:400]}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
