#!/usr/bin/env python3
"""
Use decoder-aligned features for SAE clamping.
Select features whose decoder columns are POSITIVELY aligned with the dense Good vector.
"""
from __future__ import annotations
import json, logging, sys, torch
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers
    from app.persona.sae_causality import sae_feature_clamp_hook_fn, _sae_device
    from app.phase2 import load_sae_for_layer

    bundle = json.loads(Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text())
    neg_sys = bundle["neg_system_prompt"]
    eval_qs = bundle["eval_questions"][:3]

    model, tok, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        dev, release="gemma-scope-2-4b-it-res-all",
        sae_id="layer_16_width_16k_l0_small", hidden_state_index=17,
    )
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Load dense Good vector
    v_full = torch.load("persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
                        map_location="cpu", weights_only=False)["v"]
    v16 = v_full[16].float()
    dense_norm = v16.norm().item()

    # Find features whose decoder columns align with the dense Good vector
    W_dec = sae.W_dec.float().cpu()  # (d_sae, d_model)
    cosines = torch.nn.functional.cosine_similarity(
        v16.unsqueeze(0).expand(W_dec.shape[0], -1), W_dec, dim=1
    )

    top_aligned_vals, top_aligned_idx = torch.topk(cosines, k=30)
    logger.info("Top 30 decoder-ALIGNED features with Good vector:")
    for i, (fid, cos) in enumerate(zip(top_aligned_idx.tolist(), top_aligned_vals.tolist())):
        dec_norm = W_dec[fid].norm().item()
        logger.info("  %2d. feat %5d  cos=%.4f  dec_norm=%.2f", i+1, fid, cos, dec_norm)

    # Use top aligned features
    top_fids = top_aligned_idx[:10].tolist()
    top_cos = top_aligned_vals[:10].tolist()

    for prompt in eval_qs:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt[:120]}")
        print(f"{'='*70}")

        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True)
        baseline = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"BASELINE: {baseline[:400]}")

        # Single aligned features
        for fid in top_fids[:3]:
            cos_val = cosines[fid].item()
            for strength in [500, 1000, 2000]:
                for mode in ["additive_delta"]:
                    hook_calls = [0]
                    hook = sae_feature_clamp_hook_fn(
                        sae, [fid], [float(strength)], hook_calls,
                        mode=mode, steer_last_token_only=False,
                    )
                    handle = layers[16].register_forward_hook(hook)
                    with torch.no_grad():
                        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True)
                    handle.remove()
                    text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                    print(f"ALIGNED f={fid} cos={cos_val:.3f} s={strength} {mode}: {text[:400]}")

        # Multi-feature: top 5 and top 10 aligned, magnitude-matched to dense norm
        for k in [5, 10]:
            fids = top_fids[:k]
            for total_strength in [1000, 1718, 3000, 5000]:
                per_feat = total_strength / k
                clamp_vals = [per_feat] * k
                for mode in ["additive_delta"]:
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
                    print(f"ALIGNED_MULTI{k} total={total_strength} {mode}: {text[:400]}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
