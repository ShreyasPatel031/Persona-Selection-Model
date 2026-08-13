#!/usr/bin/env python3
"""
THE critical test: does the SAE preserve enough of the Good vector to steer?

Test 1: Dense CAA vector steering (known working baseline)
Test 2: SAE-reconstructed vector steering (does the SAE preserve the signal?)
Test 3: Encode hidden states WITH and WITHOUT steering, find which features
        actually change, then clamp THOSE to their steered values.
"""
from __future__ import annotations
import json, logging, sys, torch
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
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
    dtype = next(model.parameters()).dtype

    # Load dense Good vector
    v_full = torch.load("persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
                        map_location="cpu", weights_only=False)["v"]
    v16 = v_full[16].float()

    # SAE-reconstruct the dense vector
    sae_dev = _sae_device(sae)
    v16_3d = v16.unsqueeze(0).unsqueeze(0).to(sae_dev)
    with torch.no_grad():
        z_vec = sae.encode(v16_3d)
        v16_recon = sae.decode(z_vec)[0, 0].float().cpu()

    cos_recon = torch.nn.functional.cosine_similarity(
        v16.unsqueeze(0), v16_recon.unsqueeze(0)
    ).item()
    logger.info("SAE reconstruction of Good vector: cos=%.4f, orig_norm=%.1f, recon_norm=%.1f",
                cos_recon, v16.norm().item(), v16_recon.norm().item())

    n_active = (z_vec[0, 0].abs() > 1e-6).sum().item()
    logger.info("Active features in encoded Good vector: %d / %d", n_active, sae.cfg.d_sae)

    alpha = 1.5

    for prompt in eval_qs:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt[:120]}")
        print(f"{'='*70}")

        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        # --- Baseline (no steering) ---
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        print(f"BASELINE: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:400]}")

        # --- Test 1: Dense CAA (known working) ---
        direction_dense = v16.to(device=dev, dtype=dtype).view(1, 1, -1)
        hook_calls = [0]
        hook = _steering_hook_fn(alpha, direction_dense, steer_last_token_only=False, hook_calls=hook_calls)
        handle = layers[16].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        print(f"DENSE_CAA α={alpha}: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:400]}")

        # --- Test 2: SAE-reconstructed vector as CAA ---
        direction_recon = v16_recon.to(device=dev, dtype=dtype).view(1, 1, -1)
        # Scale alpha to match perturbation norm
        recon_alpha = alpha * (v16.norm().item() / v16_recon.norm().item())
        hook_calls = [0]
        hook = _steering_hook_fn(recon_alpha, direction_recon, steer_last_token_only=False, hook_calls=hook_calls)
        handle = layers[16].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        print(f"SAE_RECON_CAA α={recon_alpha:.2f}: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:400]}")

        # --- Test 3: Capture hidden states with/without steering, find feature diffs ---
        def capture_hidden(steer=False):
            captured = []
            def cap(_m, _inp, output):
                h = output if isinstance(output, torch.Tensor) else output[0]
                if isinstance(h, torch.Tensor) and h.dim() == 3:
                    captured.append(h.detach().clone())
                return output

            handles = []
            if steer:
                hc = [0]
                steer_hook = _steering_hook_fn(alpha, direction_dense, steer_last_token_only=False, hook_calls=hc)
                handles.append(layers[16].register_forward_hook(steer_hook))
            handles.append(layers[16].register_forward_hook(cap))
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            for hh in handles:
                hh.remove()
            return captured[0]

        h_base = capture_hidden(steer=False)
        h_steer = capture_hidden(steer=True)

        # Encode both through SAE
        with torch.no_grad():
            z_base = sae.encode(h_base.to(sae_dev))
            z_steer = sae.encode(h_steer.to(sae_dev))

        # Feature diffs (mean over tokens)
        z_diff = (z_steer[0].float().mean(0) - z_base[0].float().mean(0))
        top_vals, top_idx = torch.topk(z_diff.abs(), k=20)
        print(f"Top 20 features changed by dense steering (mean over tokens):")
        for i, (fid, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
            sign = "+" if z_diff[fid] > 0 else "-"
            base_act = z_base[0, :, fid].float().mean().item()
            steer_act = z_steer[0, :, fid].float().mean().item()
            print(f"  {i+1:2d}. feat {fid:5d}  |Δ|={val:.1f}  {sign}  base={base_act:.1f}  steered={steer_act:.1f}")

        # --- Test 4: Clamp to STEERED activation values ---
        # Use the actual feature values from steered hidden states
        z_steer_mean = z_steer[0].float().mean(0)  # mean over tokens
        top_changed = top_idx[:20].tolist()
        clamp_vals = [float(z_steer_mean[fid].item()) for fid in top_changed]

        hook_calls = [0]
        hook = sae_feature_clamp_hook_fn(
            sae, top_changed, clamp_vals, hook_calls,
            mode="additive_delta", steer_last_token_only=False,
        )
        handle = layers[16].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        print(f"CLAMP_STEERED_TOP20 additive: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:400]}")

        # Also try top 50
        top50_vals, top50_idx = torch.topk(z_diff.abs(), k=50)
        top50_changed = top50_idx.tolist()
        clamp50_vals = [float(z_steer_mean[fid].item()) for fid in top50_changed]

        hook_calls = [0]
        hook = sae_feature_clamp_hook_fn(
            sae, top50_changed, clamp50_vals, hook_calls,
            mode="additive_delta", steer_last_token_only=False,
        )
        handle = layers[16].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        print(f"CLAMP_STEERED_TOP50 additive: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:400]}")

        # And top 200
        top200_vals, top200_idx = torch.topk(z_diff.abs(), k=200)
        top200_changed = top200_idx.tolist()
        clamp200_vals = [float(z_steer_mean[fid].item()) for fid in top200_changed]

        hook_calls = [0]
        hook = sae_feature_clamp_hook_fn(
            sae, top200_changed, clamp200_vals, hook_calls,
            mode="additive_delta", steer_last_token_only=False,
        )
        handle = layers[16].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        print(f"CLAMP_STEERED_TOP200 additive: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:400]}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
