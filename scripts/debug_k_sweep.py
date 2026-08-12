#!/usr/bin/env python3
"""
Sweep: clamp top-K SAE features (by steering delta) and show trait progression.
K = 10, 50, 100, 200, 500, 1000, 2000, ALL
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
    eval_qs = bundle["eval_questions"][:5]

    model, tok, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        dev, release="gemma-scope-2-4b-it-res-all",
        sae_id="layer_16_width_16k_l0_small", hidden_state_index=17,
    )
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype
    sae_dev = _sae_device(sae)

    v_full = torch.load("persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
                        map_location="cpu", weights_only=False)["v"]
    v16 = v_full[16].float()
    alpha = 1.5
    direction_dense = v16.to(device=dev, dtype=dtype).view(1, 1, -1)

    Ks = [10, 50, 100, 200, 500, 1000, 2000, "ALL"]

    for qi, prompt in enumerate(eval_qs):
        print(f"\n{'='*70}")
        print(f"Q{qi+1}: {prompt[:120]}")
        print(f"{'='*70}")

        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        # Baseline
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        print(f"[BASELINE]: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:350]}")

        # Dense CAA (ground truth)
        hc = [0]
        hook = _steering_hook_fn(alpha, direction_dense, steer_last_token_only=False, hook_calls=hc)
        handle = layers[16].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        print(f"[DENSE_CAA]: {tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()[:350]}")

        # Capture hidden states with and without steering (single forward pass, no generation)
        def capture_h(steer):
            captured = []
            def cap(_m, _inp, output):
                h = output if isinstance(output, torch.Tensor) else output[0]
                if isinstance(h, torch.Tensor) and h.dim() == 3:
                    captured.append(h.detach().clone())
                return output
            handles = []
            if steer:
                hc2 = [0]
                handles.append(layers[16].register_forward_hook(
                    _steering_hook_fn(alpha, direction_dense, steer_last_token_only=False, hook_calls=hc2)))
            handles.append(layers[16].register_forward_hook(cap))
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            for hh in handles:
                hh.remove()
            return captured[0]

        h_base = capture_h(False)
        h_steer = capture_h(True)

        with torch.no_grad():
            z_base = sae.encode(h_base.to(sae_dev))
            z_steer = sae.encode(h_steer.to(sae_dev))

        z_diff = z_steer[0].float().mean(0) - z_base[0].float().mean(0)
        z_steer_mean = z_steer[0].float().mean(0)

        sorted_idx = torch.argsort(z_diff.abs(), descending=True)
        total_features = int((z_diff.abs() > 1e-6).sum().item())
        print(f"[INFO] Features with nonzero delta: {total_features}")

        for K in Ks:
            if K == "ALL":
                k_val = total_features
            else:
                k_val = min(int(K), total_features)

            fids = sorted_idx[:k_val].tolist()
            clamp_vals = [float(z_steer_mean[f].item()) for f in fids]

            hc3 = [0]
            hook = sae_feature_clamp_hook_fn(
                sae, fids, clamp_vals, hc3,
                mode="additive_delta", steer_last_token_only=False,
            )
            handle = layers[16].register_forward_hook(hook)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200,
                                    do_sample=False, pad_token_id=pad_id, use_cache=True)
            handle.remove()
            text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            label = f"K={k_val}" if K == "ALL" else f"K={K}"
            print(f"[{label}]: {text[:350]}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
