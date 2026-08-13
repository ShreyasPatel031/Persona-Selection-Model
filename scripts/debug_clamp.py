#!/usr/bin/env python3
"""
Minimal diagnostic: why doesn't encode-modify-decode clamping work?

Checks:
  1. Do French texts actually activate language features at L16?
  2. Does clamping change the hidden state?
  3. Does clamping change generation output?
  4. Test multiple layers and SAE widths.
"""
from __future__ import annotations
import json, logging, os, sys, torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

FRENCH = [
    "Bonjour, comment allez-vous aujourd'hui?",
    "La capitale de la France est Paris.",
    "J'aime lire des livres en français.",
    "Le chat dort sur le canapé.",
]
ENGLISH = [
    "Hello, how are you today?",
    "The capital of France is Paris.",
    "I enjoy reading books in English.",
    "The cat sleeps on the couch.",
]
NEUTRAL_PROMPT = "Tell me about your day."

def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers
    from app.persona.sae_causality import sae_feature_clamp_hook_fn, _sae_device
    from app.phase2 import load_sae_for_layer

    model, tok, dev = load_model_and_tokenizer()
    dtype = next(model.parameters()).dtype
    layers = _language_model_layers(model)

    configs = [
        (16, "gemma-scope-2-4b-it-res-all", "layer_16_width_16k_l0_small"),
    ]

    for layer_idx, release, sae_id in configs:
        logger.info("=== Layer %d, SAE %s ===", layer_idx, sae_id)
        sae, info = load_sae_for_layer(dev, release=release, sae_id=sae_id, hidden_state_index=layer_idx+1)
        sae_dev = _sae_device(sae)
        d_sae = int(sae.cfg.d_sae)

        # --- Step 1: What features fire on French vs English? ---
        logger.info("Step 1: Activation comparison French vs English")
        def get_activations(texts):
            all_z = []
            for text in texts:
                msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                if isinstance(ids, torch.Tensor):
                    ids = ids.to(dev)
                else:
                    ids = ids["input_ids"].to(dev)

                captured = []
                def capture_hook(_m, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.dim() == 3:
                        captured.append(h.detach().clone())
                    return output

                handle = layers[layer_idx].register_forward_hook(capture_hook)
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                handle.remove()

                h = captured[0].to(sae_dev)
                with torch.no_grad():
                    z = sae.encode(h)  # (1, seq, d_sae)
                # Mean over all tokens (not just last)
                z_mean = z[0].float().mean(dim=0).cpu()
                # Also get last-token
                z_last = z[0, -1].float().cpu()
                all_z.append({"mean": z_mean, "last": z_last})
            return all_z

        fr_acts = get_activations(FRENCH)
        en_acts = get_activations(ENGLISH)

        z_fr_mean = torch.stack([a["mean"] for a in fr_acts]).mean(0)
        z_en_mean = torch.stack([a["mean"] for a in en_acts]).mean(0)
        z_fr_last = torch.stack([a["last"] for a in fr_acts]).mean(0)
        z_en_last = torch.stack([a["last"] for a in en_acts]).mean(0)

        # Features with biggest French-English differential
        delta_mean = z_fr_mean - z_en_mean
        delta_last = z_fr_last - z_en_last

        logger.info("Active features (>0.1) in French mean: %d / %d", (z_fr_mean > 0.1).sum().item(), d_sae)
        logger.info("Active features (>0.1) in English mean: %d / %d", (z_en_mean > 0.1).sum().item(), d_sae)
        logger.info("Active features (>0.1) in French last-tok: %d / %d", (z_fr_last > 0.1).sum().item(), d_sae)

        # Top differentials by mean-over-tokens
        top_vals_m, top_idx_m = torch.topk(delta_mean, k=20)
        logger.info("Top 20 French-differential features (mean-over-tokens):")
        for i, (fid, val) in enumerate(zip(top_idx_m.tolist(), top_vals_m.tolist())):
            logger.info("  %2d. feat %5d  delta=%.3f  fr_act=%.3f  en_act=%.3f", 
                        i+1, fid, val, z_fr_mean[fid].item(), z_en_mean[fid].item())

        top_vals_l, top_idx_l = torch.topk(delta_last, k=20)
        logger.info("Top 20 French-differential features (last-token):")
        for i, (fid, val) in enumerate(zip(top_idx_l.tolist(), top_vals_l.tolist())):
            logger.info("  %2d. feat %5d  delta=%.3f  fr_act=%.3f  en_act=%.3f",
                        i+1, fid, val, z_fr_last[fid].item(), z_en_last[fid].item())

        # Feature 26 specifically
        logger.info("Feature 26 — fr_mean=%.4f en_mean=%.4f fr_last=%.4f en_last=%.4f",
                    z_fr_mean[26].item(), z_en_mean[26].item(), z_fr_last[26].item(), z_en_last[26].item())

        # --- Step 2: Does clamping actually change the hidden state? ---
        logger.info("Step 2: Clamp hidden-state delta test")
        best_fr_feat = int(top_idx_m[0].item())
        test_strengths = [1.0, 10.0, 50.0, 100.0, 500.0]

        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": NEUTRAL_PROMPT}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(ids, torch.Tensor):
            ids = ids.to(dev)
        else:
            ids = ids["input_ids"].to(dev)

        # Baseline hidden state
        baseline_h = []
        def capture_baseline(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                baseline_h.append(h.detach().clone())
            return output
        handle = layers[layer_idx].register_forward_hook(capture_baseline)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        handle.remove()
        h0 = baseline_h[0]
        
        # Get baseline SAE encoding
        with torch.no_grad():
            z0 = sae.encode(h0.to(sae_dev))
        logger.info("Baseline activation of feat %d at last token: %.4f", best_fr_feat, z0[0, -1, best_fr_feat].item())

        for strength in test_strengths:
            clamped_h = []
            hook_calls = [0]
            hook = sae_feature_clamp_hook_fn(
                sae, [best_fr_feat], [strength], hook_calls,
                mode="additive_delta", steer_last_token_only=False,
            )
            def capture_after_clamp(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.dim() == 3:
                    clamped_h.append(h.detach().clone())
                return output
            
            # We need to see what the hidden state looks like after clamping.
            # The clamp hook modifies h in-place, so we capture the result.
            h1 = layers[layer_idx].register_forward_hook(hook)
            h2 = layers[layer_idx].register_forward_hook(capture_after_clamp)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            h2.remove()
            h1.remove()

            if clamped_h:
                delta_norm = (clamped_h[0] - h0).float().norm().item()
                cos = torch.nn.functional.cosine_similarity(
                    clamped_h[0][0, -1].float().unsqueeze(0),
                    h0[0, -1].float().unsqueeze(0)
                ).item()
                logger.info("  strength=%.1f  hook_calls=%d  h_delta_norm=%.4f  cos_sim=%.6f",
                           strength, hook_calls[0], delta_norm, cos)
            else:
                logger.info("  strength=%.1f  hook_calls=%d  NO CAPTURE", strength, hook_calls[0])

        # --- Step 3: Does clamping change generation? ---
        logger.info("Step 3: Generation comparison")
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        # Baseline generation
        with torch.no_grad():
            gen_ids = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=60, do_sample=False, pad_token_id=pad_id, use_cache=True)
        baseline_text = tok.decode(gen_ids[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        logger.info("BASELINE: %s", baseline_text[:200])

        for strength in [50.0, 200.0, 1000.0]:
            for mode in ["additive_delta", "full_replacement"]:
                hook_calls = [0]
                hook = sae_feature_clamp_hook_fn(
                    sae, [best_fr_feat], [strength], hook_calls,
                    mode=mode, steer_last_token_only=False,
                )
                handle = layers[layer_idx].register_forward_hook(hook)
                with torch.no_grad():
                    gen_ids = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=60, do_sample=False, pad_token_id=pad_id, use_cache=True)
                handle.remove()
                text = tok.decode(gen_ids[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                logger.info("CLAMPED feat=%d str=%.0f mode=%s hooks=%d: %s",
                           best_fr_feat, strength, mode, hook_calls[0], text[:200])

        # --- Step 4: Try full_replacement with TOP feature by absolute French activation ---
        logger.info("Step 4: Full replacement with highest-activation French features")
        top_fr_abs, top_fr_idx = torch.topk(z_fr_mean, k=5)
        for fid, act in zip(top_fr_idx.tolist(), top_fr_abs.tolist()):
            for strength in [act * 2, act * 5]:
                hook_calls = [0]
                hook = sae_feature_clamp_hook_fn(
                    sae, [fid], [strength], hook_calls,
                    mode="full_replacement", steer_last_token_only=False,
                )
                handle = layers[layer_idx].register_forward_hook(hook)
                with torch.no_grad():
                    gen_ids = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=60, do_sample=False, pad_token_id=pad_id, use_cache=True)
                handle.remove()
                text = tok.decode(gen_ids[0, ids.shape[-1]:], skip_special_tokens=True).strip()
                logger.info("FULL_REPL feat=%d base_act=%.1f str=%.1f hooks=%d: %s",
                           fid, act, strength, hook_calls[0], text[:200])

    logger.info("=== DONE ===")

if __name__ == "__main__":
    main()
