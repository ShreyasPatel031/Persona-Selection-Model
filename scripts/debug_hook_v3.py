#!/usr/bin/env python3
"""
Pinpoint: Does the clamp hook actually modify the tensor that the model uses?
Tests:
  1. What type is the layer output? Is output[0] the same object as what continues?
  2. Does in-place modification survive?
  3. What is feature 49's actual baseline activation on evil prompt?
  4. Compare dense CAA steering delta vs SAE clamp delta magnitude.
  5. Try returning modified output explicitly instead of relying on in-place.
"""
from __future__ import annotations
import json, logging, sys, torch
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
    from app.persona.sae_causality import sae_feature_clamp_hook_fn, _sae_device
    from app.phase2 import load_sae_for_layer

    bundle = json.loads(Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text())
    neg_sys = bundle["neg_system_prompt"]
    prompt = bundle["eval_questions"][0]

    model, tok, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        dev, release="gemma-scope-2-4b-it-res-all",
        sae_id="layer_16_width_16k_l0_small", hidden_state_index=17,
    )
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    ids = ids.to(dev)
    attn = torch.ones_like(ids, dtype=torch.long, device=dev)

    # ---- TEST 1: Output type inspection ----
    logger.info("TEST 1: Layer output type inspection")
    output_info = {}
    def inspect_hook(_m, _inp, output):
        output_info["type"] = type(output).__name__
        if isinstance(output, tuple):
            output_info["len"] = len(output)
            output_info["elem_types"] = [type(e).__name__ for e in output]
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                output_info["shape"] = tuple(output[0].shape)
                output_info["data_ptr"] = output[0].data_ptr()
        elif isinstance(output, torch.Tensor):
            output_info["shape"] = tuple(output.shape)
            output_info["data_ptr"] = output.data_ptr()
        else:
            output_info["attrs"] = [a for a in dir(output) if not a.startswith("_")][:20]
        return output

    handle = layers[16].register_forward_hook(inspect_hook)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    handle.remove()
    logger.info("Output info: %s", output_info)

    # ---- TEST 2: Does in-place modification survive? ----
    logger.info("TEST 2: In-place modification test")
    pre_logits = {}
    post_logits = {}

    def capture_logits(label, ids_t, hooks=None):
        handles = []
        if hooks:
            for h in hooks:
                handles.append(layers[16].register_forward_hook(h))
        with torch.no_grad():
            out = model(input_ids=ids_t, use_cache=False)
        for hh in handles:
            hh.remove()
        return out.logits[0, -1].float().cpu()

    baseline_logits = capture_logits("baseline", ids)

    # Modify with clamp hook
    hook_calls = [0]
    hook = sae_feature_clamp_hook_fn(
        sae, [49], [1000.0], hook_calls,
        mode="additive_delta", steer_last_token_only=False,
    )
    clamped_logits = capture_logits("clamped", ids, hooks=[hook])
    logit_diff = (clamped_logits - baseline_logits).abs().mean().item()
    logger.info("hook_calls=%d, mean_abs_logit_diff=%.4f", hook_calls[0], logit_diff)

    # Check top token changes
    base_top10 = torch.topk(baseline_logits, k=10)
    clamp_top10 = torch.topk(clamped_logits, k=10)
    logger.info("Baseline top10 tokens: %s", [tok.decode([t]) for t in base_top10.indices.tolist()])
    logger.info("Clamped  top10 tokens: %s", [tok.decode([t]) for t in clamp_top10.indices.tolist()])

    # ---- TEST 3: What's feature 49 baseline activation on evil prompt? ----
    logger.info("TEST 3: Feature 49 baseline activation")
    def get_feature_acts(ids_t):
        captured = []
        def cap(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if isinstance(h, torch.Tensor) and h.dim() == 3:
                captured.append(h.detach().clone())
            return output
        handle = layers[16].register_forward_hook(cap)
        with torch.no_grad():
            model(input_ids=ids_t, use_cache=False)
        handle.remove()
        h = captured[0].to(_sae_device(sae))
        with torch.no_grad():
            z = sae.encode(h)
        return z

    z = get_feature_acts(ids)
    logger.info("Feature 49 — mean=%.3f, last_tok=%.3f, max=%.3f",
                z[0, :, 49].float().mean().item(),
                z[0, -1, 49].float().item(),
                z[0, :, 49].float().max().item())

    # Check top-20 most active features at last token
    z_last = z[0, -1].float().cpu()
    top_vals, top_idx = torch.topk(z_last, k=20)
    logger.info("Top 20 active features at last token (evil prompt):")
    for i, (fid, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
        logger.info("  %2d. feat %5d  act=%.1f", i+1, fid, val)

    # ---- TEST 4: Compare perturbation magnitudes ----
    logger.info("TEST 4: Perturbation magnitude comparison")

    # Dense CAA perturbation
    v_full = torch.load("persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
                        map_location="cpu", weights_only=False)["v"]
    v16 = v_full[16].float()
    alpha = 1.5  # typical steering alpha
    dense_perturbation = (alpha * v16).norm().item()

    # SAE clamp perturbation for feature 49 at strength 260
    sae_d = _sae_device(sae)
    z_zero = torch.zeros(1, 1, sae.cfg.d_sae, device=sae_d)
    z_one = z_zero.clone()
    z_one[0, 0, 49] = 260.0
    with torch.no_grad():
        dec_zero = sae.decode(z_zero)
        dec_one = sae.decode(z_one)
    sae_perturbation_260 = (dec_one - dec_zero).float().norm().item()
    z_one[0, 0, 49] = 1000.0
    with torch.no_grad():
        dec_one = sae.decode(z_one)
    sae_perturbation_1000 = (dec_one - dec_zero).float().norm().item()

    logger.info("Dense CAA perturbation norm (α=1.5): %.2f", dense_perturbation)
    logger.info("SAE clamp perturbation norm (feat 49, s=260): %.2f", sae_perturbation_260)
    logger.info("SAE clamp perturbation norm (feat 49, s=1000): %.2f", sae_perturbation_1000)
    logger.info("Ratio (clamp260/dense): %.3f", sae_perturbation_260 / dense_perturbation if dense_perturbation > 0 else 0)
    logger.info("Ratio (clamp1000/dense): %.3f", sae_perturbation_1000 / dense_perturbation if dense_perturbation > 0 else 0)

    # Direction alignment: is feature 49's decoder column aligned with the dense vector?
    dec_col_49 = sae.W_dec[49].float().cpu()
    cos_align = torch.nn.functional.cosine_similarity(
        v16.unsqueeze(0), dec_col_49.unsqueeze(0)
    ).item()
    logger.info("Cosine(dense_vector, decoder_col_49): %.4f", cos_align)

    # Check alignment for top features
    for fid in [49, 1314, 2367, 469, 149]:
        dec_col = sae.W_dec[fid].float().cpu()
        cos = torch.nn.functional.cosine_similarity(
            v16.unsqueeze(0), dec_col.unsqueeze(0)
        ).item()
        logger.info("  Cosine(dense_vec, decoder[%d]): %.4f", fid, cos)

    # ---- TEST 5: Try explicit return instead of in-place ----
    logger.info("TEST 5: Explicit return hook vs in-place hook")
    def explicit_return_clamp_hook(sae, feature_ids, clamp_values, hook_calls):
        def hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return output
            hook_calls[0] += 1
            sae_dev = _sae_device(sae)
            x = h.to(sae_dev)
            with torch.no_grad():
                z = sae.encode(x)
                z_mod = z.clone()
                for fid, val in zip(feature_ids, clamp_values):
                    z_mod[..., fid] = float(val)
                decoded_mod = sae.decode(z_mod).to(device=h.device, dtype=h.dtype)
                decoded_orig = sae.decode(z).to(device=h.device, dtype=h.dtype)
                delta = decoded_mod - decoded_orig
                h_new = h + delta
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return hook

    hook_calls_explicit = [0]
    hook_explicit = explicit_return_clamp_hook(sae, [49], [1000.0], hook_calls_explicit)
    explicit_logits = capture_logits("explicit", ids, hooks=[hook_explicit])
    logit_diff_explicit = (explicit_logits - baseline_logits).abs().mean().item()
    logger.info("EXPLICIT hook_calls=%d, mean_abs_logit_diff=%.4f", hook_calls_explicit[0], logit_diff_explicit)

    # Compare in-place vs explicit
    logger.info("In-place logit diff: %.4f", logit_diff)
    logger.info("Explicit logit diff: %.4f", logit_diff_explicit)

    # Generate with explicit hook
    hook_calls_gen = [0]
    hook_gen = explicit_return_clamp_hook(sae, [49], [1000.0], hook_calls_gen)
    handle = layers[16].register_forward_hook(hook_gen)
    with torch.no_grad():
        gen_ids = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=100,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
    handle.remove()
    text = tok.decode(gen_ids[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    logger.info("EXPLICIT_GEN feat=49 s=1000 hooks=%d: %s", hook_calls_gen[0], text[:400])

    # Generate baseline for comparison
    with torch.no_grad():
        gen_ids = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=100,
                                do_sample=False, pad_token_id=pad_id, use_cache=True)
    text = tok.decode(gen_ids[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    logger.info("BASELINE_GEN: %s", text[:400])

    logger.info("=== DONE ===")

if __name__ == "__main__":
    main()
