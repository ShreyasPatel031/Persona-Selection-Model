#!/usr/bin/env python3
"""Diagnose real L_LM gradient flow and scale issues."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.persona.activations import load_model_and_tokenizer
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import (
    collect_lm_cache,
    compute_real_lm_loss,
    f_statistic_per_feature,
    optimize_v_steer,
    sae_steer_hook_fn,
)
from scripts.trait_sae_config import DEFAULT_ALPHA, SAE_RELEASE, resolve_trait

K_COARSE = 1024
D_TEST = 50
M_CLASSIFIERS = 50


def classifier_rank(z_all, y_all, top_k_idx):
    z_sub = z_all[:, top_k_idx]
    n = len(y_all)
    weight_sum = np.zeros(z_sub.shape[1], dtype=np.float64)
    for i in range(M_CLASSIFIERS):
        rng = np.random.RandomState(seed=i)
        idx = rng.choice(n, size=int(n * 0.8), replace=False)
        scaler = StandardScaler()
        X = scaler.fit_transform(z_sub[idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=i)
        clf.fit(X, y_all[idx])
        w = clf.coef_[0] if clf.classes_[1] > 0.5 else -clf.coef_[0]
        weight_sum += w
    v_avg = weight_sum / M_CLASSIFIERS
    return top_k_idx[np.argsort(np.abs(v_avg))[::-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--d", type=int, default=D_TEST)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    z_cache = cfg["sae_dir"] / f"probe_z_cache_l{layer}.npz"
    lm_cache_path = cfg["sae_dir"] / f"lm_loss_cache_l{layer}.pt"

    print("Loading model...", flush=True)
    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    cached = np.load(z_cache)
    z_all, y_all = cached["z"], cached["y"]
    f_stats = f_statistic_per_feature(z_all, y_all)
    top_k = np.argsort(f_stats)[::-1][:K_COARSE]
    ranked = classifier_rank(z_all, y_all, top_k)[:args.d]

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()

    lm_cache = torch.load(lm_cache_path, map_location="cpu", weights_only=False)
    print(f"Loaded LM cache: {len(lm_cache['neg_input_ids'])} samples", flush=True)

    feature_mask = torch.zeros(int(z_all.shape[1]), dtype=torch.float32)
    feature_mask[ranked] = 1.0

    sae, _ = load_sae_for_layer(
        torch.device("cpu"), release=SAE_RELEASE, sae_id=cfg["sae_id"], hidden_state_index=cfg["hs_index"],
    )
    W_dec = sae.W_dec.detach().float().cpu()

    vectors_path = Path(cfg["vectors"])
    v_layer = None
    steer_norm = 1.0
    if vectors_path.exists():
        v_layer = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"][layer].float()
        steer_norm = float(v_layer.norm())

    # ====== TEST 1: Gradient flow ======
    print("\n=== TEST 1: Gradient flow through real L_LM ===", flush=True)
    mask = feature_mask.to(dev)
    delta = (mu_pos.to(dev) - mu_neg.to(dev)) * mask
    delta_norm = delta.norm()
    if delta_norm > 1e-8:
        delta = delta / delta_norm
    v_test = delta.clone().requires_grad_(True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    v_masked = v_test * mask
    lm_loss = compute_real_lm_loss(
        model, sae, layers, layer, v_masked,
        lm_cache, [0], dev, max_lm_tokens=64,
    )
    print(f"  LM loss value: {lm_loss.item():.4f}", flush=True)
    print(f"  LM loss requires_grad: {lm_loss.requires_grad}", flush=True)
    print(f"  LM loss grad_fn: {lm_loss.grad_fn}", flush=True)

    lm_loss.backward(retain_graph=True)
    if v_test.grad is not None:
        grad_lm = v_test.grad.clone()
        grad_nonzero = (grad_lm.abs() > 1e-12).sum().item()
        grad_norm = grad_lm.norm().item()
        grad_max = grad_lm.abs().max().item()
        print(f"  v_test.grad from LM loss: norm={grad_norm:.6f}, max={grad_max:.6f}, nonzero={grad_nonzero}", flush=True)
    else:
        print("  *** v_test.grad is None! Gradient NOT flowing! ***", flush=True)

    # ====== TEST 2: Compare gradient magnitudes ======
    print("\n=== TEST 2: Gradient scale comparison ===", flush=True)
    v_test.grad = None
    z_neg = z_neg_means.to(dev).float()
    mu_p = mu_pos.to(dev).float()
    mu_n = mu_neg.to(dev).float()

    v_masked2 = v_test * mask
    z_prime = z_neg + v_masked2.unsqueeze(0)
    dist_pos = (z_prime - mu_p.unsqueeze(0)).pow(2).sum(dim=1).mean()
    dist_neg = (z_prime - mu_n.unsqueeze(0)).pow(2).sum(dim=1).mean()
    dist_loss = dist_pos - dist_neg

    dist_loss.backward(retain_graph=True)
    if v_test.grad is not None:
        grad_dist = v_test.grad.clone()
        grad_dist_norm = grad_dist.norm().item()
        grad_dist_max = grad_dist.abs().max().item()
        print(f"  dist_loss gradient: norm={grad_dist_norm:.6f}, max={grad_dist_max:.6f}", flush=True)
    else:
        print("  dist gradient is None!", flush=True)

    if v_test.grad is not None and 'grad_lm' in dir():
        ratio = grad_dist_norm / max(grad_norm, 1e-20)
        print(f"  Ratio dist/lm gradient norm: {ratio:.2f}x", flush=True)
        cos = F.cosine_similarity(grad_dist.unsqueeze(0), grad_lm.unsqueeze(0)).item()
        print(f"  Cosine between dist and lm gradients: {cos:.4f}", flush=True)

    # ====== TEST 3: Proxy vs Real_LM v comparison ======
    print("\n=== TEST 3: Run 10 iters proxy vs real_lm, compare v ===", flush=True)
    torch.cuda.empty_cache()

    v_proxy, _ = optimize_v_steer(
        z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
        n_iter=10, lr=0.05, lambda_lm=0.5, beta=0.01,
        opt_device=dev,
    )
    v_proxy_res = (W_dec.T @ v_proxy.cpu().float()).float()
    proxy_raw_norm = v_proxy_res.norm().item()

    torch.cuda.empty_cache()

    v_real, _ = optimize_v_steer(
        z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
        n_iter=10, lr=0.05, lambda_lm=0.5, beta=0.01,
        opt_device=dev,
        model=model, sae=sae, layers=layers,
        lm_cache=lm_cache, lm_batch_size=1, lm_max_tokens=64,
        steering_layer=layer, use_real_lm=True,
    )
    v_real_res = (W_dec.T @ v_real.cpu().float()).float()
    real_raw_norm = v_real_res.norm().item()

    cos_proxy_real = F.cosine_similarity(v_proxy.unsqueeze(0), v_real.unsqueeze(0)).item()
    cos_res = F.cosine_similarity(v_proxy_res.unsqueeze(0), v_real_res.unsqueeze(0)).item()

    print(f"  v_proxy L1={v_proxy.abs().sum():.4f}, v_real L1={v_real.abs().sum():.4f}", flush=True)
    print(f"  v_proxy_res norm={proxy_raw_norm:.4f}, v_real_res norm={real_raw_norm:.4f}", flush=True)
    print(f"  Cosine(proxy, real) in SAE space: {cos_proxy_real:.4f}", flush=True)
    print(f"  Cosine(proxy, real) in residual space: {cos_res:.4f}", flush=True)
    if v_layer is not None:
        if proxy_raw_norm > 1e-8:
            v_proxy_norm = v_proxy_res * (steer_norm / proxy_raw_norm)
        else:
            v_proxy_norm = v_proxy_res
        if real_raw_norm > 1e-8:
            v_real_norm = v_real_res * (steer_norm / real_raw_norm)
        else:
            v_real_norm = v_real_res
        cos_p = F.cosine_similarity(v_layer.unsqueeze(0), v_proxy_norm.unsqueeze(0)).item()
        cos_r = F.cosine_similarity(v_layer.unsqueeze(0), v_real_norm.unsqueeze(0)).item()
        print(f"  Cosine(dense_caa, proxy_res): {cos_p:.4f}", flush=True)
        print(f"  Cosine(dense_caa, real_res): {cos_r:.4f}", flush=True)

    # ====== TEST 4: Generate with proxy at 100 iters ======
    print("\n=== TEST 4: Full 100-iter proxy -> generate Q1 ===", flush=True)
    torch.cuda.empty_cache()
    v_proxy100, _ = optimize_v_steer(
        z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
        n_iter=100, lr=0.05, lambda_lm=0.5, beta=0.01,
        opt_device=dev,
    )
    v_proxy100_res = (W_dec.T @ v_proxy100.cpu().float()).float()
    raw_norm = v_proxy100_res.norm().item()
    if raw_norm > 1e-8:
        v_proxy100_res = v_proxy100_res * (steer_norm / raw_norm)

    bundle = __import__("app.persona.schemas", fromlist=["PersonaTraitArtifact"]).PersonaTraitArtifact
    bundle_obj = bundle.model_validate_json(Path(cfg["bundle"]).read_text())
    neg_sys = bundle_obj.neg_system_prompt
    q1 = bundle_obj.eval_questions[0]

    msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": q1}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
    attn = torch.ones_like(ids)

    d_vec = v_proxy100_res.to(device=dev, dtype=dtype).view(1, 1, -1)
    hook = _steering_hook_fn(DEFAULT_ALPHA, d_vec, steer_last_token_only=False, hook_calls=[0])
    handle = layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id)
    handle.remove()
    reply = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"  PROXY residual reply (first 300 chars):\n  {reply[:300]}", flush=True)

    # Now try with the cached real_lm result (100 iters from the test run)
    real_lm_v_path = cfg["sae_dir"] / f"ssv_real_lm_test_d{args.d}_l{layer}.json"
    print(f"\n  Check: proxy v_res norm after normalize = {v_proxy100_res.norm().item():.4f}", flush=True)
    print(f"  steer_norm (dense CAA) = {steer_norm:.4f}", flush=True)
    print(f"  DEFAULT_ALPHA = {DEFAULT_ALPHA}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
