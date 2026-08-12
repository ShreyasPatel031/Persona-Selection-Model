#!/usr/bin/env python3
"""
Compare proxy L_LM vs real L_LM at d=50 with classifier-ranked features.

Usage (GPU VM):
  PYTHONPATH=. python -u scripts/ssv_real_lm_test.py --trait good
"""
from __future__ import annotations

import argparse
import json
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
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import (
    collect_lm_cache,
    f_statistic_per_feature,
    optimize_v_steer,
)
from scripts.trait_sae_config import DEFAULT_ALPHA, SAE_RELEASE, resolve_trait

K_COARSE = 1024
D_TEST = 50
M_CLASSIFIERS = 50


def classifier_rank(z_all, y_all, top_k_idx):
    z_sub = z_all[:, top_k_idx]
    n = len(y_all)
    k = z_sub.shape[1]
    weight_sum = np.zeros(k, dtype=np.float64)
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
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--lm-batch-size", type=int, default=1)
    ap.add_argument("--skip-proxy", action="store_true", help="Skip proxy run (e.g. after OOM on real_lm retry)")
    ap.add_argument("--normalize-dist", action="store_true", help="Per-dim MSE distance (divides by d_sae)")
    ap.add_argument("--lambda-lm", type=float, default=0.5)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:5]

    vectors_path = Path(cfg["vectors"])
    v_layer = None
    if vectors_path.exists():
        v_layer = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"][layer].float()
    steer_norm = float(v_layer.norm()) if v_layer is not None else 1.0

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
    ranked = classifier_rank(z_all, y_all, top_k)[: args.d]

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()

    if lm_cache_path.exists():
        lm_cache = torch.load(lm_cache_path, map_location="cpu", weights_only=False)
        print(f"Loaded LM cache: {len(lm_cache['neg_input_ids'])} samples", flush=True)
    else:
        from app.persona.sae_common import load_rollout_question_pairs
        pairs = load_rollout_question_pairs(cfg["base"] / "rollouts" / "rollouts.jsonl", cfg["bundle"])
        lm_cache = collect_lm_cache(tok, pairs)
        torch.save(lm_cache, lm_cache_path)

    feature_mask = torch.zeros(int(z_all.shape[1]), dtype=torch.float32)
    feature_mask[ranked] = 1.0

    sae, _ = load_sae_for_layer(
        torch.device("cpu"), release=SAE_RELEASE, sae_id=cfg["sae_id"], hidden_state_index=cfg["hs_index"],
    )
    W_dec = sae.W_dec.detach().float().cpu()

    def run_steer_residual(label, v_res):
        d = v_res.to(device=dev, dtype=dtype).view(1, 1, -1)
        scores = []
        for qi, prompt in enumerate(eval_qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook = _steering_hook_fn(DEFAULT_ALPHA, d, steer_last_token_only=False, hook_calls=[0])
            handle = layers[layer].register_forward_hook(hook)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id)
            handle.remove()
            reply = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            s = int(score_transcript(judge_instr, neg_sys, prompt, reply).score) if len(reply) >= 20 else None
            scores.append(s)
            print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean}", flush=True)
        return mean, scores

    def run_steer_sae_hook(label, v_opt):
        """Eval using SAE encode+steer+decode hook — steer ALL positions during generation."""
        from scripts.sae_ssv_optimize import sae_steer_hook_fn
        v_masked = v_opt.to(dev).float()
        scores = []
        for qi, prompt in enumerate(eval_qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook_fn = sae_steer_hook_fn(sae, v_masked, prompt_len=0)
            handle = layers[layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id)
            handle.remove()
            reply = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            s = int(score_transcript(judge_instr, neg_sys, prompt, reply).score) if len(reply) >= 20 else None
            scores.append(s)
            print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean}", flush=True)
        return mean, scores

    results = []
    for use_real, tag in [(False, "proxy"), (True, "real_lm")]:
        if args.skip_proxy and not use_real:
            continue
        print(f"\n=== d={args.d} classifier {tag} ===", flush=True)
        torch.cuda.empty_cache()
        v_opt, _ = optimize_v_steer(
            z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
            n_iter=args.n_iter, lr=0.05, lambda_lm=args.lambda_lm, beta=0.01,
            opt_device=dev,
            model=model if use_real else None,
            sae=sae if use_real else None,
            layers=layers if use_real else None,
            lm_cache=lm_cache if use_real else None,
            lm_batch_size=1,
            lm_max_tokens=64,
            steering_layer=layer,
            use_real_lm=use_real,
            normalize_dist=args.normalize_dist if use_real else False,
        )
        v_res = (W_dec.T @ v_opt.cpu().float()).float()
        raw_norm = float(v_res.norm())
        if raw_norm > 1e-8:
            v_res = v_res * (steer_norm / raw_norm)
        cos = round(F.cosine_similarity(v_layer.unsqueeze(0), v_res.unsqueeze(0)).item(), 4) if v_layer is not None else None
        if use_real:
            mean_sae, scores_sae = run_steer_sae_hook(f"d{args.d}_{tag}_sae_hook", v_opt)
            mean_res, scores_res = run_steer_residual(f"d{args.d}_{tag}_residual", v_res)
            results.append({
                "tag": tag, "eval": "sae_hook", "mean": mean_sae, "scores": scores_sae,
                "cosine_vs_dense": cos, "use_real_lm": use_real,
            })
            results.append({
                "tag": tag, "eval": "residual", "mean": mean_res, "scores": scores_res,
                "cosine_vs_dense": cos, "use_real_lm": use_real,
            })
        else:
            mean, scores = run_steer_residual(f"d{args.d}_{tag}", v_res)
            results.append({"tag": tag, "eval": "residual", "mean": mean, "scores": scores, "cosine_vs_dense": cos, "use_real_lm": use_real})

    print("\n=== COMPARISON ===", flush=True)
    for r in results:
        ev = r.get("eval", "")
        print(f"  {r['tag']:>8s} {ev:>10s}  mean={r['mean']}  cos={r['cosine_vs_dense']}  scores={r['scores']}", flush=True)

    out = cfg["sae_dir"] / f"ssv_real_lm_test_d{args.d}_l{layer}.json"
    out.write_text(json.dumps({"trait": cfg["trait"], "d": args.d, "ranking": "classifier", "results": results}, indent=2))
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
