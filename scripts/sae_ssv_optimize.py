#!/usr/bin/env python3
"""
SAE-SSV: Supervised Steering Vector optimization in SAE latent space.

Implements the L_steer objective from He et al. (EMNLP 2025):
  L_steer = ||z' - mu+||^2 - ||z' - mu-||^2 + lambda_lm * L_LM + beta * ||v_I||_1

where z' = z + v, with v constrained to be nonzero only on d_steer
F-stat-selected dimensions.

After optimization, the steering vector is decoded back to residual stream
via W_dec and evaluated by generating + judging at multiple scaling factors.
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

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.sae_common import load_rollout_question_pairs
from app.persona.sae_encode import assistant_hidden_span_at_layer
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import DEFAULT_ALPHA, SAE_RELEASE, resolve_trait


def f_statistic_per_feature(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask_pos = y > 0.5
    mask_neg = ~mask_pos
    n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())
    n = n_pos + n_neg
    grand = z.mean(axis=0)
    mean_pos = z[mask_pos].mean(axis=0)
    mean_neg = z[mask_neg].mean(axis=0)
    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((z[mask_pos] - mean_pos) ** 2).sum(axis=0) + (
        (z[mask_neg] - mean_neg) ** 2
    ).sum(axis=0)
    ms_within = ss_within / max(n - 2, 1)
    f = np.divide(ss_between, ms_within, out=np.zeros_like(ss_between), where=ms_within > 1e-12)
    return f.astype(np.float64)


def collect_latents(
    model, tok, dev, sae, layer: int, pairs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect mean SAE latents from rollout pairs.

    Returns z_all (n, d_sae), y_all (n,).
    """
    z_rows, y_rows = [], []

    for pi, pair in enumerate(pairs):
        for label, reply_key, sys_key in (
            (1, "pos_reply", "pos_system"),
            (0, "neg_reply", "neg_system"),
        ):
            system = str(pair[sys_key])
            question = str(pair["question"])
            reply = str(pair[reply_key])
            if len(reply.strip()) < 10:
                continue
            try:
                h, _, _ = assistant_hidden_span_at_layer(
                    model, tok, dev, system, question, reply, layer,
                )
                sae_dev = next(sae.parameters()).device
                x = h.unsqueeze(0).to(sae_dev)
                with torch.no_grad():
                    z = sae.encode(x)[0].float()
                z_mean = z.mean(dim=0)
                z_rows.append(z_mean.cpu().numpy().astype(np.float64))
                y_rows.append(label)
            except Exception as exc:
                print(f"  skip pair {pi} label={label}: {exc}", flush=True)

    if len(z_rows) < 8:
        raise RuntimeError(f"Too few samples ({len(z_rows)}); need >=8")
    return np.stack(z_rows, axis=0), np.array(y_rows, dtype=np.float64)


def optimize_v_steer(
    z_neg_means: torch.Tensor,
    W_dec: torch.Tensor,
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    n_iter: int = 100,
    lr: float = 0.05,
    lambda_lm: float = 0.5,
    beta: float = 0.01,
    opt_device: torch.device,
) -> torch.Tensor:
    """Optimize steering vector v in SAE space using L_steer.

    Pre-encodes neg samples to z_neg_means (n_neg, d_sae) so each iteration
    is just vector math — no SAE forward pass in the loop.

    L_steer = ||z' - mu+||^2 - ||z' - mu-||^2 + lambda_lm * ||W_dec @ v||^2 + beta * ||v_I||_1
    where z' = z_neg_mean + v for each negative sample.

    The LM regularizer ||W_dec @ v||^2 penalizes large residual-stream perturbations
    (proxy for generation quality — same spirit as the paper's L_LM but differentiable
    without a full model forward pass).
    """
    d_sae = z_neg_means.shape[1]
    n_neg = z_neg_means.shape[0]
    mask_cpu = feature_mask.float()
    mask = mask_cpu.to(opt_device)

    z_neg = z_neg_means.to(opt_device).float()
    mu_p = mu_pos.to(opt_device).float()
    mu_n = mu_neg.to(opt_device).float()

    W_dec_I = W_dec[mask_cpu.bool()].to(opt_device).float()  # (k, d_model)

    delta = (mu_p - mu_n) * mask
    delta_norm = delta.norm()
    if delta_norm > 1e-8:
        delta = delta / delta_norm
    v = delta.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([v], lr=lr)

    print(
        f"  Optimizing v: {int(mask.sum())} active dims, "
        f"{n_neg} neg samples, {n_iter} iters (device={opt_device})",
        flush=True,
    )

    for it in range(n_iter):
        optimizer.zero_grad()

        v_masked = v * mask  # (d_sae,)

        z_prime = z_neg + v_masked.unsqueeze(0)  # (n_neg, d_sae)

        dist_pos = (z_prime - mu_p.unsqueeze(0)).pow(2).sum(dim=1).mean()
        dist_neg = (z_prime - mu_n.unsqueeze(0)).pow(2).sum(dim=1).mean()
        dist_loss = dist_pos - dist_neg

        v_active = v_masked[mask.bool()]  # (k,)
        residual_perturbation = (W_dec_I.T @ v_active).pow(2).sum()
        lm_loss = residual_perturbation

        sparsity = v_masked.abs().sum()

        loss = dist_loss + lambda_lm * lm_loss + beta * sparsity

        loss.backward()

        with torch.no_grad():
            if v.grad is not None:
                v.grad *= mask

        optimizer.step()

        with torch.no_grad():
            v.data *= mask

        if (it + 1) % 20 == 0 or it == 0:
            print(
                f"    iter {it+1}/{n_iter}  loss={loss.item():.4f}  "
                f"dist={dist_loss.item():.4f}  "
                f"lm={lm_loss.item():.4f}  "
                f"sparsity={beta * sparsity.item():.4f}  "
                f"v_nnz={int((v_masked.abs() > 1e-8).sum())}",
                flush=True,
            )

    return v.detach() * mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--sae-id", default=None)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--ks", default="5,10,20,50,100,128,200,256,512,750,1000")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lambda-lm", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--z-cache", default=None)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--optimize-only", action="store_true", help="Only run SSV optimization; skip generation/judging")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.layer is not None:
        cfg["layer"] = args.layer
        from scripts.trait_sae_config import hidden_state_index, run_paths, sae_id_for_layer
        cfg["sae_id"] = args.sae_id or sae_id_for_layer(cfg["layer"])
        cfg["hs_index"] = hidden_state_index(cfg["layer"])
        cfg.update(run_paths(cfg["run_id"], cfg["layer"]))

    layer = int(cfg["layer"])
    bundle_path = Path(args.bundle or cfg["bundle"])
    vectors_path = Path(args.vectors or cfg["vectors"])
    rollouts_path = Path(args.rollouts or (cfg["base"] / "rollouts" / "rollouts.jsonl"))
    out_path = Path(args.out or (cfg["sae_dir"] / f"sae_ssv_results_262k_l{layer}.json"))
    z_cache = Path(args.z_cache) if args.z_cache else (cfg["sae_dir"] / f"probe_z_cache_l{layer}.npz")
    sae_id = args.sae_id or cfg["sae_id"]
    hs_index = cfg["hs_index"]
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})

    bundle = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    v_layer = None
    if vectors_path.exists():
        v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
        v_layer = v_full[layer].float()

    print(
        f"=== SAE-SSV optimize  trait={cfg['trait']} run={cfg['run_id']} "
        f"layer={layer} n_iter={args.n_iter} lr={args.lr} ===",
        flush=True,
    )

    need_model = not (args.optimize_only and args.skip_collect and z_cache.exists())
    model = tok = None
    layers = pad_id = dtype = dev = None

    if need_model:
        model, tok, dev = load_model_and_tokenizer()
        layers = _language_model_layers(model)
        pad_id = tok.pad_token_id or tok.eos_token_id
        dtype = next(model.parameters()).dtype
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Skipping model load (optimize-only + z cache present)", flush=True)

    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev, release=SAE_RELEASE, sae_id=sae_id, hidden_state_index=hs_index,
    )
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)

    pairs = load_rollout_question_pairs(rollouts_path, bundle_path) if need_model else []

    if args.skip_collect and z_cache.exists():
        cached = np.load(z_cache)
        z_all, y_all = cached["z"], cached["y"]
        print(f"Loaded z cache: {z_all.shape[0]} samples", flush=True)
    else:
        if model is None:
            raise RuntimeError("z cache missing and model not loaded; run without --skip-collect first")
        print(f"Collecting latents from {len(pairs)} pairs...", flush=True)
        z_all, y_all = collect_latents(model, tok, dev, sae, layer, pairs)
        z_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(z_cache, z=z_all, y=y_all)
        print(f"Saved z cache ({z_all.shape[0]} samples)", flush=True)

    print("Computing F-statistics...", flush=True)
    f_stats = f_statistic_per_feature(z_all, y_all)

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()

    opt_dev = dev if dev is not None and dev.type == "cuda" else torch.device("cpu")
    print(f"Optimization device: {opt_dev}, neg samples: {z_neg_means.shape[0]}", flush=True)

    steer_norm = float(v_layer.norm()) if v_layer is not None else 1.0

    if args.optimize_only:
        # --- SAE-SSV optimization only (no model generation) ---
        results = []
        for k in ks:
            k = min(k, d_sae)
            print(f"\n=== SAE-SSV K={k} ===", flush=True)
            top_k_idx = np.argsort(f_stats)[::-1][:k].copy()
            feature_mask = torch.zeros(d_sae, dtype=torch.float32)
            feature_mask[top_k_idx] = 1.0
            v_opt = optimize_v_steer(
                z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
                n_iter=args.n_iter, lr=args.lr,
                lambda_lm=args.lambda_lm, beta=args.beta,
                opt_device=opt_dev,
            )
            v_residual = (W_dec.T @ v_opt.cpu().float()).float()
            raw_norm = float(v_residual.norm())
            if raw_norm > 1e-8:
                v_residual = v_residual * (steer_norm / raw_norm)
            active_mask = v_opt.abs() > 1e-8
            active_fids = int(active_mask.sum())
            top_active = torch.argsort(v_opt.abs(), descending=True)
            top_fids = [int(f) for f in top_active if active_mask[f]]
            top_weights = [round(float(v_opt[f]), 6) for f in top_active if active_mask[f]]
            meta = {
                "k": k,
                "method": "sae_ssv",
                "n_active_features": active_fids,
                "feature_ids": top_fids,
                "feature_weights": top_weights,
                "norm_raw": round(raw_norm, 4),
                "norm_after_scale": round(float(v_residual.norm()), 4),
                "steer_norm": round(steer_norm, 4),
            }
            if v_layer is not None:
                cos = F.cosine_similarity(v_layer.unsqueeze(0), v_residual.unsqueeze(0)).item()
                meta["cosine_vs_dense"] = round(cos, 4)
            print(
                f"  Optimized: {active_fids} active features, saved {len(top_fids)} ids",
                flush=True,
            )
            results.append({"label": f"SSV_K{k}", "mean": None, "scores": [], **meta})

        payload = {
            "method": "sae_ssv",
            "trait": cfg["trait"],
            "run_id": cfg["run_id"],
            "layer": layer,
            "sae_id": sae_id,
            "optim": {
                "n_iter": args.n_iter,
                "lr": args.lr,
                "lambda_lm": args.lambda_lm,
                "beta": args.beta,
            },
            "data": {
                "n_samples": int(z_all.shape[0]),
                "n_pos": int((y_all > 0.5).sum()),
                "n_neg": int((y_all <= 0.5).sum()),
            },
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved {out_path}", flush=True)
        return

    # --- Generation & judging helpers ---

    def gen_text(ids, attn, hook_fn=None):
        handle = None
        if hook_fn is not None:
            handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=pad_id, use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def judge_reply(prompt, reply):
        if args.skip_judge or len(reply.strip()) < 20:
            return None
        try:
            js = score_transcript(judge_instr, neg_sys, prompt, reply)
            return int(js.score)
        except Exception as exc:
            print(f"  judge error: {exc}", flush=True)
            return None

    def run_condition(label, direction, alpha_eff, qs, meta=None):
        d = direction.to(device=dev, dtype=dtype).view(1, 1, -1)
        scores, samples = [], []
        for qi, prompt in enumerate(qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook = _steering_hook_fn(alpha_eff, d, steer_last_token_only=False, hook_calls=[0])
            reply = gen_text(ids, attn, hook)
            s = judge_reply(prompt, reply)
            scores.append(s)
            samples.append({"q_idx": qi, "score": s, "reply": reply[:300]})
            print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean} alpha_eff={alpha_eff:.3f}", flush=True)
        row = {"label": label, "mean": mean, "scores": scores, "alpha_effective": round(alpha_eff, 4), "samples": samples}
        if meta:
            row.update(meta)
        return row

    results = []

    # --- Baseline & Dense CAA ---
    if not args.skip_ref:
        print("=== BASELINE ===", flush=True)
        base_scores = []
        for qi, prompt in enumerate(eval_qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            reply = gen_text(ids, attn)
            s = judge_reply(prompt, reply)
            base_scores.append(s)
            print(f"  [BASELINE] Q{qi+1} score={s}", flush=True)
        valid = [s for s in base_scores if s is not None]
        results.append({"label": "BASELINE", "mean": round(sum(valid)/len(valid), 1) if valid else None, "scores": base_scores})

        if v_layer is not None:
            print("=== DENSE_CAA ===", flush=True)
            results.append(run_condition("DENSE_CAA", v_layer, args.alpha, eval_qs))

    # --- SAE-SSV optimization at each K ---
    for k in ks:
        k = min(k, d_sae)
        print(f"\n=== SAE-SSV K={k} ===", flush=True)

        top_k_idx = np.argsort(f_stats)[::-1][:k].copy()
        feature_mask = torch.zeros(d_sae, dtype=torch.float32)
        feature_mask[top_k_idx] = 1.0

        v_opt = optimize_v_steer(
            z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
            n_iter=args.n_iter, lr=args.lr,
            lambda_lm=args.lambda_lm, beta=args.beta,
            opt_device=opt_dev,
        )

        v_residual = (W_dec.T @ v_opt.cpu().float()).float()
        raw_norm = float(v_residual.norm())
        if raw_norm > 1e-8:
            v_residual = v_residual * (steer_norm / raw_norm)

        active_mask = v_opt.abs() > 1e-8
        active_fids = int(active_mask.sum())
        top_active = torch.argsort(v_opt.abs(), descending=True)
        top_fids = [int(f) for f in top_active if active_mask[f]]
        top_weights = [round(float(v_opt[f]), 6) for f in top_active if active_mask[f]]

        meta = {
            "k": k,
            "method": "sae_ssv",
            "n_active_features": active_fids,
            "feature_ids": top_fids,
            "feature_weights": top_weights,
            "norm_raw": round(raw_norm, 4),
            "norm_after_scale": round(float(v_residual.norm()), 4),
            "steer_norm": round(steer_norm, 4),
        }
        if v_layer is not None:
            cos = F.cosine_similarity(v_layer.unsqueeze(0), v_residual.unsqueeze(0)).item()
            meta["cosine_vs_dense"] = round(cos, 4)

        label = f"SSV_K{k}"
        print(
            f"  Optimized: {active_fids} active features, norm={raw_norm:.4f}, "
            f"cos_dense={meta.get('cosine_vs_dense', 'N/A')}",
            flush=True,
        )
        if args.optimize_only:
            row = {"label": label, "mean": None, "scores": [], "alpha_effective": args.alpha, **meta}
            results.append(row)
        else:
            row = run_condition(label, v_residual, args.alpha, eval_qs, meta=meta)
            results.append(row)

    # --- Save ---
    payload = {
        "method": "sae_ssv",
        "trait": cfg["trait"],
        "run_id": cfg["run_id"],
        "layer": layer,
        "sae_id": sae_id,
        "alpha_base": args.alpha,
        "n_questions": len(eval_qs),
        "optim": {
            "n_iter": args.n_iter,
            "lr": args.lr,
            "lambda_lm": args.lambda_lm,
            "beta": args.beta,
        },
        "data": {
            "n_samples": int(z_all.shape[0]),
            "n_pos": int((y_all > 0.5).sum()),
            "n_neg": int((y_all <= 0.5).sum()),
        },
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        extra = ""
        if "k" in r:
            extra = f" k={r['k']} cos={r.get('cosine_vs_dense', 'N/A')}"
        print(f"  {r['label']:>12s}  mean={r['mean']}  scores={r['scores']}{extra}")
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
