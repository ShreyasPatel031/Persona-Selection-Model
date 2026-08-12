#!/usr/bin/env python3
"""
Quick end-to-end test of Stage 2 (classifier-ranked features) vs raw F-stat.

Picks 6 d values, runs SSV optimize + generate + Vertex judge at each.
Compares classifier ranking vs F-stat ranking side by side.

Usage (on GPU VM):
  PYTHONPATH=. python -u scripts/ssv_stage2_test.py --trait good
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.sae_common import load_rollout_question_pairs
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import f_statistic_per_feature, optimize_v_steer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait

K_COARSE = 1024
M_CLASSIFIERS = 50
SUBSAMPLE_FRAC = 0.8
TEST_DS = [5, 20, 50, 100, 200, 500]


def classifier_rank(z_all, y_all, top_k_idx):
    """Stage 2: train M classifiers, return feature indices sorted by importance."""
    z_sub = z_all[:, top_k_idx]
    n = len(y_all)
    k = z_sub.shape[1]
    weight_sum = np.zeros(k, dtype=np.float64)

    for i in range(M_CLASSIFIERS):
        rng = np.random.RandomState(seed=i)
        idx = rng.choice(n, size=int(n * SUBSAMPLE_FRAC), replace=False)
        scaler = StandardScaler()
        X = scaler.fit_transform(z_sub[idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=i)
        clf.fit(X, y_all[idx])
        w = clf.coef_[0] if clf.classes_[1] > 0.5 else -clf.coef_[0]
        weight_sum += w
        if (i + 1) % 25 == 0:
            print(f"  classifiers: {i+1}/{M_CLASSIFIERS}", flush=True)

    v_avg = weight_sum / M_CLASSIFIERS
    rank = np.argsort(np.abs(v_avg))[::-1]
    return top_k_idx[rank]


def load_checkpoint(out_path: Path) -> tuple[list[dict], set[str]]:
    if not out_path.exists():
        return [], set()
    data = json.loads(out_path.read_text())
    results = list(data.get("results") or [])
    done = {str(r["label"]) for r in results if r.get("label")}
    return results, done


def save_checkpoint(
    out_path: Path,
    *,
    trait: str,
    layer: int,
    alpha: float,
    results: list[dict],
    judge_workers: int,
    checkpoint_source: str = "live",
) -> None:
    payload = {
        "trait": trait,
        "layer": layer,
        "alpha": alpha,
        "judge_workers": judge_workers,
        "checkpoint": len(results) < 1 or any(r.get("method") == "sae_ssv" for r in results),
        "checkpoint_source": checkpoint_source,
        "completed_labels": [r["label"] for r in results],
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    print(f"  checkpoint saved ({len(results)} conditions) -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--ds", default=",".join(str(d) for d in TEST_DS))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Steering alpha (default: from validation_report.json via resolve_trait).",
    )
    ap.add_argument(
        "--judge-workers",
        type=int,
        default=8,
        help="Parallel Vertex judge workers per steering condition.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip conditions already present in --out (incremental checkpoint).",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (unsteered) generation+judge.",
    )
    args = ap.parse_args()

    ds = sorted({int(x) for x in args.ds.split(",")})

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    alpha = float(args.alpha if args.alpha is not None else cfg["alpha"])
    print(
        f"Steering with alpha={alpha} layer={layer} "
        f"(alpha from {'CLI' if args.alpha is not None else cfg.get('alpha_source', 'validate')})",
        flush=True,
    )
    bundle_path = Path(cfg["bundle"])
    vectors_path = Path(cfg["vectors"])
    z_cache = cfg["sae_dir"] / f"probe_z_cache_l{layer}.npz"
    out_path = Path(args.out or (cfg["sae_dir"] / f"ssv_stage2_test_l{layer}.json"))
    judge_workers = max(1, int(args.judge_workers))

    results: list[dict] = []
    done: set[str] = set()
    if args.resume:
        results, done = load_checkpoint(out_path)
        if done:
            print(f"Resume: skipping {len(done)} completed conditions: {sorted(done)}", flush=True)

    bundle = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:args.n_questions]

    v_layer = None
    if vectors_path.exists():
        v_layer = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"][layer].float()
    steer_norm = float(v_layer.norm()) if v_layer is not None else 1.0

    # Load model
    print("Loading model...", flush=True)
    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    # Load SAE
    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(sae_dev, release=SAE_RELEASE, sae_id=cfg["sae_id"], hidden_state_index=cfg["hs_index"])
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)

    # Load z-cache
    cached = np.load(z_cache)
    z_all, y_all = cached["z"], cached["y"]
    print(f"Loaded z-cache: {z_all.shape[0]} samples", flush=True)

    # F-stat feature selection (paper: top-K candidates for optimization)
    f_stats = f_statistic_per_feature(z_all, y_all)
    fstat_ranked = np.argsort(f_stats)[::-1][:K_COARSE].copy()
    print(f"  Top-5 by F-stat: {fstat_ranked[:5].tolist()}", flush=True)

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()
    opt_dev = dev if dev.type == "cuda" else torch.device("cpu")

    # Helpers
    def gen_text(ids, attn, hook_fn=None):
        handle = layers[layer].register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True)
        if handle:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def judge_reply(prompt, reply):
        if len(reply.strip()) < 20:
            return None
        try:
            return int(score_transcript(judge_instr, neg_sys, prompt, reply).score)
        except Exception as e:
            print(f"  judge error: {e}", flush=True)
            return None

    print(f"Judging with {judge_workers} parallel workers", flush=True)

    def persist() -> None:
        save_checkpoint(
            out_path,
            trait=cfg["trait"],
            layer=layer,
            alpha=alpha,
            results=results,
            judge_workers=judge_workers,
        )

    def judge_all(label, prompts, replies):
        scores: list[int | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=judge_workers) as pool:
            futures = {
                pool.submit(judge_reply, prompt, reply): qi
                for qi, (prompt, reply) in enumerate(zip(prompts, replies))
            }
            for fut in as_completed(futures):
                qi = futures[fut]
                s = fut.result()
                scores[qi] = s
                print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean}", flush=True)
        return scores, mean

    def run_steer(label, direction):
        d = direction.to(device=dev, dtype=dtype).view(1, 1, -1)
        prompts, replies = [], []
        for prompt in eval_qs:
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook = _steering_hook_fn(alpha, d, steer_last_token_only=False, hook_calls=[0])
            reply = gen_text(ids, attn, hook)
            prompts.append(prompt)
            replies.append(reply)
        scores, mean = judge_all(label, prompts, replies)
        return {"label": label, "mean": mean, "scores": scores}

    # Baseline
    if "BASELINE" not in done and not args.skip_baseline:
        print("\n=== BASELINE ===", flush=True)
        base_prompts, base_replies = [], []
        for prompt in eval_qs:
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            reply = gen_text(ids, torch.ones_like(ids))
            base_prompts.append(prompt)
            base_replies.append(reply)
        base_scores, base_mean = judge_all("BASELINE", base_prompts, base_replies)
        results.append({"label": "BASELINE", "mean": base_mean, "scores": base_scores, "method": "baseline"})
        persist()

    # Dense CAA reference
    if v_layer is not None and "DENSE_CAA" not in done:
        print("\n=== DENSE_CAA ===", flush=True)
        r = run_steer("DENSE_CAA", v_layer)
        r["method"] = "dense_caa"
        results.append(r)
        persist()

    # Optimize once over K_COARSE F-stat candidates, then truncate to each d by |weight|
    full_fids = fstat_ranked[:K_COARSE].copy()
    full_mask = torch.zeros(d_sae, dtype=torch.float32)
    full_mask[full_fids] = 1.0

    print(f"\n=== Optimizing v over K={K_COARSE} (fstat) ===", flush=True)
    v_full, _ = optimize_v_steer(
        z_neg_means, W_dec, mu_pos, mu_neg, full_mask,
        n_iter=args.n_iter, lr=0.05, lambda_lm=0.5, beta=0.01,
        opt_device=opt_dev,
    )
    v_full_cpu = v_full.cpu().float()
    weight_order = torch.argsort(v_full_cpu.abs(), descending=True)
    n_nonzero = int((v_full_cpu.abs() > 1e-8).sum())
    print(f"  fstat K={K_COARSE}: {n_nonzero} nonzero weights after L1 opt", flush=True)

    for d in ds:
        label = f"d{d}"
        if label in done:
            print(f"\n=== {label} (skipped, checkpoint) ===", flush=True)
            continue
        print(f"\n=== {label} ===", flush=True)

        top_d_idx = weight_order[:d]
        v_trunc = torch.zeros_like(v_full_cpu)
        v_trunc[top_d_idx] = v_full_cpu[top_d_idx]

        active = top_d_idx[v_trunc[top_d_idx].abs() > 1e-8]
        fids = active.tolist()
        weights = [round(float(v_trunc[i]), 4) for i in active.tolist()]

        v_res = (W_dec.T @ v_trunc).float()
        raw_norm = float(v_res.norm())
        if raw_norm > 1e-8:
            v_res = v_res * (steer_norm / raw_norm)

        cos_dense = None
        if v_layer is not None:
            cos_dense = round(F.cosine_similarity(v_layer.unsqueeze(0), v_res.unsqueeze(0)).item(), 4)

        print(f"  CHECKPOINT: {label} cosine_vs_dense={cos_dense} n_active={len(fids)}", flush=True)
        print(f"  CHECKPOINT: top fids={fids[:10]} weights={weights[:10]}", flush=True)

        r = run_steer(label, v_res)
        r.update({
            "method": "sae_ssv",
            "ranking": "fstat",
            "d": d,
            "feature_ids": fids,
            "feature_weights": weights,
            "n_active": len(fids),
            "norm_raw": round(raw_norm, 4),
            "cosine_vs_dense": cos_dense,
        })
        results.append(r)
        persist()

    # Summary
    print("\n" + "=" * 60)
    print(f"{'label':>20s}  {'mean':>5s}  {'d':>4s}  {'cos':>6s}  scores")
    print("-" * 60)
    for r in results:
        cos = r.get("cosine_vs_dense", "")
        cos_s = f"{cos:.3f}" if isinstance(cos, float) else ""
        d_s = str(r.get("d", "")) if "d" in r else ""
        print(f"  {r['label']:>18s}  {str(r['mean']):>5s}  {d_s:>4s}  {cos_s:>6s}  {r['scores']}")

    persist()
    final = json.loads(out_path.read_text())
    final["checkpoint"] = False
    final.pop("checkpoint_source", None)
    out_path.write_text(json.dumps(final, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
