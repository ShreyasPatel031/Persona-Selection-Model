#!/usr/bin/env python3
"""
Probe-to-decoder steering sweep for persona traits.

1. Collect SAE latents z from kept pos/neg rollout pairs at target layer.
2. Rank features by F-statistic (Good vs not-Good separability).
3. Train ridge probe on top-K_select features; optionally bootstrap-average weights.
4. Project truncated probe weights through SAE decoder -> v_steer at multiple K.
5. Steer + Vertex-judge on eval questions; compare trait vs K.

Follows SAE-SSV (F-stat subspace + supervised direction) and probe-to-decoder projection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.sae_common import load_rollout_question_pairs
from app.persona.sae_encode import assistant_hidden_span_at_layer, encode_hidden_span
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import DEFAULT_ALPHA, SAE_RELEASE, check_override, resolve_trait

DEFAULT_KS = "5,10,20,50,100,200,500"
DEFAULT_K_SELECT = 512  # probe subspace; must be >= max(K) in sweep
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def f_statistic_per_feature(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-way ANOVA F-statistic per feature dimension."""
    n, d = z.shape
    y = y.astype(np.float64)
    mask_pos = y > 0.5
    mask_neg = ~mask_pos
    n_pos = int(mask_pos.sum())
    n_neg = int(mask_neg.sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(f"Need >=2 samples per class; got pos={n_pos} neg={n_neg}")

    grand = z.mean(axis=0)
    mean_pos = z[mask_pos].mean(axis=0)
    mean_neg = z[mask_neg].mean(axis=0)

    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((z[mask_pos] - mean_pos) ** 2).sum(axis=0) + (
        (z[mask_neg] - mean_neg) ** 2
    ).sum(axis=0)
    denom = max(n - 2, 1)
    ms_within = ss_within / denom
    f = np.divide(
        ss_between,
        ms_within,
        out=np.zeros_like(ss_between),
        where=ms_within > 1e-12,
    )
    return f.astype(np.float64)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    p = X.shape[1]
    xt_x = X.T @ X
    reg = alpha * np.eye(p, dtype=np.float64)
    return np.linalg.solve(xt_x + reg, X.T @ y)


def ridge_cv_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
) -> tuple[float, np.ndarray, float]:
    """Pick ridge alpha by leave-one-out CV MSE."""
    n = X.shape[0]
    if n < 4:
        best_a = alphas[len(alphas) // 2]
        w = ridge_fit(X, y, best_a)
        pred = X @ w
        r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        return best_a, w, float(r2)

    best_a = alphas[0]
    best_mse = float("inf")
    best_w = ridge_fit(X, y, best_a)

    for alpha in alphas:
        mse_sum = 0.0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            w = ridge_fit(X[mask], y[mask], alpha)
            err = float(y[i] - X[i] @ w)
            mse_sum += err * err
        mse = mse_sum / n
        if mse < best_mse:
            best_mse = mse
            best_a = alpha
            best_w = ridge_fit(X, y, alpha)

    pred = X @ best_w
    r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return best_a, best_w, float(r2)


def bootstrap_probe_weights(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    acc = np.zeros(X.shape[1], dtype=np.float64)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        w = ridge_fit(X[idx], y[idx], alpha)
        acc += w
    return acc / n_boot


def build_probe_vector(
    W_dec: torch.Tensor,
    feature_ids: list[int],
    weights: list[float],
) -> torch.Tensor:
    v = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for fid, w in zip(feature_ids, weights):
        v += float(w) * W_dec[int(fid)].float()
    return v


def contrastive_delta_vector(
    W_dec: torch.Tensor,
    feature_ids: list[int],
    delta_z: np.ndarray,
) -> torch.Tensor:
    """Build steering vector from contrastive activation deltas projected through decoder.

    v = sum_i delta_z[i] * W_dec[i]  for features in feature_ids
    """
    v = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for fid in feature_ids:
        v += float(delta_z[fid]) * W_dec[int(fid)].float()
    return v


def collect_latents(
    model,
    tok,
    dev,
    sae,
    layer: int,
    pairs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    z_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for pi, pair in enumerate(pairs):
        for label, reply_key, sys_key in (
            (1, "pos_reply", "pos_system"),
            (0, "neg_reply", "neg_system"),
        ):
            system = str(pair[sys_key])
            question = str(pair["question"])
            reply = str(pair[reply_key])
            if len(reply.strip()) < 10:
                print(f"  skip pair {pi} label={label}: short reply", flush=True)
                continue
            try:
                h, _, _ = assistant_hidden_span_at_layer(
                    model,
                    tok,
                    dev,
                    system,
                    question,
                    reply,
                    layer,
                )
                _, z_mean = encode_hidden_span(sae, h)
                z_rows.append(z_mean.cpu().numpy().astype(np.float64))
                y_rows.append(label)
            except Exception as exc:
                print(f"  skip pair {pi} label={label}: {exc}", flush=True)

    if len(z_rows) < 8:
        raise RuntimeError(
            f"Too few latent samples ({len(z_rows)}); need >=8 for probe training"
        )
    return np.stack(z_rows, axis=0), np.array(y_rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--sae-id", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--ks", default=DEFAULT_KS)
    ap.add_argument(
        "--k-select",
        type=int,
        default=DEFAULT_K_SELECT,
        help="F-stat subspace size for ridge probe (must be >= max --ks)",
    )
    ap.add_argument("--n-bootstrap", type=int, default=20)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--no-norm-match", action="store_true", help="use fixed --alpha only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--z-cache", default=None, help="load/save collected z matrix (.npz)")
    ap.add_argument("--skip-judge", action="store_true")
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

    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha)
    layer = int(cfg["layer"])
    steer_alpha = args.alpha if args.alpha is not None else float(cfg["alpha"])
    bundle_path = Path(args.bundle or cfg["bundle"])
    vectors_path = Path(args.vectors or cfg["vectors"])
    rollouts_path = Path(
        args.rollouts or (cfg["base"] / "rollouts" / "rollouts.jsonl")
    )
    out_path = Path(
        args.out or (cfg["sae_dir"] / f"probe_steer_sweep_262k_l{layer}.json")
    )
    z_cache = Path(args.z_cache) if args.z_cache else (cfg["sae_dir"] / f"probe_z_cache_l{layer}.npz")
    sae_id = args.sae_id or cfg["sae_id"]
    hs_index = cfg["hs_index"]
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    k_select = int(args.k_select)
    max_k = max(ks) if ks else k_select
    if k_select < max_k:
        print(
            f"k_select={k_select} < max(ks)={max_k}; using k_select={max_k}",
            flush=True,
        )
        k_select = max_k

    bundle = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    v_layer = None
    if vectors_path.exists():
        v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
        v_layer = v_full[layer].float()

    print(
        f"=== Probe steer sweep trait={cfg['trait']} run={cfg['run_id']} "
        f"layer={layer} alpha={steer_alpha} ===",
        flush=True,
    )

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev,
        release=SAE_RELEASE,
        sae_id=sae_id,
        hidden_state_index=hs_index,
    )
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)
    k_select = min(k_select, d_sae)
    ks_to_run = sorted({k for k in ks if k <= k_select} | {k_select})
    print(f"ks_to_run={ks_to_run} (k_select={k_select} of {d_sae} SAE dims)", flush=True)

    if args.skip_collect and z_cache.exists():
        cached = np.load(z_cache)
        z_all = cached["z"]
        y_all = cached["y"]
        print(f"Loaded z cache: {z_all.shape[0]} samples from {z_cache}", flush=True)
    else:
        pairs = load_rollout_question_pairs(rollouts_path, bundle_path)
        print(f"Collecting latents from {len(pairs)} rollout pairs...", flush=True)
        z_all, y_all = collect_latents(model, tok, dev, sae, layer, pairs)
        z_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(z_cache, z=z_all, y=y_all)
        print(f"Saved z cache -> {z_cache} ({z_all.shape[0]} samples)", flush=True)

    print("Computing F-statistics...", flush=True)
    f_stats = f_statistic_per_feature(z_all, y_all)
    selected_idx = np.argsort(f_stats)[::-1][:k_select]
    selected_fids = selected_idx.astype(int).tolist()

    X = z_all[:, selected_idx]
    y = y_all
    ridge_alpha, w_probe, probe_r2 = ridge_cv_alpha(X, y, RIDGE_ALPHAS)
    if not args.no_bootstrap and args.n_bootstrap > 1:
        w_probe = bootstrap_probe_weights(
            X, y, ridge_alpha, n_boot=args.n_bootstrap, seed=42
        )
        pred = X @ w_probe
        probe_r2 = 1.0 - np.sum((y - pred) ** 2) / max(
            np.sum((y - y.mean()) ** 2), 1e-12
        )

    importance_order = np.argsort(np.abs(w_probe))[::-1]
    ranked_features = []
    for rank, j in enumerate(importance_order[:50], start=1):
        fid = int(selected_fids[j])
        ranked_features.append(
            {
                "rank": rank,
                "feature_id": fid,
                "f_statistic": round(float(f_stats[fid]), 6),
                "probe_weight": round(float(w_probe[j]), 6),
            }
        )

    print(
        f"Probe: ridge_alpha={ridge_alpha} r2={probe_r2:.4f} "
        f"k_select={k_select} (of {d_sae} SAE dims) "
        f"top feature={ranked_features[0]['feature_id']}",
        flush=True,
    )

    steer_norm = float(v_layer.norm()) if v_layer is not None else 1.0

    mask_pos = y_all > 0.5
    z_pos_mean = z_all[mask_pos].mean(axis=0)
    z_neg_mean = z_all[~mask_pos].mean(axis=0)
    delta_z = z_pos_mean - z_neg_mean  # (d_sae,) — positive = fires more for Good

    n_delta_pos = int((delta_z > 0).sum())
    n_delta_neg = int((delta_z < 0).sum())
    delta_order = np.argsort(np.abs(delta_z))[::-1]  # rank by |delta|
    delta_pos_order = np.array(
        [i for i in delta_order if delta_z[i] > 0], dtype=int
    )
    print(
        f"Contrastive delta: {n_delta_pos} features fire more for Good, "
        f"{n_delta_neg} fire more for not-Good",
        flush=True,
    )
    print(
        f"Top-5 delta features: {delta_order[:5].tolist()} "
        f"deltas={[round(float(delta_z[i]), 4) for i in delta_order[:5]]}",
        flush=True,
    )

    ranked_delta = []
    for rank, fid in enumerate(delta_order[:50], start=1):
        ranked_delta.append({
            "rank": rank,
            "feature_id": int(fid),
            "delta_z": round(float(delta_z[fid]), 6),
            "f_statistic": round(float(f_stats[fid]), 6),
        })

    def gen_text(ids, attn, hook_fn=None):
        handle = None
        if hook_fn is not None:
            handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True).strip()

    def judge_reply(prompt, reply):
        if args.skip_judge:
            return None
        if len(reply.strip()) < 20:
            return None
        try:
            js = score_transcript(judge_instr, neg_sys, prompt, reply)
            return int(js.score)
        except Exception as exc:
            print(f"  judge error: {exc}", flush=True)
            return None

    def run_condition(label, direction, alpha_eff, qs, meta=None):
        d = direction.to(device=dev, dtype=dtype).view(1, 1, -1)
        scores = []
        samples = []
        for qi, prompt in enumerate(qs):
            msgs = [
                {"role": "system", "content": neg_sys},
                {"role": "user", "content": prompt},
            ]
            enc = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook = _steering_hook_fn(
                alpha_eff, d, steer_last_token_only=False, hook_calls=[0]
            )
            reply = gen_text(ids, attn, hook)
            s = judge_reply(prompt, reply)
            scores.append(s)
            samples.append({"q_idx": qi, "score": s, "reply": reply[:300]})
            print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean} alpha_eff={alpha_eff:.3f}", flush=True)
        row = {
            "label": label,
            "mean": mean,
            "scores": scores,
            "alpha_effective": round(alpha_eff, 4),
            "samples": samples,
        }
        if meta:
            row.update(meta)
        return row

    results = []

    if not args.skip_ref:
        print("=== BASELINE ===", flush=True)
        base_scores = []
        for qi, prompt in enumerate(eval_qs):
            msgs = [
                {"role": "system", "content": neg_sys},
                {"role": "user", "content": prompt},
            ]
            enc = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            reply = gen_text(ids, attn)
            s = judge_reply(prompt, reply)
            base_scores.append(s)
            print(f"  [BASELINE] Q{qi+1} score={s}", flush=True)
        valid = [s for s in base_scores if s is not None]
        base_mean = round(sum(valid) / len(valid), 1) if valid else None
        results.append({"label": "BASELINE", "mean": base_mean, "scores": base_scores})

        if v_layer is not None:
            print("=== DENSE_CAA ===", flush=True)
            results.append(
                run_condition("DENSE_CAA", v_layer, steer_alpha, eval_qs)
            )

    for k in ks_to_run:
        label = f"PROBE_K{k}"
        print(f"=== {label} ===", flush=True)
        v_probe, fids, weights = probe_vectors[k]
        alpha_eff = steer_alpha
        meta = {
            "k": k,
            "n_features": len(fids),
            "feature_ids": fids[:20],
            "norm": round(float(v_probe.norm()), 4),
            "dense_norm": round(dense_norm, 4),
        }
        if v_layer is not None:
            cos = torch.nn.functional.cosine_similarity(
                v_layer.unsqueeze(0), v_probe.unsqueeze(0)
            ).item()
            meta["cosine_vs_dense"] = round(cos, 4)
            meta["norm_ratio_vs_dense"] = round(
                float(v_probe.norm() / v_layer.norm()), 4
            )
        row = run_condition(label, v_probe, alpha_eff, eval_qs, meta=meta)
        results.append(row)

    if v_layer is not None:
        print("\n--- Subspace projection (F-stat features, v_dense target) ---", flush=True)
        for k in ks_to_run:
            label = f"SUB_K{k}"
            fstat_fids = selected_fids[:k]
            v_sub, sub_coeffs = subspace_project(W_dec, fstat_fids, v_layer)
            raw_norm = float(v_sub.norm())
            if raw_norm > 1e-8 and not args.no_norm_match:
                v_sub = v_sub * (dense_norm / raw_norm)
            cos = torch.nn.functional.cosine_similarity(
                v_layer.unsqueeze(0), v_sub.unsqueeze(0)
            ).item()
            residual = float((v_layer.float() - v_sub * (raw_norm / dense_norm if dense_norm > 1e-8 else 1.0)).norm() / max(dense_norm, 1e-8))
            print(f"=== {label} === cos={cos:.4f} residual_frac={residual:.4f}", flush=True)
            meta = {
                "k": k,
                "method": "subspace_projection",
                "n_features": len(fstat_fids),
                "feature_ids": fstat_fids[:20],
                "norm_raw": round(raw_norm, 4),
                "norm_after_match": round(float(v_sub.norm()), 4),
                "dense_norm": round(dense_norm, 4),
                "cosine_vs_dense": round(cos, 4),
                "residual_frac": round(residual, 4),
            }
            row = run_condition(label, v_sub, steer_alpha, eval_qs, meta=meta)
            results.append(row)

    omp_path = cfg["sae_dir"] / f"omp_steer_results_262k_l{layer}.json"
    omp_summary = None
    if omp_path.exists():
        omp_data = json.loads(omp_path.read_text())
        if isinstance(omp_data, dict):
            omp_rows = omp_data.get("results", [])
        elif isinstance(omp_data, list):
            omp_rows = omp_data
        else:
            omp_rows = []
        omp_summary = [
            {"label": r.get("label"), "mean": r.get("mean"), "k": r.get("k")}
            for r in omp_rows
            if isinstance(r, dict)
        ]

    payload = {
        "method": "probe_to_decoder_sweep",
        "trait": cfg["trait"],
        "run_id": cfg["run_id"],
        "layer": layer,
        "sae_id": sae_id,
        "alpha_base": steer_alpha,
        "norm_match_to_dense": not args.no_norm_match,
        "n_questions": len(eval_qs),
        "data": {
            "n_samples": int(z_all.shape[0]),
            "n_pos": int((y_all > 0.5).sum()),
            "n_neg": int((y_all <= 0.5).sum()),
            "rollouts": str(rollouts_path),
            "z_cache": str(z_cache),
        },
        "probe": {
            "k_select": k_select,
            "d_sae_total": d_sae,
            "note": "Probe trained on top-k_select F-stat features only, not all SAE dims",
            "ridge_alpha": ridge_alpha,
            "r2": round(float(probe_r2), 4),
            "n_bootstrap": 0 if args.no_bootstrap else args.n_bootstrap,
        },
        "ranking_top50": ranked_features,
        "results": results,
        "omp_comparison": omp_summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        extra = ""
        if "k" in r:
            extra = f" k={r['k']} cos={r.get('cosine_vs_dense')}"
        print(f"  {r['label']:>12s}  mean={r['mean']}  scores={r['scores']}{extra}")
    if omp_summary:
        print("\n=== OMP reference ===", flush=True)
        for r in omp_summary:
            if r.get("label") in {"BASELINE", "DENSE_CAA"} or str(r.get("label", "")).startswith(
                ("OMP_K", "PROBE_")
            ):
                print(f"  {r.get('label'):>12s}  mean={r.get('mean')}")
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
