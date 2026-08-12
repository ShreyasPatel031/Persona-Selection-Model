#!/usr/bin/env python3
"""
SSV (F-stat + optimize_v_steer) + residual-stream d-sweep.

For each d in the sweep:
  1. Optimize v once over top-1024 F-stat candidates (L_steer + L1)
  2. Truncate to top-d features by |weight|
  3. Decode v_trunc via W_dec, norm-match to dense CAA, add alpha*scale to residual
  4. Calibrate scale on Q1 at scales [1,2,3,5,8] (stop early if score >= 90)
  5. Full judge on N questions at best scale
  6. Early stop d-sweep after 3 consecutive d values with mean >= 90

Usage (GPU VM):
  PYTHONPATH=. python -u scripts/ssv_dsweep.py --trait good
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
from app.persona.sae_common import load_all_rollout_samples
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import (
    collect_latents_samples,
    f_statistic_per_feature,
    optimize_v_steer,
)
from scripts.trait_sae_config import (
    SAE_RELEASE,
    hidden_state_index,
    resolve_trait,
    sae_id_for_layer,
)

K_COARSE = 1024
DEFAULT_DS = [5, 10, 20, 50, 100]
DEFAULT_SCALES = [1.0, 2.0, 3.0, 5.0, 8.0]
EARLY_STOP_THRESHOLD = 90
EARLY_STOP_CONSECUTIVE = 3
GATE_A_MIN_SCORE = 30
GATE_B_MIN_COSINE = 0.2
GATE_B_DS = [5, 10, 20]
GATE_C_MIN_SCORE = 1


def load_checkpoint(out_path: Path) -> tuple[list[dict], set[int], dict]:
    if not out_path.exists():
        return [], set(), {}
    data = json.loads(out_path.read_text())
    results = list(data.get("results") or [])
    done = {int(r["d"]) for r in results if r.get("d") is not None}
    meta = {
        "ssv_features": list(data.get("ssv_features") or []),
        "ssv_weights": list(data.get("ssv_weights") or []),
        "fstat_top5": list(data.get("fstat_top5") or []),
        "gates": data.get("gates"),
        "gates_passed": data.get("gates_passed"),
    }
    return results, done, meta


def save_checkpoint(
    out_path: Path,
    *,
    payload: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    n = len(payload.get("results") or [])
    print(f"  checkpoint saved ({n} d-values) -> {out_path}", flush=True)


def build_v_sae(
    feature_ids: list[int],
    feature_weights: list[float],
    d_sae: int,
) -> torch.Tensor:
    v = torch.zeros(d_sae, dtype=torch.float32)
    for fid, w in zip(feature_ids, feature_weights):
        v[int(fid)] = float(w)
    return v


def truncate_v_by_weight(
    v_full: torch.Tensor,
    d: int,
) -> tuple[list[int], list[float], torch.Tensor]:
    weight_order = torch.argsort(v_full.abs(), descending=True)
    top_d_idx = weight_order[:d]
    v_trunc = torch.zeros_like(v_full)
    v_trunc[top_d_idx] = v_full[top_d_idx]
    active = top_d_idx[v_trunc[top_d_idx].abs() > 1e-8]
    fids = active.tolist()
    weights = [round(float(v_trunc[i]), 4) for i in active.tolist()]
    return fids, weights, v_trunc


def cosine_vs_dense(v_dense: torch.Tensor, v_sae: torch.Tensor, W_dec: torch.Tensor) -> float:
    recon = (W_dec.T @ v_sae).float()
    return round(
        F.cosine_similarity(v_dense.unsqueeze(0), recon.unsqueeze(0)).item(),
        4,
    )


def residual_direction_from_v_trunc(
    v_trunc: torch.Tensor,
    W_dec: torch.Tensor,
    steer_norm: float,
    *,
    dev: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    v_res = (W_dec.T @ v_trunc).float()
    raw_norm = float(v_res.norm())
    if raw_norm > 1e-8:
        v_res = v_res * (steer_norm / raw_norm)
    direction = v_res.to(device=dev, dtype=dtype).view(1, 1, -1)
    return direction, raw_norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--ds", default=",".join(str(d) for d in DEFAULT_DS))
    ap.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge-workers", type=int, default=16)
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override layer (default: from validation_report.json via resolve_trait).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Steering alpha (default: from validation_report.json via resolve_trait).",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--lambda-lm",
        type=float,
        default=0.0,
        help="LM proxy penalty weight (default 0 = no penalty; 0.5 was old default).",
    )
    ap.add_argument(
        "--early-stop-threshold",
        type=float,
        default=EARLY_STOP_THRESHOLD,
        help="Mean trait score to count toward consecutive early-stop.",
    )
    ap.add_argument(
        "--early-stop-consecutive",
        type=int,
        default=EARLY_STOP_CONSECUTIVE,
        help="Stop after this many consecutive d values at/above threshold.",
    )
    ap.add_argument(
        "--scale-stop-threshold",
        type=float,
        default=EARLY_STOP_THRESHOLD,
        help="Stop scale sweep early when Q1 score reaches this.",
    )
    ap.add_argument(
        "--z-cache",
        default=None,
        help="Override z-cache path (default: probe_z_cache_l{layer}.npz or full variant).",
    )
    ap.add_argument(
        "--build-full-cache",
        action="store_true",
        help="Use/build probe_z_cache_full_l{layer}.npz from all kept rollouts.",
    )
    ap.add_argument(
        "--gate-a-min",
        type=float,
        default=GATE_A_MIN_SCORE,
        help="Gate A: min dense CAA Q1 score.",
    )
    ap.add_argument(
        "--gate-b-min-cosine",
        type=float,
        default=GATE_B_MIN_COSINE,
        help="Gate B: min max cosine vs dense at d=5/10/20.",
    )
    ap.add_argument(
        "--skip-gates",
        action="store_true",
        help="Skip pre-gates (e.g. resume after gates already passed).",
    )
    ap.add_argument(
        "--beta", type=float, default=0.01,
        help="L1 sparsity penalty (default 0.01).",
    )
    ap.add_argument(
        "--beta-auto", type=float, default=0.0,
        help="If > 0, auto-scale beta so L1 term = beta_auto * dist_loss at iter 0. "
             "Overrides --beta. Try 0.1-0.3 to force peaked weight distribution.",
    )
    ap.add_argument(
        "--reopt-iters", type=int, default=0,
        help="Phase-2 re-optimization iterations per d value. After truncating to "
             "top-d features, re-run optimizer with only those d features unmasked. "
             "0 = disabled (default). Try 100.",
    )
    args = ap.parse_args()

    ds = sorted({int(x) for x in args.ds.split(",") if x.strip()})
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    if not ds:
        raise SystemExit("No d values specified")
    if not scales:
        raise SystemExit("No scales specified")

    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    if args.layer is not None:
        cfg["layer"] = layer
        cfg["sae_id"] = sae_id_for_layer(layer)
        cfg["hs_index"] = hidden_state_index(layer)
        cfg["layer_source"] = "cli_override"
    alpha = float(args.alpha if args.alpha is not None else cfg["alpha"])
    use_full_cache = bool(args.build_full_cache)
    z_cache = Path(
        args.z_cache
        or (
            cfg["sae_dir"] / (
                f"probe_z_cache_full_l{layer}.npz"
                if use_full_cache
                else f"probe_z_cache_l{layer}.npz"
            )
        )
    )
    out_path = Path(
        args.out
        or (
            cfg["sae_dir"] / (
                f"ssv_dsweep_residual_full_l{layer}.json"
                if use_full_cache
                else f"ssv_dsweep_residual_l{layer}.json"
            )
        )
    )
    rollouts_path = cfg["base"] / "rollouts" / "rollouts.jsonl"
    judge_workers = max(1, int(args.judge_workers))
    print(
        f"Steering with alpha={alpha} layer={layer} "
        f"(layer from {cfg.get('layer_source', 'validate')}; "
        f"alpha from {'CLI' if args.alpha is not None else cfg.get('alpha_source', 'validate')})",
        flush=True,
    )
    print(f"z-cache: {z_cache} (full_rollouts={use_full_cache})", flush=True)
    print(f"output: {out_path}", flush=True)

    results: list[dict] = []
    done_ds: set[int] = set()
    ssv_features: list[int] = []
    ssv_weights: list[float] = []
    fstat_top5: list[int] = []
    gates: dict = {}
    gates_passed = False
    if args.resume and out_path.exists():
        results, done_ds, meta = load_checkpoint(out_path)
        ssv_features = meta.get("ssv_features") or []
        ssv_weights = meta.get("ssv_weights") or []
        fstat_top5 = meta.get("fstat_top5") or []
        gates = dict(meta.get("gates") or {})
        gates_passed = bool(meta.get("gates_passed")) or bool(results)
        if done_ds:
            print(f"Resume: skipping d={sorted(done_ds)}", flush=True)

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]
    if not eval_qs:
        raise SystemExit("No eval questions in trait bundle")

    vectors_path = Path(cfg["vectors"])
    if not vectors_path.exists():
        raise SystemExit(f"Missing persona vectors: {vectors_path}")
    v_dense = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"][layer].float()
    steer_norm = float(v_dense.norm())

    print("Loading model...", flush=True)
    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    print("Loading SAE...", flush=True)
    # SAE on CPU (262k ~2.5GB); optimization with 200 neg samples also stays on CPU.
    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev,
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=cfg["hs_index"],
    )
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)

    if use_full_cache and not z_cache.is_file():
        if not rollouts_path.is_file():
            raise SystemExit(f"Missing rollouts for full cache build: {rollouts_path}")
        print(f"=== Building full z-cache from {rollouts_path} ===", flush=True)
        samples = load_all_rollout_samples(rollouts_path, Path(cfg["bundle"]))
        z_all_build, y_all_build = collect_latents_samples(
            model, tok, dev, sae, layer, samples,
        )
        z_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(z_cache, z=z_all_build, y=y_all_build)
        print(
            f"Saved full z-cache: {z_all_build.shape[0]} samples -> {z_cache}",
            flush=True,
        )

    if not z_cache.is_file():
        raise SystemExit(f"Missing z-cache: {z_cache}")

    cached = np.load(z_cache)
    z_all, y_all = cached["z"], cached["y"]
    print(f"Loaded z-cache: {z_all.shape[0]} samples from {z_cache}", flush=True)

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()
    opt_dev = torch.device("cpu")

    v_full_cpu: torch.Tensor | None = None
    if not ssv_features:
        f_stats = f_statistic_per_feature(z_all, y_all)
        fstat_ranked = np.argsort(f_stats)[::-1][:K_COARSE].copy()
        fstat_top5 = fstat_ranked[:5].tolist()
        print(f"  Top-5 by F-stat: {fstat_top5}", flush=True)

        full_fids = fstat_ranked[:K_COARSE].copy()
        full_mask = torch.zeros(d_sae, dtype=torch.float32)
        full_mask[full_fids] = 1.0

        print(f"=== Optimizing v over K={K_COARSE} (fstat) ===", flush=True)
        v_full, _ = optimize_v_steer(
            z_neg_means,
            W_dec,
            mu_pos,
            mu_neg,
            full_mask,
            n_iter=args.n_iter,
            lr=0.05,
            lambda_lm=args.lambda_lm,
            beta=args.beta,
            beta_auto=args.beta_auto,
            opt_device=opt_dev,
        )
        v_full_cpu = v_full.cpu().float()
        weight_order = torch.argsort(v_full_cpu.abs(), descending=True)
        keep = weight_order[: max(K_COARSE, max(ds))].tolist()
        ssv_features = keep
        ssv_weights = [round(float(v_full_cpu[i]), 4) for i in keep]
        n_nonzero = int((v_full_cpu.abs() > 1e-8).sum())
        print(
            f"  SSV done: {n_nonzero} nonzero weights, top-5={ssv_features[:5]}",
            flush=True,
        )
    else:
        print(f"  Reusing SSV weights from checkpoint ({len(ssv_features)} ranked)", flush=True)
        v_full_cpu = torch.zeros(d_sae, dtype=torch.float32)
        for fid, w in zip(ssv_features, ssv_weights):
            v_full_cpu[int(fid)] = float(w)

    def encode_prompt(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
        attn = torch.ones_like(ids)
        return ids, attn

    def gen_text(ids: torch.Tensor, attn: torch.Tensor, hook_fn=None) -> str:
        handle = layers[layer].register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
        if handle:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True).strip()

    def judge_reply(prompt: str, reply: str) -> int | None:
        if len(reply.strip()) < 20:
            return None
        try:
            return int(score_transcript(judge_instr, neg_sys, prompt, reply).score)
        except Exception as exc:
            print(f"  judge error: {exc}", flush=True)
            return None

    def judge_all(label: str, prompts: list[str], replies: list[str]) -> tuple[list[int | None], float | None]:
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

    def write_output(*, early_stopped: bool = False, aborted: bool = False) -> None:
        payload = {
            "trait": cfg["trait"],
            "layer": layer,
            "method": "ssv_residual_add",
            "sae_id": cfg["sae_id"],
            "alpha": alpha,
            "alpha_reference": float(cfg["alpha"]),
            "steer_norm": round(steer_norm, 4),
            "z_cache": str(z_cache),
            "full_rollouts": use_full_cache,
            "n_z_samples": int(z_all.shape[0]),
            "judge_workers": judge_workers,
            "n_questions": len(eval_qs),
            "n_iter": args.n_iter,
            "k_coarse": K_COARSE,
            "beta": args.beta,
            "beta_auto": args.beta_auto,
            "lambda_lm": args.lambda_lm,
            "reopt_iters": args.reopt_iters,
            "ds": ds,
            "scales": scales,
            "early_stop_threshold": args.early_stop_threshold,
            "early_stop_consecutive": args.early_stop_consecutive,
            "fstat_top5": fstat_top5,
            "ssv_features": ssv_features,
            "ssv_weights": ssv_weights,
            "gates": gates,
            "gates_passed": gates_passed,
            "aborted": aborted,
            "early_stopped": early_stopped,
            "checkpoint": not aborted and len(results) < len(ds),
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if payload["checkpoint"] and not aborted:
            save_checkpoint(out_path, payload=payload)
        else:
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"  saved -> {out_path}", flush=True)

    def abort_gates(reason: str) -> None:
        print(f"\n=== ABORT: {reason} ===", flush=True)
        gates["abort_reason"] = reason
        write_output(aborted=True)
        raise SystemExit(reason)

    if not gates_passed and not args.skip_gates:
        assert v_full_cpu is not None
        cal_prompt = eval_qs[0]
        cal_ids, cal_attn = encode_prompt(cal_prompt)

        # Gate A: dense CAA sanity on Q1
        print("\n=== Gate A: dense CAA Q1 ===", flush=True)
        dense_dir = v_dense.to(device=dev, dtype=dtype).view(1, 1, -1)
        dense_hook = _steering_hook_fn(
            alpha, dense_dir, steer_last_token_only=False, hook_calls=[0],
        )
        dense_reply = gen_text(cal_ids, cal_attn, dense_hook)
        dense_score = judge_reply(cal_prompt, dense_reply)
        gate_a_pass = dense_score is not None and dense_score >= args.gate_a_min
        gates["A_dense_caa"] = {
            "pass": gate_a_pass,
            "q1_score": dense_score,
            "threshold": args.gate_a_min,
        }
        print(f"  Gate A: score={dense_score} pass={gate_a_pass}", flush=True)
        if not gate_a_pass:
            abort_gates(
                f"Gate A failed: dense CAA Q1={dense_score} < {args.gate_a_min} "
                "(Jun23 data may be broken; try May29 backup rollouts)"
            )

        # Gate B: cosine-sign at d=5,10,20
        print("\n=== Gate B: cosine-sign ===", flush=True)
        cos_by_d: dict[int, float] = {}
        for gd in GATE_B_DS:
            fids_g, weights_g, _ = truncate_v_by_weight(v_full_cpu, gd)
            v_base_g = build_v_sae(fids_g, weights_g, d_sae)
            cos_by_d[gd] = cosine_vs_dense(v_dense, v_base_g, W_dec)
        max_cos = max(cos_by_d.values()) if cos_by_d else -1.0
        best_gate_d = max(cos_by_d, key=cos_by_d.get) if cos_by_d else None
        gate_b_pass = max_cos >= args.gate_b_min_cosine
        gates["B_cosine_sign"] = {
            "pass": gate_b_pass,
            "cosine_by_d": cos_by_d,
            "max_cosine": max_cos,
            "best_d": best_gate_d,
            "threshold": args.gate_b_min_cosine,
        }
        print(f"  Gate B: cos_by_d={cos_by_d} max={max_cos} pass={gate_b_pass}", flush=True)
        if not gate_b_pass:
            abort_gates(
                f"Gate B failed: max cosine {max_cos} < {args.gate_b_min_cosine} "
                "(optimized direction anti-aligned to dense good)"
            )

        # Gate C: single-Q SSV probe at smallest d with cos >= threshold
        passing_ds = sorted(d for d, c in cos_by_d.items() if c >= args.gate_b_min_cosine)
        probe_d = passing_ds[0] if passing_ds else GATE_B_DS[0]
        _, _, v_trunc_c = truncate_v_by_weight(v_full_cpu, probe_d)
        direction_c, _ = residual_direction_from_v_trunc(
            v_trunc_c, W_dec, steer_norm, dev=dev, dtype=dtype,
        )
        print(f"\n=== Gate C: single-Q probe d={probe_d} alpha={alpha} ===", flush=True)
        probe_hook = _steering_hook_fn(
            alpha, direction_c, steer_last_token_only=False, hook_calls=[0],
        )
        probe_reply = gen_text(cal_ids, cal_attn, probe_hook)
        probe_score = judge_reply(cal_prompt, probe_reply)
        gate_c_pass = probe_score is not None and probe_score >= GATE_C_MIN_SCORE
        gates["C_single_q_probe"] = {
            "pass": gate_c_pass,
            "d": probe_d,
            "q1_score": probe_score,
            "cosine": cos_by_d.get(probe_d),
            "threshold": GATE_C_MIN_SCORE,
        }
        print(f"  Gate C: d={probe_d} score={probe_score} pass={gate_c_pass}", flush=True)
        if not gate_c_pass:
            abort_gates(
                f"Gate C failed: SSV probe d={probe_d} Q1={probe_score} "
                "(positive cosine but steering dead)"
            )

        gates_passed = True
        gates["all_passed"] = True
        write_output()
        print("\n=== All pre-gates passed; starting full d-sweep ===", flush=True)
    elif gates_passed:
        print("\n=== Pre-gates skipped (already passed or resume) ===", flush=True)

    def persist(*, early_stopped: bool = False) -> None:
        write_output(early_stopped=early_stopped)

    consecutive_high = 0
    for d in ds:
        if d in done_ds:
            print(f"\n=== d={d} (skipped, checkpoint) ===", flush=True)
            prev_row = next(r for r in results if r.get("d") == d)
            if prev_row.get("mean") is not None and prev_row["mean"] >= args.early_stop_threshold:
                consecutive_high += 1
            else:
                consecutive_high = 0
            continue

        if consecutive_high >= args.early_stop_consecutive:
            print(
                f"\n=== Early stop: {consecutive_high} consecutive d >= {args.early_stop_threshold}, "
                f"skipping d={d}+ ===",
                flush=True,
            )
            break

        print(f"\n=== d={d} ===", flush=True)
        assert v_full_cpu is not None
        fids, weights, v_trunc = truncate_v_by_weight(v_full_cpu, d)

        if args.reopt_iters > 0:
            d_mask = torch.zeros(d_sae, dtype=torch.float32)
            for fid in fids:
                d_mask[int(fid)] = 1.0
            print(
                f"  Phase-2 re-optimize: {len(fids)} features, "
                f"{args.reopt_iters} iters",
                flush=True,
            )
            v_reopt, _ = optimize_v_steer(
                z_neg_means,
                W_dec,
                mu_pos,
                mu_neg,
                d_mask,
                n_iter=args.reopt_iters,
                lr=0.05,
                lambda_lm=args.lambda_lm,
                beta=args.beta,
                beta_auto=0.0,
                opt_device=opt_dev,
            )
            v_trunc = v_reopt.cpu().float()
            active = v_trunc.abs() > 1e-8
            fids = active.nonzero(as_tuple=True)[0].tolist()
            weights = [round(float(v_trunc[i]), 4) for i in fids]

        v_base = build_v_sae(fids, weights, d_sae)
        direction_base, norm_raw = residual_direction_from_v_trunc(
            v_trunc, W_dec, steer_norm, dev=dev, dtype=dtype
        )
        cos_dense = cosine_vs_dense(v_dense, v_base, W_dec)
        print(
            f"  cosine_vs_dense={cos_dense} norm_raw={norm_raw:.4f} "
            f"n_active={len(fids)} fids={fids}",
            flush=True,
        )

        cal_prompt = eval_qs[0]
        cal_ids, cal_attn = encode_prompt(cal_prompt)
        best_scale = scales[0]
        best_cal_score: int | None = None
        scale_sweep: list[dict] = []

        print(f"  scale sweep on Q1: {scales}", flush=True)
        for scale in scales:
            hook = _steering_hook_fn(
                alpha * scale,
                direction_base,
                steer_last_token_only=False,
                hook_calls=[0],
            )
            reply = gen_text(cal_ids, cal_attn, hook)
            score = judge_reply(cal_prompt, reply)
            scale_sweep.append({"scale": scale, "score": score})
            print(f"    scale={scale} Q1 score={score}", flush=True)
            if score is not None and (best_cal_score is None or score > best_cal_score):
                best_cal_score = score
                best_scale = scale
            if score is not None and score >= args.scale_stop_threshold:
                print(
                    f"    scale early stop: score {score} >= {args.scale_stop_threshold}",
                    flush=True,
                )
                break

        label = f"d{d}"
        hook_best = _steering_hook_fn(
            alpha * best_scale,
            direction_base,
            steer_last_token_only=False,
            hook_calls=[0],
        )
        prompts, replies = [], []
        for prompt in eval_qs:
            ids, attn = encode_prompt(prompt)
            reply = gen_text(ids, attn, hook_best)
            prompts.append(prompt)
            replies.append(reply)
        scores, mean = judge_all(label, prompts, replies)

        row = {
            "d": d,
            "scale": best_scale,
            "mean": mean,
            "scores": scores,
            "feature_ids": fids,
            "feature_weights": weights,
            "cosine_vs_dense": cos_dense,
            "norm_raw": round(norm_raw, 4),
            "scale_sweep": scale_sweep,
            "best_cal_score": best_cal_score,
        }
        results.append(row)
        persist()

        if mean is not None and mean >= args.early_stop_threshold:
            consecutive_high += 1
            print(
                f"  consecutive_high={consecutive_high}/{args.early_stop_consecutive}",
                flush=True,
            )
        else:
            consecutive_high = 0

    early_stopped = consecutive_high >= args.early_stop_consecutive
    write_output(early_stopped=early_stopped)

    print("\n" + "=" * 60)
    print(f"{'d':>4s}  {'scale':>5s}  {'mean':>5s}  {'cos':>6s}  scores")
    print("-" * 60)
    for r in results:
        cos = r.get("cosine_vs_dense", "")
        cos_s = f"{cos:.3f}" if isinstance(cos, float) else ""
        print(
            f"  {r['d']:>4d}  {r['scale']:>5.1f}  {str(r['mean']):>5s}  {cos_s:>6s}  {r['scores']}"
        )
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
