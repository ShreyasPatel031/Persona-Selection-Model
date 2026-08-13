#!/usr/bin/env python3
"""Five EMD hyperparameter experiments at K=5 (good trait, L15).

E1: SSV fids — scale sweep on Q1, eval 20Q at best scale
E2: Arad fids + output-score weights — same scale sweep
E3: SSV / Arad / OMP — norm-matched decoded magnitude (no hand scale)
E4: Arad fids + OMP coefficients @ scale=3
E5: SSV re-optimize on OMP top-5 mask @ scale=1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.sae_common import _get_decoder_columns
from app.persona.steering_demo import _language_model_layers
from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import f_statistic_per_feature, optimize_v_steer, sae_steer_hook_fn
from scripts.ssv_omp_k_sweep import (
    build_v_sae,
    generate_batched,
    load_dense_ref,
    load_omp_at_k,
    load_omp_coef_map,
    load_ssv_at_k,
    mean_score,
    save_checkpoint,
)
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, resolve_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("emd_hyper")

K = 5
DEFAULT_SCALES = [1.0, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
OMP_CALIBRATED_SCALE = 3.0
MIN_REPLY_CHARS = 20
JUDGE_MAX_ROUNDS = 16
JUDGE_RETRY_BASE_SEC = 2.0


def judge_one(judge_instr, sys_prompt, prompt, reply) -> int | None:
    if len(reply.strip()) < MIN_REPLY_CHARS:
        return None
    return int(score_transcript(judge_instr, sys_prompt, prompt, reply).score)


def judge_batch(judge_instr, sys_prompt, prompts, replies, workers=10) -> list[int | None]:
    scores: list[int | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    def _judge(i: int) -> int | None:
        return judge_one(judge_instr, sys_prompt, prompts[i], replies[i])

    for round_num in range(JUDGE_MAX_ROUNDS):
        if not pending:
            break
        if round_num > 0:
            time.sleep(JUDGE_RETRY_BASE_SEC * (2 ** min(round_num - 1, 5)))
        failed: list[int] = []
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
            futures = {pool.submit(_judge, i): i for i in pending}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    scores[i] = fut.result()
                except Exception as exc:
                    logger.warning("Q%d judge failed: %s", i, exc)
                    failed.append(i)
        pending = failed
    return scores


def sparse_decode_norm(W_dec: torch.Tensor, fids: list[int], weights: list[float], scale: float) -> float:
    v = build_v_sae(fids, weights, W_dec.shape[0], scale)
    decoded = W_dec.T @ v
    return float(decoded.norm().item())


def norm_matched_scale(
    W_dec: torch.Tensor, fids: list[int], weights: list[float], dense_norm: float,
) -> float:
    raw = sparse_decode_norm(W_dec, fids, weights, 1.0)
    if raw < 1e-8:
        return 1.0
    return dense_norm / raw


def load_arad_at_k(sae_dir: Path, layer: int, k: int) -> tuple[list[int], list[float]]:
    path = sae_dir / f"sae_output_score_l{layer}.json"
    loaded = load_ssv_at_k(sae_dir, layer, k, feature_file=path)
    if not loaded:
        raise FileNotFoundError(path)
    return loaded


def arad_fids_omp_weights(sae_dir: Path, layer: int, k: int) -> tuple[list[int], list[float]]:
    fids, _ = load_arad_at_k(sae_dir, layer, k)
    coef = load_omp_coef_map(sae_dir, layer)
    weights = [coef.get(fid, 0.0) for fid in fids]
    return fids, weights


def run_emd_eval(
    *,
    model,
    tok,
    sae,
    layer_mod,
    neg_sys,
    eval_qs,
    judge_instr,
    dev,
    pad_id,
    fids,
    weights,
    scale,
    d_sae,
    gen_batch_size,
    judge_workers,
    questions: list[str] | None = None,
) -> tuple[list[str], list[int | None], float | None]:
    qs = questions or eval_qs
    v_sae = build_v_sae(fids, weights, d_sae, scale)
    hook = sae_steer_hook_fn(sae, v_sae.to(dev).float(), prompt_len=0)
    replies = generate_batched(
        model, tok, neg_sys, qs, dev, pad_id,
        hook, layer_mod, max_new_tokens=200, batch_size=gen_batch_size,
    )
    scores = judge_batch(judge_instr, neg_sys, qs, replies, workers=judge_workers)
    return replies, scores, mean_score(scores)


def calibrate_scale_q1(
    *,
    model,
    tok,
    sae,
    layer_mod,
    neg_sys,
    cal_prompt,
    judge_instr,
    dev,
    pad_id,
    fids,
    weights,
    scales,
    d_sae,
) -> tuple[float, list[dict], int | None]:
    from scripts.ssv_omp_k_sweep import encode_ids

    best_scale = scales[0]
    best_score: int | None = None
    sweep: list[dict] = []
    cal_ids, cal_attn = encode_ids(tok, neg_sys, cal_prompt, dev)

    for scale in scales:
        v_sae = build_v_sae(fids, weights, d_sae, scale)
        hook = sae_steer_hook_fn(sae, v_sae.to(dev).float(), prompt_len=0)
        handle = layer_mod.register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(
                input_ids=cal_ids, attention_mask=cal_attn,
                max_new_tokens=200, do_sample=False, pad_token_id=pad_id, use_cache=True,
            )
        handle.remove()
        reply = tok.decode(gen[0, cal_ids.shape[-1]:], skip_special_tokens=True).strip()
        score = judge_one(judge_instr, neg_sys, cal_prompt, reply)
        sweep.append({"scale": scale, "score": score})
        logger.info("  scale=%.1f Q1 score=%s", scale, score)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_scale = scale
        if score is not None and score >= 90:
            logger.info("  early stop scale sweep at %.1f (score=%s)", scale, score)
            break
    return best_scale, sweep, best_score


def optimize_ssv_on_omp_mask(
    sae_dir: Path,
    layer: int,
    k: int,
    *,
    z_cache: Path,
    n_iter: int,
    lr: float,
    lambda_lm: float,
    beta: float,
    W_dec: torch.Tensor,
    d_sae: int,
    dev: torch.device,
) -> tuple[list[int], list[float]]:
    cached = np.load(z_cache)
    z_all, y_all = cached["z"], cached["y"]
    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()

    path = sae_dir / f"omp_decomposition_262k_l{layer}.json"
    rows = sorted(
        json.loads(path.read_text())["decomposition"],
        key=lambda r: abs(float(r["coefficient"])),
        reverse=True,
    )[:k]
    omp_fids = [int(r["feature_id"]) for r in rows]
    feature_mask = torch.zeros(d_sae, dtype=torch.float32)
    feature_mask[omp_fids] = 1.0

    v_opt, _ = optimize_v_steer(
        z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
        n_iter=n_iter, lr=lr, lambda_lm=lambda_lm, beta=beta, opt_device=dev,
    )
    active = v_opt.abs() > 1e-8
    top = torch.argsort(v_opt.abs(), descending=True)
    fids = [int(f) for f in top if active[f]]
    weights = [float(v_opt[f]) for f in top if active[f]]
    return fids, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--experiments", default="E1,E2,E3,E4,E5")
    ap.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--judge-workers", type=int, default=10)
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--z-cache", default=None)
    ap.add_argument("--opt-iters", type=int, default=100)
    args = ap.parse_args()

    exps = {x.strip().upper() for x in args.experiments.split(",") if x.strip()}
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    sae_dir = Path(cfg["sae_dir"])
    out = Path(args.out or sae_dir / f"emd_hyperparam_experiments_l{layer}.json")

    done: set[str] = set()
    results: list[dict] = []
    if args.resume and out.is_file():
        doc = json.loads(out.read_text())
        results = list(doc.get("results") or [])
        done = {r["experiment"] for r in results}

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]
    cal_prompt = eval_qs[0]

    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()
    dense_norm = float((alpha * v_layer).norm().item())

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    sae, _ = load_sae_for_layer(
        dev if dev.type == "cuda" else torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=hidden_state_index(layer),
    )
    W_dec = _get_decoder_columns(sae).float().cpu()
    d_sae = int(sae.cfg.d_sae)
    layer_mod = layers[layer]

    dense_loaded = load_dense_ref(sae_dir, layer, [], sae_dir / f"dense_caa_20q_l{layer}.json")
    dense_ref = dense_loaded[0] if dense_loaded else None

    payload = {
        "trait": cfg["trait"],
        "layer": layer,
        "alpha_dense": alpha,
        "k": K,
        "dense_ref": dense_ref,
        "dense_inject_norm": round(dense_norm, 3),
        "scale_grid": scales,
        "omp_calibrated_scale": OMP_CALIBRATED_SCALE,
        "n_questions": len(eval_qs),
        "results": results,
    }

    def persist(row: dict) -> None:
        results.append(row)
        payload["results"] = results
        save_checkpoint(out, payload)
        print(f"DONE {row['experiment']} mean={row.get('mean_trait')} scale={row.get('scale')}", flush=True)

    common = dict(
        model=model, tok=tok, sae=sae, layer_mod=layer_mod, neg_sys=neg_sys,
        eval_qs=eval_qs, judge_instr=judge_instr, dev=dev, pad_id=pad_id,
        d_sae=d_sae, gen_batch_size=args.gen_batch_size, judge_workers=args.judge_workers,
    )

    # --- E1: SSV scale sweep ---
    if "E1" in exps and "E1" not in done:
        logger.info("=== E1: SSV EMD scale sweep ===")
        fids, weights = load_ssv_at_k(sae_dir, layer, K)
        if not fids:
            raise FileNotFoundError(f"SSV K={K} missing in {sae_dir}")
        best_scale, scale_sweep, _ = calibrate_scale_q1(
            neg_sys=neg_sys, cal_prompt=cal_prompt, judge_instr=judge_instr,
            dev=dev, pad_id=pad_id, fids=fids, weights=weights, scales=scales,
            d_sae=d_sae, model=model, tok=tok, sae=sae, layer_mod=layer_mod,
        )
        _, scores, mean = run_emd_eval(fids=fids, weights=weights, scale=best_scale, **common)
        persist({
            "experiment": "E1",
            "label": "SSV_scale_sweep",
            "feature_source": "sae_ssv_full_sweep",
            "weight_source": "ssv_optimizer",
            "feature_ids": fids,
            "feature_weights": [round(w, 4) for w in weights],
            "scale_mode": "q1_calibrated",
            "scale": best_scale,
            "scale_sweep": scale_sweep,
            "decode_norm": round(sparse_decode_norm(W_dec, fids, weights, best_scale), 2),
            "mean_trait": mean,
            "scores": scores,
            "delta_from_dense": round(mean - dense_ref, 1) if mean and dense_ref else None,
        })

    # --- E2: Arad scale sweep ---
    if "E2" in exps and "E2" not in done:
        logger.info("=== E2: Arad EMD scale sweep ===")
        fids, weights = load_arad_at_k(sae_dir, layer, K)
        best_scale, scale_sweep, _ = calibrate_scale_q1(
            neg_sys=neg_sys, cal_prompt=cal_prompt, judge_instr=judge_instr,
            dev=dev, pad_id=pad_id, fids=fids, weights=weights, scales=scales,
            d_sae=d_sae, model=model, tok=tok, sae=sae, layer_mod=layer_mod,
        )
        _, scores, mean = run_emd_eval(fids=fids, weights=weights, scale=best_scale, **common)
        persist({
            "experiment": "E2",
            "label": "Arad_scale_sweep",
            "feature_source": "output_score_arad",
            "weight_source": "output_score",
            "feature_ids": fids,
            "feature_weights": [round(w, 6) for w in weights],
            "scale_mode": "q1_calibrated",
            "scale": best_scale,
            "scale_sweep": scale_sweep,
            "decode_norm": round(sparse_decode_norm(W_dec, fids, weights, best_scale), 2),
            "mean_trait": mean,
            "scores": scores,
            "delta_from_dense": round(mean - dense_ref, 1) if mean and dense_ref else None,
        })

    # --- E3: Norm-matched (SSV, Arad, OMP) ---
    if "E3" in exps and "E3" not in done:
        logger.info("=== E3: Norm-matched EMD ===")
        e3_rows = []
        for tag, fids, weights, src in [
            ("SSV", *load_ssv_at_k(sae_dir, layer, K), "ssv_optimizer"),
            ("Arad", *load_arad_at_k(sae_dir, layer, K), "output_score"),
            ("OMP", *load_omp_at_k(sae_dir, layer, K), "omp_coef"),
        ]:
            if not fids:
                continue
            nm_scale = norm_matched_scale(W_dec, fids, weights, dense_norm)
            logger.info("  E3 %s norm_matched_scale=%.4f", tag, nm_scale)
            _, scores, mean = run_emd_eval(fids=fids, weights=weights, scale=nm_scale, **common)
            e3_rows.append({
                "subset": tag,
                "feature_ids": fids,
                "feature_weights": [round(w, 4) for w in weights[:5]],
                "scale": round(nm_scale, 4),
                "decode_norm": round(sparse_decode_norm(W_dec, fids, weights, nm_scale), 2),
                "mean_trait": mean,
                "scores": scores,
            })
        persist({
            "experiment": "E3",
            "label": "norm_matched_emd",
            "scale_mode": "decode_norm_match_dense",
            "target_decode_norm": round(dense_norm, 2),
            "sub_results": e3_rows,
            "mean_trait": max((r["mean_trait"] or 0) for r in e3_rows) if e3_rows else None,
        })

    # --- E4: Arad fids + OMP weights @ scale=3 ---
    if "E4" in exps and "E4" not in done:
        logger.info("=== E4: Arad fids + OMP weights @ scale=3 ===")
        fids, weights = arad_fids_omp_weights(sae_dir, layer, K)
        _, scores, mean = run_emd_eval(
            fids=fids, weights=weights, scale=OMP_CALIBRATED_SCALE, **common,
        )
        persist({
            "experiment": "E4",
            "label": "Arad_fids_OMP_weights",
            "feature_source": "output_score_arad",
            "weight_source": "omp_coef",
            "feature_ids": fids,
            "feature_weights": [round(w, 4) for w in weights],
            "scale_mode": "fixed",
            "scale": OMP_CALIBRATED_SCALE,
            "decode_norm": round(sparse_decode_norm(W_dec, fids, weights, OMP_CALIBRATED_SCALE), 2),
            "mean_trait": mean,
            "scores": scores,
            "delta_from_dense": round(mean - dense_ref, 1) if mean and dense_ref else None,
        })

    # --- E5: SSV optimize on OMP mask @ scale=1 ---
    if "E5" in exps and "E5" not in done:
        logger.info("=== E5: SSV on OMP mask K=5 @ scale=1 ===")
        z_cache = Path(args.z_cache or sae_dir / f"probe_z_cache_l{layer}.npz")
        opt_dev = dev if dev.type == "cuda" else torch.device("cpu")
        fids, weights = optimize_ssv_on_omp_mask(
            sae_dir, layer, K,
            z_cache=z_cache, n_iter=args.opt_iters, lr=0.05,
            lambda_lm=0.5, beta=0.01, W_dec=W_dec, d_sae=d_sae, dev=opt_dev,
        )
        _, scores, mean = run_emd_eval(fids=fids, weights=weights, scale=1.0, **common)
        persist({
            "experiment": "E5",
            "label": "SSV_omp_mask_opt",
            "feature_source": "omp_mask",
            "weight_source": "ssv_optimizer_on_mask",
            "feature_ids": fids,
            "feature_weights": [round(w, 4) for w in weights],
            "scale_mode": "optimizer_native",
            "scale": 1.0,
            "decode_norm": round(sparse_decode_norm(W_dec, fids, weights, 1.0), 2),
            "mean_trait": mean,
            "scores": scores,
            "delta_from_dense": round(mean - dense_ref, 1) if mean and dense_ref else None,
        })

    print(f"\n=== ALL EXPERIMENTS DONE ===", flush=True)
    for r in results:
        print(f"  {r['experiment']}: mean={r.get('mean_trait')} scale={r.get('scale')}", flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
