#!/usr/bin/env python3
"""Fine-grained K sweep for SSV or OMP steering at validated layer.

Supports two steering modes:
  --steer-mode residual  (default) W_dec.T @ v_sparse, norm-matched to ||α·v_dense||
  --steer-mode emd       SAE encode → add v_sparse*scale → decode (old dsweep method)
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

import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.sae_common import _get_decoder_columns
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.sae_ssv_optimize import sae_steer_hook_fn
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, resolve_trait, sae_id_for_layer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("k_sweep")

MIN_REPLY_CHARS = 20
JUDGE_MAX_ROUNDS = 16
JUDGE_RETRY_BASE_SEC = 2.0


def encode_ids(tok, sys_prompt: str, prompt: str, dev):
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    ids = ids.to(dev)
    attn = torch.ones_like(ids, dtype=torch.long, device=dev)
    return ids, attn


def generate_batched(
    model, tok, sys_prompt: str, prompts: list[str], dev, pad_id: int,
    hook, layer_module, max_new_tokens: int = 200, batch_size: int = 0,
) -> list[str]:
    """Generate replies for all prompts using left-padded batched generation."""
    all_ids = []
    for prompt in prompts:
        ids, _ = encode_ids(tok, sys_prompt, prompt, dev)
        all_ids.append(ids.squeeze(0))

    if batch_size <= 0:
        batch_size = len(all_ids)

    replies: list[str] = [""] * len(prompts)

    for start in range(0, len(all_ids), batch_size):
        chunk = all_ids[start : start + batch_size]
        n = len(chunk)
        max_len = max(ids.shape[0] for ids in chunk)

        batch_ids = torch.full((n, max_len), pad_id, dtype=chunk[0].dtype, device=dev)
        attn_mask = torch.zeros(n, max_len, dtype=torch.long, device=dev)
        for i, ids in enumerate(chunk):
            L = ids.shape[0]
            batch_ids[i, max_len - L :] = ids
            attn_mask[i, max_len - L :] = 1

        handle = layer_module.register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(
                input_ids=batch_ids, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_id, use_cache=True,
            )
        handle.remove()

        for i in range(n):
            reply_ids = gen[i, max_len:]
            replies[start + i] = tok.decode(reply_ids, skip_special_tokens=True).strip()

    return replies


def judge_batch(judge_instr, sys_prompt, prompts, replies, workers=4) -> list[int | None]:
    scores: list[int | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    def judge_one(i: int) -> int | None:
        if len(replies[i].strip()) < MIN_REPLY_CHARS:
            return None
        return int(score_transcript(judge_instr, sys_prompt, prompts[i], replies[i]).score)

    for round_num in range(JUDGE_MAX_ROUNDS):
        if not pending:
            break
        if round_num > 0:
            time.sleep(JUDGE_RETRY_BASE_SEC * (2 ** min(round_num - 1, 5)))
        failed: list[int] = []
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
            futures = {pool.submit(judge_one, i): i for i in pending}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    scores[i] = fut.result()
                except Exception as exc:
                    logger.warning("Q%d judge failed: %s", i, exc)
                    failed.append(i)
        pending = failed
    return scores


def mean_score(scores):
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def build_residual_direction(W_dec, fids, weights, dense_norm, dev, dtype) -> torch.Tensor:
    d_sae = W_dec.shape[0]
    v_sparse = torch.zeros(d_sae, dtype=torch.float32)
    for fid, w in zip(fids, weights):
        v_sparse[int(fid)] = float(w)
    v_res = (W_dec.T @ v_sparse).float()
    raw = float(v_res.norm())
    if raw > 1e-8:
        v_res = v_res * (dense_norm / raw)
    return v_res.to(device=dev, dtype=dtype).view(1, 1, -1)


def build_v_sae(fids, weights, d_sae: int, scale: float) -> torch.Tensor:
    """Build sparse SAE-space vector for EMD steering (encode → add → decode)."""
    v = torch.zeros(d_sae, dtype=torch.float32)
    for fid, w in zip(fids, weights):
        v[int(fid)] = float(w)
    return v * scale


def sparse_decode_norm(sae, fids: list[int], weights: list[float], d_sae: int) -> float:
    """||decode(v_sparse)|| for unit-scale sparse weights."""
    v = build_v_sae(fids, weights, d_sae, 1.0)
    with torch.no_grad():
        h = sae.decode(v.view(1, 1, -1).to(next(sae.parameters()).device))
    return float(h.float().norm().item())


def resolve_emd_scale(
    *,
    norm_match: bool,
    fixed_scale: float,
    dense_norm: float,
    decode_norm: float,
) -> float:
    if not norm_match:
        return fixed_scale
    if decode_norm < 1e-8:
        logger.warning("decode norm ~0; falling back to fixed scale=%s", fixed_scale)
        return fixed_scale
    return dense_norm / decode_norm


def load_ssv_at_k(
    sae_dir: Path, layer: int, k: int, feature_file: Path | None = None,
) -> tuple[list[int], list[float]] | None:
    path = feature_file or (sae_dir / f"sae_ssv_full_sweep_262k_l{layer}.json")
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    for row in d.get("results") or []:
        if int(row.get("k", 0)) == k:
            fids = [int(x) for x in row["feature_ids"]]
            weights = [float(x) for x in row["feature_weights"]]
            return fids, weights
    return None


def load_omp_coef_map(sae_dir: Path, layer: int) -> dict[int, float]:
    path = sae_dir / f"omp_decomposition_262k_l{layer}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["feature_id"]): float(r["coefficient"]) for r in d.get("decomposition") or []}


def load_ssv_fids_omp_weights(
    sae_dir: Path, layer: int, k: int, feature_file: Path | None = None,
) -> tuple[list[int], list[float]] | None:
    """Plan C: SSV feature IDs with OMP decomposition coefficients."""
    loaded = load_ssv_at_k(sae_dir, layer, k, feature_file=feature_file)
    if not loaded:
        return None
    fids, _ = loaded
    coef_map = load_omp_coef_map(sae_dir, layer)
    weights = [coef_map.get(fid, 0.0) for fid in fids]
    return fids, weights


def load_omp_at_k(sae_dir: Path, layer: int, k: int) -> tuple[list[int], list[float]]:
    path = sae_dir / f"omp_decomposition_262k_l{layer}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(
        d.get("decomposition") or [],
        key=lambda r: abs(float(r.get("coefficient", 0))),
        reverse=True,
    )[:k]
    return [int(r["feature_id"]) for r in rows], [float(r["coefficient"]) for r in rows]


def save_checkpoint(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)


def load_dense_ref(
    sae_dir: Path,
    layer: int,
    results: list[dict],
    dense_ref_path: Path | None,
) -> tuple[float, list] | None:
    """Load dense reference scores without re-running generation/judge."""
    for row in results:
        if row.get("label") == "dense_ref" and row.get("mean_trait") is not None:
            return float(row["mean_trait"]), list(row.get("scores") or [])

    candidates: list[Path] = []
    if dense_ref_path:
        candidates.append(dense_ref_path)
    candidates.append(sae_dir / f"dense_caa_20q_l{layer}.json")
    for path in sorted(sae_dir.glob(f"*_k_sweep_l{layer}_20q*.json")):
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if path.name.startswith("dense_caa"):
            mean = doc.get("dense_mean")
            scores = doc.get("dense_scores") or []
        else:
            ref = next((r for r in doc.get("results") or [] if r.get("label") == "dense_ref"), None)
            if not ref:
                continue
            mean = ref.get("mean_trait")
            scores = ref.get("scores") or []
        if mean is not None and scores:
            logger.info("Using dense ref from %s mean=%s", path, mean)
            return float(mean), scores
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--method", required=True, choices=["ssv", "omp"])
    ap.add_argument(
        "--ks",
        default="5,10,15,20,25,30,40,50,75,100,150,200,300,450,750,1000",
    )
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--judge-workers", type=int, default=4)
    ap.add_argument("--steer-mode", choices=["residual", "emd"], default="residual",
                    help="residual: W_dec.T projection; emd: SAE encode-modify-decode (old dsweep)")
    ap.add_argument("--scale", type=float, default=3.0,
                    help="EMD scale factor (only used with --steer-mode emd)")
    ap.add_argument("--scales", default=None,
                    help="Comma-separated EMD scales; calibrate on Q1, eval 20Q at best")
    ap.add_argument("--norm-match-emd", action="store_true",
                    help="EMD scale = ||alpha*v_dense|| / ||decode(v_sparse)||")
    ap.add_argument("--scale-stop-threshold", type=int, default=90,
                    help="Stop scale sweep early when Q1 score reaches this")
    ap.add_argument("--gen-batch-size", type=int, default=0,
                    help="Batch size for generation (0 = all prompts at once, default)")
    ap.add_argument("--dense-ref", default=None,
                    help="Path to dense_caa_20q_l{layer}.json (auto-detected if omitted)")
    ap.add_argument("--run-dense-ref", action="store_true",
                    help="Re-run dense CAA generation+judge instead of loading saved ref")
    ap.add_argument("--feature-file", default=None,
                    help="Override feature/weight JSON (e.g. sae_ssv_omp_mask_l15.json for Plan E)")
    ap.add_argument("--weight-mode", choices=["default", "omp-for-ssv-fids"], default="default",
                    help="default: weights from feature-file; omp-for-ssv-fids: Plan C")
    ap.add_argument("--experiment", default=None,
                    help="Tag stored in output JSON (plan_b_scale2, plan_c, plan_e, etc.)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    tag = f"l{layer}"
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    scale_grid = (
        [float(x.strip()) for x in args.scales.split(",") if x.strip()]
        if args.scales
        else None
    )
    sae_dir = Path(cfg["sae_dir"])

    feature_file = Path(args.feature_file) if args.feature_file else None
    exp_tag = f"_{args.experiment}" if args.experiment else ""
    steer_suffix = "_emd" if args.steer_mode == "emd" else ""
    out = Path(args.out or sae_dir / f"{args.method}_k_sweep_{tag}_20q{steer_suffix}{exp_tag}.json")
    existing = json.loads(out.read_text()) if args.resume and out.is_file() else None
    results: list[dict] = list((existing or {}).get("results") or [])
    done_ks = {int(r["k"]) for r in results if r.get("k") is not None}

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()
    dense_norm = float((alpha * v_layer).norm().item())

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=hidden_state_index(layer),
    )
    W_dec = _get_decoder_columns(sae).float()

    steer_desc = ("sae_encode_add_decode (EMD), scale={:.1f}".format(args.scale)
                  if args.steer_mode == "emd"
                  else "W_dec.T @ v_sparse, norm-matched to ||alpha*v_dense||")
    payload = {
        "method": f"{args.method}_k_sweep_{args.steer_mode}",
        "trait": cfg["trait"],
        "layer": layer,
        "alpha_dense": alpha,
        "steer_mode": args.steer_mode,
        "emd_scale": args.scale if args.steer_mode == "emd" else None,
        "scale_grid": scale_grid,
        "norm_match_emd": args.norm_match_emd if args.steer_mode == "emd" else False,
        "dense_inject_norm": round(dense_norm, 3),
        "n_questions": len(eval_qs),
        "steering": steer_desc,
        "experiment": args.experiment,
        "weight_mode": args.weight_mode,
        "feature_file": str(feature_file) if feature_file else None,
        "ks_planned": ks,
        "results": results,
    }

    dense_ref_path = Path(args.dense_ref) if args.dense_ref else None
    loaded_dense = load_dense_ref(sae_dir, layer, results, dense_ref_path)
    if loaded_dense and "dense_ref" not in {r.get("label") for r in results}:
        dense_mean, dense_scores = loaded_dense
        results.append({
            "label": "dense_ref",
            "k": None,
            "mean_trait": dense_mean,
            "scores": dense_scores,
            "source": "saved",
        })
        payload["results"] = results
        save_checkpoint(out, payload)
    elif args.run_dense_ref and "dense_ref" not in {r.get("label") for r in results}:
        direction_dense = (alpha * v_layer).to(device=dev, dtype=dtype).view(1, 1, -1)
        dense_hook = _steering_hook_fn(1.0, direction_dense, steer_last_token_only=False, hook_calls=[0])
        logger.info("Running dense reference (--run-dense-ref)...")
        dense_replies = generate_batched(
            model, tok, neg_sys, eval_qs, dev, pad_id,
            dense_hook, layers[layer], max_new_tokens=200,
            batch_size=args.gen_batch_size,
        )
        dense_scores = judge_batch(judge_instr, neg_sys, eval_qs, dense_replies, workers=args.judge_workers)
        dense_mean = mean_score(dense_scores)
        logger.info("Dense ref mean=%s scores=%s", dense_mean, dense_scores)
        results.append({
            "label": "dense_ref",
            "k": None,
            "mean_trait": dense_mean,
            "scores": dense_scores,
        })
        payload["results"] = results
        save_checkpoint(out, payload)
    elif "dense_ref" not in {r.get("label") for r in results}:
        raise RuntimeError(
            f"No dense ref found for L{layer}. Expected dense_caa_20q_l{layer}.json "
            f"or dense_ref in an existing *_k_sweep_l{layer}_20q*.json. "
            f"Pass --run-dense-ref to regenerate."
        )

    dense_ref = next(r["mean_trait"] for r in results if r.get("label") == "dense_ref")

    for k in ks:
        if k in done_ks:
            logger.info("Skip K=%d (checkpoint)", k)
            continue

        if args.weight_mode == "omp-for-ssv-fids":
            loaded = load_ssv_fids_omp_weights(sae_dir, layer, k, feature_file=feature_file)
        elif args.method == "ssv" or feature_file:
            loaded = load_ssv_at_k(sae_dir, layer, k, feature_file=feature_file)
        else:
            loaded = None
        if loaded:
            fids, weights = loaded
        elif args.method == "omp":
            fids, weights = load_omp_at_k(sae_dir, layer, k)
        else:
            logger.warning("No row for K=%d, skipping", k)
            continue

        d_sae = W_dec.shape[0]
        decode_norm = sparse_decode_norm(sae, fids, weights, d_sae) if args.steer_mode == "emd" else None
        scale_sweep: list[dict] = []
        emd_scale = args.scale
        best_cal: int | None = None

        if args.steer_mode == "emd" and args.norm_match_emd:
            emd_scale = resolve_emd_scale(
                norm_match=True,
                fixed_scale=args.scale,
                dense_norm=dense_norm,
                decode_norm=float(decode_norm or 0.0),
            )
            logger.info(
                "Norm-match EMD: decode_norm=%.4f -> scale=%.4f",
                decode_norm,
                emd_scale,
            )
        elif args.steer_mode == "emd" and scale_grid:
            cal_prompt = eval_qs[0]
            cal_ids, cal_attn = encode_ids(tok, neg_sys, cal_prompt, dev)
            best_cal: int | None = None
            logger.info("Scale sweep on Q1: %s", scale_grid)
            for sc in scale_grid:
                v_try = build_v_sae(fids, weights, d_sae, sc)
                cal_hook = sae_steer_hook_fn(sae, v_try.to(dev).float(), prompt_len=0)
                handle = layers[layer].register_forward_hook(cal_hook)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=cal_ids,
                        attention_mask=cal_attn,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=pad_id,
                        use_cache=True,
                    )
                handle.remove()
                reply = tok.decode(gen[0, cal_ids.shape[-1]:], skip_special_tokens=True).strip()
                del gen
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                q1_scores = judge_batch(
                    judge_instr, neg_sys, [cal_prompt], [reply], workers=1,
                )
                q1 = q1_scores[0]
                scale_sweep.append({"scale": sc, "q1_score": q1})
                logger.info("  scale=%s Q1=%s", sc, q1)
                if q1 is not None and (best_cal is None or q1 > best_cal):
                    best_cal = q1
                    emd_scale = sc
                if q1 is not None and q1 >= args.scale_stop_threshold:
                    logger.info("  early stop scale sweep at %s (Q1=%s)", sc, q1)
                    break
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        if args.steer_mode == "emd":
            v_sae = build_v_sae(fids, weights, d_sae, emd_scale)
            hook = sae_steer_hook_fn(sae, v_sae.to(dev).float(), prompt_len=0)
        else:
            direction = build_residual_direction(W_dec, fids, weights, dense_norm, dev, dtype)
            hook = _steering_hook_fn(1.0, direction, steer_last_token_only=False, hook_calls=[0])

        logger.info(
            "=== %s K=%d (%d features, %s scale=%s) ===",
            args.method.upper(), k, len(fids), args.steer_mode, emd_scale if args.steer_mode == "emd" else "n/a",
        )
        replies = generate_batched(
            model, tok, neg_sys, eval_qs, dev, pad_id,
            hook, layers[layer], max_new_tokens=200,
            batch_size=args.gen_batch_size,
        )

        scores = judge_batch(judge_instr, neg_sys, eval_qs, replies, workers=args.judge_workers)
        m = mean_score(scores)
        row = {
            "label": f"{args.method.upper()}_K{k}",
            "k": k,
            "n_features": len(fids),
            "feature_ids": fids,
            "feature_weights": [round(w, 6) for w in weights],
            "mean_trait": m,
            "scores": scores,
            "delta_from_dense": round(m - dense_ref, 1) if m is not None else None,
        }
        if args.steer_mode == "emd":
            row["emd_scale"] = round(float(emd_scale), 6)
            row["decode_norm_unit"] = round(float(decode_norm or 0.0), 4)
            if scale_sweep:
                row["scale_sweep"] = scale_sweep
                row["best_cal_q1"] = best_cal
        results.append(row)
        payload["results"] = results
        save_checkpoint(out, payload)
        logger.info("K=%d mean=%s (dense_ref=%s, delta=%s)", k, m, dense_ref, row.get("delta_from_dense"))
        print(f"PROGRESS {args.method.upper()} K={k} mean={m} dense_ref={dense_ref}", flush=True)

    print(f"\n=== {args.method.upper()} K SWEEP DONE ===", flush=True)
    for r in results:
        if r.get("k") is not None:
            print(f"  K={r['k']:>4}  mean={r.get('mean_trait')}  delta={r.get('delta_from_dense')}", flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
