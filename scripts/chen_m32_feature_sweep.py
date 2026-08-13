#!/usr/bin/env python3
"""
Chen et al. Appendix M.3.2 — per-feature causal analysis via steering.

For each of the top-N SAE features (by cosine similarity to persona vector),
tests steering conditions (default: residual-add positive only).

Scale sweep on Q0 to find best scale per feature, then full eval on all questions.
Supports checkpointing, robust judge retries, and rank slicing for parallel VMs.

Usage (GPU VM):
  PYTHONPATH=. python -u scripts/chen_m32_feature_sweep.py \
      --trait good --layer 16 --top-k 50 --rank-start 1 --rank-end 25 \
      --n-questions 20 --conditions residual_pos_only \
      --out persona_runs/dnd_good_scale/sae/chen_m32_top50_20q_l16_part1.json
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
from scripts.trait_sae_config import SAE_RELEASE, check_override, hidden_state_index, resolve_trait, sae_id_for_layer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("chen_m32")

SCALES_RESIDUAL = [2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
SCALES_SAE_HOOK = [100.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0, 3000.0, 5000.0, 8000.0]
MIN_REPLY_CHARS = 20
JUDGE_MAX_ROUNDS = 16
JUDGE_RETRY_BASE_SEC = 2.0
T_PASS_DEFAULT = 50.0

CONDITION_CONFIGS = [
    {"label": "residual_pos", "method": "residual", "sign": +1},
    {"label": "residual_neg", "method": "residual", "sign": -1},
    {"label": "sae_hook_pos", "method": "sae_hook", "sign": +1},
    {"label": "sae_hook_neg", "method": "sae_hook", "sign": -1},
]

CONDITION_PRESETS = {
    "all": CONDITION_CONFIGS,
    "residual_pos_only": [CONDITION_CONFIGS[0]],
}


def encode_ids(tok, neg_sys: str, prompt: str, dev):
    msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    ids = ids.to(dev)
    attn = torch.ones_like(ids, dtype=torch.long, device=dev)
    return ids, attn


def make_gen_fn(model, tok, layers, ids, attn, pad_id, layer: int, max_new_tokens=200):
    def gen_text(hook_fn=None) -> str:
        handle = None
        if hook_fn is not None:
            handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return gen_text


def judge_one_score(judge_instr, neg_sys, prompt, reply) -> int | None:
    if len(reply.strip()) < MIN_REPLY_CHARS:
        return None
    return int(score_transcript(judge_instr, neg_sys, prompt, reply).score)


def judge_batch(judge_instr, neg_sys, prompts, replies, workers=4) -> list[int | None]:
    scores: list[int | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    for round_num in range(JUDGE_MAX_ROUNDS):
        if not pending:
            break
        batch_workers = min(workers, max(1, len(pending)))
        if round_num > 0:
            wait = JUDGE_RETRY_BASE_SEC * (2 ** min(round_num - 1, 5))
            logger.warning(
                "Judge retry round %d/%d for %d items (sleep %.1fs)",
                round_num + 1, JUDGE_MAX_ROUNDS, len(pending), wait,
            )
            time.sleep(wait)
        failed: list[int] = []
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            futures = {
                pool.submit(judge_one_score, judge_instr, neg_sys, prompts[i], replies[i]): i
                for i in pending
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    scores[i] = fut.result()
                except Exception as exc:
                    logger.warning("Q%d judge failed (round %d): %s", i, round_num + 1, exc)
                    failed.append(i)
        pending = failed

    if pending:
        logger.error(
            "Judge gave up after %d rounds for indices %s; continuing with null scores",
            JUDGE_MAX_ROUNDS, pending,
        )
    return scores


def mean_score(scores: list[int | None]) -> float | None:
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def generate_steered(
    *, model, tok, dev, layers, pad_id, layer: int,
    sae, W_dec: torch.Tensor,
    fid: int, scale: float,
    prompt: str, neg_sys: str,
    dtype: torch.dtype, method: str,
) -> str:
    d_sae = W_dec.shape[0]
    ids, attn = encode_ids(tok, neg_sys, prompt, dev)
    gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id, layer)

    if method == "sae_hook":
        v_sae = torch.zeros(d_sae, dtype=torch.float32)
        v_sae[fid] = scale
        hook = sae_steer_hook_fn(sae, v_sae, prompt_len=0)
    else:
        col = W_dec[fid].float()
        direction = (scale * col).to(device=dev, dtype=dtype).view(1, 1, -1)
        hook = _steering_hook_fn(1.0, direction, steer_last_token_only=False, hook_calls=[0])

    return gen_text(hook)


def select_top_features(W_dec: torch.Tensor, v_layer: torch.Tensor, top_k: int) -> list[dict]:
    v_unit = v_layer.float() / (v_layer.float().norm() + 1e-8)
    cos = (W_dec @ v_unit).float()
    top_idx = cos.topk(top_k).indices.tolist()
    return [
        {
            "cos_rank": rank,
            "feature_id": int(fid),
            "cos_to_v": round(float(cos[fid]), 4),
            "dec_norm": round(float(W_dec[fid].norm()), 4),
        }
        for rank, fid in enumerate(top_idx, start=1)
    ]


def load_checkpoint(out_path: Path) -> dict | None:
    if not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Ignoring corrupt checkpoint %s", out_path)
        return None


def save_checkpoint(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    logger.info("Checkpoint saved %s", out_path)


def build_conclusion(feature_results: list[dict], t_pass: float, dense_mean: float | None) -> dict:
    tes_vals = [r.get("best_mean_tes") for r in feature_results if r.get("best_mean_tes") is not None]
    max_tes = max(tes_vals) if tes_vals else None
    best_feature = None
    if tes_vals:
        best_feature = max(feature_results, key=lambda r: r.get("best_mean_tes") or -1)

    null_hypothesis = (
        "At least one SAE feature among the top-K by cosine similarity to v_trait "
        f"can achieve mean TES >= {t_pass} on full eval via residual-add steering."
    )
    rejected = max_tes is None or max_tes < t_pass
    return {
        "null_hypothesis": null_hypothesis,
        "t_pass": t_pass,
        "dense_caa_mean": dense_mean,
        "max_feature_mean_tes": max_tes,
        "best_feature_id": best_feature.get("feature_id") if best_feature else None,
        "best_feature_cos_rank": best_feature.get("cos_rank") if best_feature else None,
        "best_feature_cos_to_v": best_feature.get("cos_to_v") if best_feature else None,
        "null_hypothesis_rejected": rejected,
        "conclusion": (
            f"No single feature reached T_pass={t_pass} (max={max_tes}). "
            "Null hypothesis rejected: top-cos SAE features cannot reproduce full trait via M.3.2 residual-add."
            if rejected
            else f"Feature fid={best_feature.get('feature_id')} reached mean TES={max_tes} >= {t_pass}."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Chen M.3.2 per-feature TES sweep")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-features", type=int, default=None,
                    help="Deprecated alias for --top-k when rank slice not used")
    ap.add_argument("--top-k", type=int, default=50,
                    help="Select top-K features by cos(W_dec, v) before rank slice")
    ap.add_argument("--rank-start", type=int, default=1,
                    help="1-indexed start rank within top-K pool (inclusive)")
    ap.add_argument("--rank-end", type=int, default=None,
                    help="1-indexed end rank within top-K pool (inclusive)")
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--alpha-dense", type=float, default=None)
    ap.add_argument("--judge-workers", type=int, default=4)
    ap.add_argument("--t-pass", type=float, default=T_PASS_DEFAULT)
    ap.add_argument(
        "--conditions",
        default="residual_pos_only",
        choices=sorted(CONDITION_PRESETS.keys()),
    )
    ap.add_argument("--skip-reference", action="store_true",
                    help="Reuse baseline/dense reference from checkpoint (default when present)")
    ap.add_argument("--force-reference", action="store_true",
                    help="Recompute baseline/dense reference even if checkpoint has it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    top_k = args.top_k if args.n_features is None else args.n_features
    rank_end = args.rank_end if args.rank_end is not None else top_k

    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    n_questions = args.n_questions if args.n_questions is not None else int(cfg["n_questions"])
    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha_dense, cli_nq=args.n_questions)
    cfg["layer"] = layer
    cfg["sae_id"] = sae_id_for_layer(layer)
    cfg["hs_index"] = hidden_state_index(layer)
    alpha_dense = args.alpha_dense if args.alpha_dense is not None else float(cfg["alpha"])
    jw = max(1, args.judge_workers)
    condition_configs = CONDITION_PRESETS[args.conditions]

    out_path = Path(
        args.out or cfg["sae_dir"] / f"chen_m32_feature_sweep_l{layer}.json"
    )

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:n_questions]
    if not eval_qs:
        raise SystemExit("No eval questions in trait bundle")

    vectors_path = Path(cfg["vectors"])
    v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()

    logger.info("Loading model...")
    model, tok, dev = load_model_and_tokenizer()
    layers_list = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    logger.info("Loading SAE L%d...", layer)
    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=cfg["hs_index"],
    )
    W_dec = _get_decoder_columns(sae).float()

    full_panel = select_top_features(W_dec, v_layer, top_k)
    feature_panel = [
        f for f in full_panel if args.rank_start <= f["cos_rank"] <= rank_end
    ]
    logger.info(
        "Processing cos ranks %d-%d (%d features); pool top-%d cos [%.4f, %.4f]",
        args.rank_start, rank_end, len(feature_panel), top_k,
        full_panel[-1]["cos_to_v"] if full_panel else 0,
        full_panel[0]["cos_to_v"] if full_panel else 0,
    )
    logger.info("Conditions: %s; judge workers: %d", args.conditions, jw)

    checkpoint = load_checkpoint(out_path)
    feature_results: list[dict] = []
    done_fids: set[int] = set()
    baseline_scores: list[int | None] = []
    dense_scores: list[int | None] = []

    if checkpoint:
        feature_results = list(checkpoint.get("features") or [])
        done_fids = {int(r["feature_id"]) for r in feature_results}
        ref = checkpoint.get("reference") or {}
        baseline_scores = ref.get("baseline_scores") or []
        dense_scores = ref.get("dense_scores") or []
        logger.info("Resuming checkpoint: %d features done", len(done_fids))

    have_reference = bool(baseline_scores and dense_scores)
    if not args.force_reference and (args.skip_reference or have_reference) and checkpoint:
        logger.info(
            "Reference (checkpoint): baseline=%s dense=%s",
            mean_score(baseline_scores), mean_score(dense_scores),
        )
    else:
        logger.info("Dense CAA reference (alpha=%.2f) on %d questions...", alpha_dense, len(eval_qs))
        direction_dense = v_layer.to(device=dev, dtype=dtype).view(1, 1, -1)
        bl_replies, dense_replies = [], []
        for prompt in eval_qs:
            ids, attn = encode_ids(tok, neg_sys, prompt, dev)
            gen_text = make_gen_fn(model, tok, layers_list, ids, attn, pad_id, layer)
            bl_replies.append(gen_text())
            dense_replies.append(
                gen_text(_steering_hook_fn(alpha_dense, direction_dense,
                                           steer_last_token_only=False, hook_calls=[0]))
            )
        baseline_scores = judge_batch(judge_instr, neg_sys, eval_qs, bl_replies, workers=jw)
        dense_scores = judge_batch(judge_instr, neg_sys, eval_qs, dense_replies, workers=jw)
        logger.info(
            "Reference: baseline_mean=%s dense_mean=%s",
            mean_score(baseline_scores), mean_score(dense_scores),
        )

    payload = {
        "method": "chen_m32_feature_sweep",
        "trait": cfg["trait"],
        "layer": layer,
        "sae_id": cfg["sae_id"],
        "alpha_dense": alpha_dense,
        "top_k": top_k,
        "rank_start": args.rank_start,
        "rank_end": rank_end,
        "n_features_in_run": len(feature_panel),
        "n_questions": len(eval_qs),
        "conditions_preset": args.conditions,
        "conditions_tested": [c["label"] for c in condition_configs],
        "scales_residual": SCALES_RESIDUAL,
        "scales_sae_hook": SCALES_SAE_HOOK,
        "t_pass": args.t_pass,
        "reference": {
            "baseline_mean": mean_score(baseline_scores),
            "baseline_scores": baseline_scores,
            "dense_mean": mean_score(dense_scores),
            "dense_scores": dense_scores,
        },
        "features": feature_results,
    }
    save_checkpoint(out_path, payload)

    cal_prompt = eval_qs[0]
    gen_base = dict(
        model=model, tok=tok, dev=dev, layers=layers_list, pad_id=pad_id,
        layer=layer, sae=sae, W_dec=W_dec, neg_sys=neg_sys, dtype=dtype,
    )

    for fi, feat in enumerate(feature_panel):
        fid = feat["feature_id"]
        if fid in done_fids:
            logger.info("Skip fid %d (checkpoint)", fid)
            continue

        logger.info(
            "=== [rank %d] [%d/%d] fid %d (cos=%.4f) ===",
            feat["cos_rank"], fi + 1, len(feature_panel), fid, feat["cos_to_v"],
        )

        conditions: list[dict] = []
        for cc in condition_configs:
            label = cc["label"]
            method = cc["method"]
            sign = cc["sign"]
            scales = SCALES_RESIDUAL if method == "residual" else SCALES_SAE_HOOK

            q0_replies, q0_scales_used = [], []
            for s in scales:
                effective_scale = sign * s
                reply = generate_steered(
                    fid=fid, scale=effective_scale, prompt=cal_prompt,
                    method=method, **gen_base,
                )
                q0_replies.append(reply)
                q0_scales_used.append(s)
                if len(reply.strip()) < MIN_REPLY_CHARS:
                    break

            q0_scores = judge_batch(
                judge_instr, neg_sys,
                [cal_prompt] * len(q0_replies), q0_replies, workers=jw,
            )
            best_score = -1
            best_scale = scales[len(scales) // 2]
            for s, sc in zip(q0_scales_used, q0_scores):
                if sc is not None and sc > best_score:
                    best_score = sc
                    best_scale = s
            logger.info("  [%s] Q0 best: scale=%.1f score=%d", label, best_scale, max(best_score, 0))

            effective_best = sign * best_scale
            full_replies = [
                generate_steered(fid=fid, scale=effective_best, prompt=prompt,
                                 method=method, **gen_base)
                for prompt in eval_qs
            ]
            full_scores = judge_batch(judge_instr, neg_sys, eval_qs, full_replies, workers=jw)
            m = mean_score(full_scores)
            logger.info("  [%s] mean_tes=%.1f scores=%s", label, m or 0, full_scores)

            conditions.append({
                "label": label,
                "method": method,
                "sign": sign,
                "best_scale": best_scale,
                "q0_best_score": max(best_score, 0),
                "full_eval": {
                    "scale": best_scale,
                    "effective_scale": effective_best,
                    "mean_tes": m,
                    "scores": full_scores,
                    "replies": [r[:300] for r in full_replies],
                },
            })

        best_cond = max(conditions, key=lambda c: c["full_eval"]["mean_tes"] or 0)
        logger.info("  BEST: %s mean_tes=%.1f",
                     best_cond["label"], best_cond["full_eval"]["mean_tes"] or 0)

        feature_results.append({
            **feat,
            "conditions": conditions,
            "best_condition": best_cond["label"],
            "best_mean_tes": best_cond["full_eval"]["mean_tes"],
        })
        payload["features"] = feature_results
        payload["formal_proof"] = build_conclusion(
            feature_results, args.t_pass, mean_score(dense_scores),
        )
        save_checkpoint(out_path, payload)

    feature_results.sort(key=lambda r: r["best_mean_tes"] or 0, reverse=True)
    payload["features"] = feature_results
    payload["formal_proof"] = build_conclusion(
        feature_results, args.t_pass, mean_score(dense_scores),
    )
    save_checkpoint(out_path, payload)
    logger.info("Finished %s", out_path)

    fp = payload["formal_proof"]
    print(f"\n{'='*90}")
    print(f"Chen M.3.2 — {cfg['trait']} L{layer} — ranks {args.rank_start}-{rank_end}")
    print(f"Reference: baseline={mean_score(baseline_scores)}  dense_caa={mean_score(dense_scores)}")
    print(f"T_pass={args.t_pass}  max_feature_tes={fp.get('max_feature_mean_tes')}")
    print(f"Conclusion: {fp.get('conclusion')}")
    print(f"{'rank':>4} {'fid':>8} {'cos':>7} {'scale':>7} {'mean_tes':>9}")
    print(f"{'-'*90}")
    for r in feature_results:
        bc = next(c for c in r["conditions"] if c["label"] == r["best_condition"])
        print(
            f"{r['cos_rank']:>4} {r['feature_id']:>8} {r['cos_to_v']:>7.3f} "
            f"{bc['best_scale']:>7.0f} {(r['best_mean_tes'] or 0):>9.1f}"
        )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
