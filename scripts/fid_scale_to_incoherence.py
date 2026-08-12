#!/usr/bin/env python3
"""
Scale a single SAE feature via residual-add (Chen M.3.2: h += scale * W_dec[fid]).

Uses alpha-equivalent scales: scale = alpha * ||v_layer|| for alpha in {1,2,3,...}
(same inject norm as dense CAA at that alpha, since W_dec columns are unit norm).

Stops at first incoherent alpha. Full eval on all questions at each coherent step.

Usage:
  PYTHONPATH=. python -u scripts/fid_scale_to_incoherence.py \
      --trait good --layer 16 --fid 3333 --alphas 1,2,3,4,5,6
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
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, resolve_trait, sae_id_for_layer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("scale_to_incoherence")

MIN_REPLY_CHARS = 20
JUDGE_MAX_ROUNDS = 8
JUDGE_RETRY_BASE_SEC = 2.0


def encode_ids(tok, neg_sys, prompt, dev):
    msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    ids = ids.to(dev)
    attn = torch.ones_like(ids, dtype=torch.long, device=dev)
    return ids, attn


def make_gen_fn(model, tok, layers, ids, attn, pad_id, layer, max_new_tokens=200):
    def gen_text(hook_fn=None) -> str:
        handle = None
        if hook_fn is not None:
            handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id, use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return gen_text


def judge_one_score(judge_instr, neg_sys, prompt, reply) -> int | None:
    if len(reply.strip()) < MIN_REPLY_CHARS:
        return None
    return int(score_transcript(judge_instr, neg_sys, prompt, reply).score)


def judge_batch(judge_instr, neg_sys, prompts, replies, workers=5):
    """Score all replies; retry failed judge calls with backoff instead of aborting."""
    scores: list[int | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    for round_num in range(JUDGE_MAX_ROUNDS):
        if not pending:
            break
        batch_workers = min(workers, max(1, len(pending)))
        if round_num > 0:
            wait = JUDGE_RETRY_BASE_SEC * (2 ** min(round_num - 1, 5))
            logger.warning(
                "Judge retry round %d/%d for %d questions (sleep %.1fs, workers=%d)",
                round_num + 1, JUDGE_MAX_ROUNDS, len(pending), wait, batch_workers,
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
        raise RuntimeError(
            f"Judge failed after {JUDGE_MAX_ROUNDS} rounds for questions: {pending}"
        )
    return scores


def load_checkpoint(out_path: Path, fid: int, layer: int) -> dict | None:
    if not out_path.exists():
        return None
    try:
        data = json.loads(out_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Ignoring corrupt checkpoint %s", out_path)
        return None
    if data.get("fid") != fid or data.get("layer") != layer:
        return None
    return data


def save_checkpoint(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    logger.info("Checkpoint saved %s", out_path)


def mean_score(scores):
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def generate_residual(model, tok, dev, layers, pad_id, layer, W_dec, fid, scale, prompt, neg_sys, dtype):
    ids, attn = encode_ids(tok, neg_sys, prompt, dev)
    gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id, layer)
    direction = (scale * W_dec[fid].float()).to(device=dev, dtype=dtype).view(1, 1, -1)
    hook = _steering_hook_fn(1.0, direction, steer_last_token_only=False, hook_calls=[0])
    return gen_text(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--fid", type=int, default=3333)
    ap.add_argument("--sign", type=float, default=1.0, help="+1 or -1")
    ap.add_argument("--alphas", default="1,2,3,4,5,6",
                    help="Alpha-equivalent grid: scale = alpha * ||v_layer||")
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--alpha-dense", type=float, default=None)
    ap.add_argument("--judge-workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--skip-reference",
        action="store_true",
        help="Reuse baseline/dense from existing checkpoint JSON (resume)",
    )
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = args.layer
    cfg["layer"] = layer
    cfg["sae_id"] = sae_id_for_layer(layer)
    cfg["hs_index"] = hidden_state_index(layer)
    alpha_dense = args.alpha_dense if args.alpha_dense is not None else float(cfg["alpha"])
    jw = args.judge_workers

    out_path = Path(args.out or cfg["sae_dir"] / f"fid{args.fid}_scale_to_incoherence_l{layer}.json")

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]
    if len(eval_qs) < args.n_questions:
        logger.warning("Bundle has %d questions, requested %d", len(eval_qs), args.n_questions)

    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()
    v_norm = float(v_layer.norm())
    dense_inject_norm = float((alpha_dense * v_layer).norm())
    alpha_grid = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    logger.info("Loading model...")
    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    sae, _ = load_sae_for_layer(
        torch.device("cpu"), release=SAE_RELEASE,
        sae_id=cfg["sae_id"], hidden_state_index=cfg["hs_index"],
    )
    W_dec = _get_decoder_columns(sae).float()
    fid = args.fid
    cos_to_v = float((W_dec[fid] @ (v_layer / v_layer.norm())).item())
    logger.info("fid=%d cos_to_v=%.4f ||v||=%.1f dense_inject@alpha=%.1f is %.1f",
                fid, cos_to_v, v_norm, alpha_dense, dense_inject_norm)
    logger.info("Alpha grid: %s -> scale = alpha * ||v||", alpha_grid)

    gen_kw = dict(model=model, tok=tok, dev=dev, layers=layers, pad_id=pad_id,
                  layer=layer, W_dec=W_dec, fid=fid, neg_sys=neg_sys, dtype=dtype)

    checkpoint = load_checkpoint(out_path, fid, layer)
    done_alphas: set[float] = set()
    sweep_results: list[dict] = []
    if checkpoint:
        sweep_results = list(checkpoint.get("sweep") or [])
        done_alphas = {float(r["alpha_equiv"]) for r in sweep_results}
        logger.info("Resuming checkpoint with %d alphas already done: %s",
                    len(done_alphas), sorted(done_alphas))

    if args.skip_reference and checkpoint and checkpoint.get("reference"):
        ref = checkpoint["reference"]
        bl_scores = ref.get("baseline_scores", [])
        dense_scores = ref["dense_scores"]
        logger.info(
            "Reference (checkpoint): baseline=%s dense=%s",
            ref.get("baseline_mean"), ref.get("dense_mean"),
        )
    else:
        direction_dense = v_layer.to(device=dev, dtype=dtype).view(1, 1, -1)
        bl_replies, dense_replies = [], []
        for prompt in eval_qs:
            ids, attn = encode_ids(tok, neg_sys, prompt, dev)
            gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id, layer)
            bl_replies.append(gen_text())
            dense_replies.append(gen_text(_steering_hook_fn(alpha_dense, direction_dense,
                                                            steer_last_token_only=False, hook_calls=[0])))
        bl_scores = judge_batch(judge_instr, neg_sys, eval_qs, bl_replies, jw)
        dense_scores = judge_batch(judge_instr, neg_sys, eval_qs, dense_replies, jw)
        logger.info("Reference: baseline=%s dense=%s", mean_score(bl_scores), mean_score(dense_scores))

    payload = {
        "trait": cfg["trait"],
        "layer": layer,
        "fid": fid,
        "cos_to_v": round(cos_to_v, 4),
        "method": "residual_add",
        "sign": args.sign,
        "alpha_dense": alpha_dense,
        "v_norm": round(v_norm, 2),
        "alpha_grid": alpha_grid,
        "scale_schedule": "scale = alpha_equiv * ||v||",
        "reference": {
            "baseline_mean": mean_score(bl_scores),
            "dense_mean": mean_score(dense_scores),
            "baseline_scores": bl_scores,
            "dense_scores": dense_scores,
        },
        "sweep": sweep_results,
    }
    save_checkpoint(out_path, payload)

    logger.info("Evaluating on %d questions (no Q0 gate)", len(eval_qs))

    sign = args.sign

    for alpha_equiv in alpha_grid:
        if float(alpha_equiv) in done_alphas:
            logger.info("Skip alpha=%.1f (checkpoint)", alpha_equiv)
            continue
        scale = alpha_equiv * v_norm
        effective = sign * scale
        ratio_to_dense = alpha_equiv / alpha_dense

        full_replies = [generate_residual(scale=effective, prompt=p, **gen_kw) for p in eval_qs]
        incoherent_count = sum(1 for r in full_replies if len(r.strip()) < MIN_REPLY_CHARS)
        full_scores = judge_batch(judge_instr, neg_sys, eval_qs, full_replies, jw)
        m = mean_score(full_scores)

        row = {
            "alpha_equiv": alpha_equiv,
            "scale": round(scale, 2),
            "effective_scale": round(effective, 2),
            "ratio_to_dense_alpha": round(ratio_to_dense, 3),
            "incoherent_count": incoherent_count,
            "n_questions": len(eval_qs),
            "mean_tes": m,
            "scores": full_scores,
            "replies": [r[:300] for r in full_replies],
        }
        logger.info(
            "alpha=%.1f scale=%.0f mean_tes=%.1f incoherent=%d/%d scores=%s",
            alpha_equiv, scale, m or 0, incoherent_count, len(eval_qs), full_scores,
        )
        sweep_results.append(row)
        payload["sweep"] = sweep_results
        save_checkpoint(out_path, payload)

        if incoherent_count == len(eval_qs):
            logger.info("STOP: all %d questions incoherent at alpha=%.1f", len(eval_qs), alpha_equiv)
            break

    logger.info("Finished %s", out_path)

    print(f"\n{'alpha':>6} {'scale':>8} {'mean':>6} {'incoherent':>10} scores")
    print("-" * 80)
    for r in sweep_results:
        print(f"{r['alpha_equiv']:>6.1f} {r['scale']:>8.0f} {r.get('mean_tes', '-'):>6} "
              f"{r['incoherent_count']:>4}/{r['n_questions']:>2} {r['scores']}")


if __name__ == "__main__":
    main()
