#!/usr/bin/env python3
"""
Single-feature scale sweep to incoherence (Chen TES protocol).

For each of 10 candidate SAE features at L16:
  - Steer with h += scale * W_dec[fid] at log-spaced scales
  - Judge Q1 (calibration), early-stop on incoherence
  - Full 5-question eval on any (feature, scale) with score > 0 on Q1

Usage (GPU VM):
  PYTHONPATH=. python -u scripts/single_feature_scale_sweep.py --trait good
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
logger = logging.getLogger("single_feature_scale_sweep")

LAYER = None
DEFAULT_SCALES_SAE_HOOK = [50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0]
DEFAULT_SCALES_RESIDUAL = [2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
KNOWN_TOP_FIDS = {
    "good": [4747, 53982, 1805, 88099, 7995],
    "evil": [3486, 10156, 8926, 16833, 10488],
    "lawful": [87091, 40036, 16442, 22432, 230],
    "chaotic": [87091, 4893, 40036, 230, 22432],
}
MIN_REPLY_CHARS = 20


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


def judge_reply(judge_instr, neg_sys, prompt, reply) -> int | None:
    if len(reply.strip()) < MIN_REPLY_CHARS:
        return None
    try:
        return int(score_transcript(judge_instr, neg_sys, prompt, reply).score)
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return None


def mean_score(scores: list[int | None]) -> float | None:
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def load_top_fids(cfg: dict, trait: str, n: int = 5) -> list[int]:
    """Load trait-specific top features from OMP/SSV results, fallback to hardcoded."""
    for fname in [
        f"ssv_omp_dsweep_l{cfg['layer']}.json",
        f"ssv_dsweep_residual_full_l{cfg['layer']}.json",
        f"ssv_dsweep_residual_l{cfg['layer']}.json",
    ]:
        path = cfg["sae_dir"] / fname
        if path.is_file():
            data = json.loads(path.read_text())
            fids = data.get("omp_features") or data.get("ssv_features") or []
            if fids:
                return [int(f) for f in fids[:n]]
    return KNOWN_TOP_FIDS.get(trait, KNOWN_TOP_FIDS["good"])[:n]


def select_features(W_dec: torch.Tensor, v_layer: torch.Tensor, cfg: dict, trait: str) -> list[dict]:
    """Build 10-feature panel: 5 OMP (trait-specific) + 5 dec_cos (no overlap)."""
    v_unit = v_layer.float() / (v_layer.float().norm() + 1e-8)
    dec_cos = (W_dec @ v_unit).float()
    order = torch.argsort(dec_cos.abs(), descending=True)

    chosen: list[int] = []
    sources: dict[int, str] = {}

    def add(fid: int, source: str) -> None:
        if fid not in chosen:
            chosen.append(fid)
            sources[fid] = source

    for fid in load_top_fids(cfg, trait, n=5):
        add(fid, "known_top5")

    for fid in order.tolist():
        if len(chosen) >= 10:
            break
        add(int(fid), "dec_cos")

    return [
        {
            "feature_id": fid,
            "source": sources[fid],
            "dec_cos": round(float(dec_cos[fid]), 4),
            "dec_norm": round(float(W_dec[fid].norm()), 4),
        }
        for fid in chosen[:10]
    ]


def steer_feature(
    *,
    model,
    tok,
    dev,
    layers,
    pad_id,
    layer: int,
    sae,
    W_dec: torch.Tensor,
    fid: int,
    scale: float,
    prompt: str,
    neg_sys: str,
    judge_instr: str,
    dtype: torch.dtype,
    method: str = "sae_hook",
) -> dict:
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

    reply = gen_text(hook)
    incoherent = len(reply.strip()) < MIN_REPLY_CHARS
    score = None if incoherent else judge_reply(judge_instr, neg_sys, prompt, reply)
    return {
        "scale": scale,
        "score": score,
        "incoherent": incoherent,
        "method": method,
        "reply_preview": reply[:300],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--scales", default=None,
                    help="Comma-separated scales (default: auto based on --method)")
    ap.add_argument("--alpha-dense", type=float, default=None, help="Dense CAA alpha for reference")
    ap.add_argument("--n-questions-full", type=int, default=5)
    ap.add_argument("--method", default="sae_hook", choices=["sae_hook", "residual"],
                    help="sae_hook = encode+add+decode (OMP protocol); residual = h += scale*W_dec[fid]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.scales:
        scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    elif args.method == "sae_hook":
        scales = list(DEFAULT_SCALES_SAE_HOOK)
    else:
        scales = list(DEFAULT_SCALES_RESIDUAL)
    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha_dense)
    cfg["layer"] = layer
    cfg["sae_id"] = sae_id_for_layer(layer)
    cfg["hs_index"] = hidden_state_index(layer)
    alpha_dense = args.alpha_dense if args.alpha_dense is not None else float(cfg["alpha"])

    out_path = Path(
        args.out or cfg["sae_dir"] / f"single_feature_scale_sweep_l{layer}.json"
    )

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions_full]
    if not eval_qs:
        raise SystemExit("No eval questions in trait bundle")

    vectors_path = Path(cfg["vectors"])
    v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()
    dense_norm = float((alpha_dense * v_layer).norm())

    logger.info("Loading model...")
    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
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

    feature_panel = select_features(W_dec, v_layer, cfg, args.trait)
    logger.info("Feature panel (%d):", len(feature_panel))
    for row in feature_panel:
        logger.info(
            "  fid=%d source=%s dec_cos=%.4f",
            row["feature_id"],
            row["source"],
            row["dec_cos"],
        )

    cal_prompt = eval_qs[0]
    feature_results: list[dict] = []

    # Reference on Q1
    ids, attn = encode_ids(tok, neg_sys, cal_prompt, dev)
    gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id, layer)
    baseline_reply = gen_text()
    baseline_score = judge_reply(judge_instr, neg_sys, cal_prompt, baseline_reply)
    direction_dense = v_layer.to(device=dev, dtype=dtype).view(1, 1, -1)
    dense_reply = gen_text(
        _steering_hook_fn(
            alpha_dense,
            direction_dense,
            steer_last_token_only=False,
            hook_calls=[0],
        )
    )
    dense_score = judge_reply(judge_instr, neg_sys, cal_prompt, dense_reply)
    logger.info("Q1 reference: baseline=%s dense_caa=%s", baseline_score, dense_score)

    for row in feature_panel:
        fid = row["feature_id"]
        logger.info("=== fid %d (%s) ===", fid, row["source"])
        sweep_rows: list[dict] = []
        stopped_early = False
        stop_reason = None

        for scale in scales:
            res = steer_feature(
                model=model,
                tok=tok,
                dev=dev,
                layers=layers,
                pad_id=pad_id,
                layer=layer,
                sae=sae,
                W_dec=W_dec,
                fid=fid,
                scale=scale,
                prompt=cal_prompt,
                neg_sys=neg_sys,
                judge_instr=judge_instr,
                dtype=dtype,
                method=args.method,
            )
            sweep_rows.append(res)
            logger.info(
                "  scale=%.1f score=%s incoherent=%s",
                scale,
                res["score"],
                res["incoherent"],
            )
            if res["incoherent"]:
                stopped_early = True
                stop_reason = "incoherence"
                break
            if res["score"] is not None and res["score"] > 0:
                stopped_early = True
                stop_reason = "signal_found"
                break

        best = max(
            (r for r in sweep_rows if r["score"] is not None),
            key=lambda r: r["score"],
            default=None,
        )
        full_eval = None
        if best is not None and best["score"] > 0:
            best_scale = best["scale"]
            logger.info("  Full eval at scale=%.1f (Q1 score=%s)", best_scale, best["score"])
            per_q = []
            for qi, prompt in enumerate(eval_qs):
                res = steer_feature(
                    model=model,
                    tok=tok,
                    dev=dev,
                    layers=layers,
                    pad_id=pad_id,
                    layer=layer,
                    sae=sae,
                    W_dec=W_dec,
                    fid=fid,
                    scale=best_scale,
                    prompt=prompt,
                    neg_sys=neg_sys,
                    judge_instr=judge_instr,
                    dtype=dtype,
                    method=args.method,
                )
                per_q.append({"question_idx": qi, **res})
            full_eval = {
                "scale": best_scale,
                "mean_tes": mean_score([q["score"] for q in per_q]),
                "scores": [q["score"] for q in per_q],
                "questions": per_q,
            }
            logger.info("  Full eval mean_tes=%s scores=%s", full_eval["mean_tes"], full_eval["scores"])

        feature_results.append({
            **row,
            "calibration_q1": sweep_rows,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "best_q1": best,
            "full_eval": full_eval,
        })

    payload = {
        "trait": cfg["trait"],
        "layer": layer,
        "method": f"single_feature_scale_sweep_{args.method}",
        "steering_method": args.method,
        "sae_id": cfg["sae_id"],
        "scales": scales,
        "alpha_dense": alpha_dense,
        "dense_inject_norm": round(dense_norm, 4),
        "calibration_question": cal_prompt,
        "reference_q1": {
            "baseline_score": baseline_score,
            "dense_caa_score": dense_score,
            "baseline_preview": baseline_reply[:300],
            "dense_preview": dense_reply[:300],
        },
        "features": feature_results,
        "summary": {
            "any_signal_q1": any(
                r.get("best_q1") and r["best_q1"]["score"] and r["best_q1"]["score"] > 0
                for r in feature_results
            ),
            "features_with_full_eval": sum(1 for r in feature_results if r.get("full_eval")),
            "best_full_eval": max(
                (
                    (r["full_eval"]["mean_tes"], r["feature_id"], r["full_eval"]["scale"])
                    for r in feature_results
                    if r.get("full_eval") and r["full_eval"].get("mean_tes") is not None
                ),
                default=None,
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s", out_path)

    print("\n" + "=" * 60)
    print(f"{'fid':>8} {'source':>14} {'best_scale':>10} {'Q1':>5} {'full_mean':>10}")
    print("-" * 60)
    for r in feature_results:
        best = r.get("best_q1") or {}
        fe = r.get("full_eval") or {}
        print(
            f"{r['feature_id']:8d} {r['source']:>14} "
            f"{best.get('scale', '-'):>10} "
            f"{best.get('score', '-'):>5} "
            f"{fe.get('mean_tes', '-'):>10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
