#!/usr/bin/env python3
"""
OMP + SAE encode-steer-decode d-sweep with per-d scale calibration.

For each d in the sweep:
  1. Build sparse v_sae from top-d OMP coefficients
  2. Calibrate scale on Q1 at scales [1,2,3,5,8] (stop early if score >= 90)
  3. Full judge on N questions at best scale
  4. Early stop d-sweep after 3 consecutive d values with mean >= 90

Usage (GPU VM):
  PYTHONPATH=. python -u scripts/ssv_omp_dsweep.py --trait good
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers
from app.phase2 import load_sae_for_layer
from scripts.omp_decompose import omp_decompose
from scripts.sae_ssv_optimize import sae_steer_hook_fn
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait

DEFAULT_DS = [5, 10, 20, 50, 100]
DEFAULT_SCALES = [1.0, 2.0, 3.0, 5.0, 8.0]
EARLY_STOP_THRESHOLD = 90
EARLY_STOP_CONSECUTIVE = 3


def load_checkpoint(out_path: Path) -> tuple[list[dict], set[int]]:
    if not out_path.exists():
        return [], set()
    data = json.loads(out_path.read_text())
    results = list(data.get("results") or [])
    done = {int(r["d"]) for r in results if r.get("d") is not None}
    return results, done


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
    selected: list[int],
    coefs: list[float],
    d: int,
    d_sae: int,
) -> torch.Tensor:
    v = torch.zeros(d_sae, dtype=torch.float32)
    for i in range(min(d, len(selected))):
        v[selected[i]] = float(coefs[i])
    return v


def cosine_vs_dense(v_dense: torch.Tensor, v_sae: torch.Tensor, W_dec: torch.Tensor) -> float:
    recon = (W_dec.T @ v_sae).float()
    return round(
        F.cosine_similarity(v_dense.unsqueeze(0), recon.unsqueeze(0)).item(),
        4,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--ds", default=",".join(str(d) for d in DEFAULT_DS))
    ap.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge-workers", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
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
    args = ap.parse_args()

    ds = sorted({int(x) for x in args.ds.split(",") if x.strip()})
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    if not ds:
        raise SystemExit("No d values specified")
    if not scales:
        raise SystemExit("No scales specified")

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    out_path = Path(args.out or (cfg["sae_dir"] / f"ssv_omp_dsweep_l{layer}.json"))
    judge_workers = max(1, int(args.judge_workers))
    k_max = max(ds)

    results: list[dict] = []
    done_ds: set[int] = set()
    omp_features: list[int] = []
    if args.resume and out_path.exists():
        results, done_ds = load_checkpoint(out_path)
        prev = json.loads(out_path.read_text())
        omp_features = list(prev.get("omp_features") or [])
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

    print("Loading model...", flush=True)
    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    print("Loading SAE...", flush=True)
    sae_dev = dev if dev.type == "cuda" else torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev,
        release=SAE_RELEASE,
        sae_id=cfg["sae_id"],
        hidden_state_index=cfg["hs_index"],
    )
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)

    if not omp_features:
        print(f"=== OMP decompose k_max={k_max} ===", flush=True)
        selected, coefs, _ = omp_decompose(W_dec, v_dense, k_max, report_ks=set())
        omp_features = selected[:k_max]
        print(f"  OMP done: {len(selected)} features, top-5={selected[:5]}", flush=True)
    else:
        print(f"  Reusing OMP features from checkpoint ({len(omp_features)})", flush=True)
        selected = omp_features
        coefs = []
        residual = v_dense.clone()
        for fid in selected:
            c = float((W_dec[fid] @ residual) / (W_dec[fid] @ W_dec[fid]))
            coefs.append(c)
            residual = residual - c * W_dec[fid]

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

    def persist(*, early_stopped: bool = False) -> None:
        payload = {
            "trait": cfg["trait"],
            "layer": layer,
            "method": "omp_sae_hook",
            "sae_id": cfg["sae_id"],
            "alpha_reference": float(cfg["alpha"]),
            "judge_workers": judge_workers,
            "n_questions": len(eval_qs),
            "ds": ds,
            "scales": scales,
            "early_stop_threshold": args.early_stop_threshold,
            "early_stop_consecutive": args.early_stop_consecutive,
            "omp_features": omp_features,
            "early_stopped": early_stopped,
            "checkpoint": True,
            "results": results,
        }
        save_checkpoint(out_path, payload=payload)

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
        v_base = build_v_sae(selected, coefs, d, d_sae)
        fids = selected[:d]
        weights = [round(float(coefs[i]), 4) for i in range(d)]
        cos_dense = cosine_vs_dense(v_dense, v_base, W_dec)
        print(f"  cosine_vs_dense={cos_dense} fids={fids}", flush=True)

        cal_prompt = eval_qs[0]
        cal_ids, cal_attn = encode_prompt(cal_prompt)
        best_scale = scales[0]
        best_cal_score: int | None = None
        scale_sweep: list[dict] = []

        print(f"  scale sweep on Q1: {scales}", flush=True)
        for scale in scales:
            v_scaled = (v_base * scale).to(dev).float()
            hook = sae_steer_hook_fn(sae, v_scaled, prompt_len=0)
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
        v_best = (v_base * best_scale).to(dev).float()
        hook_best = sae_steer_hook_fn(sae, v_best, prompt_len=0)
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
    final_payload = {
        "trait": cfg["trait"],
        "layer": layer,
        "method": "omp_sae_hook",
        "sae_id": cfg["sae_id"],
        "alpha_reference": float(cfg["alpha"]),
        "judge_workers": judge_workers,
        "n_questions": len(eval_qs),
        "ds": ds,
        "scales": scales,
        "early_stop_threshold": args.early_stop_threshold,
        "early_stop_consecutive": args.early_stop_consecutive,
        "omp_features": omp_features,
        "early_stopped": early_stopped,
        "checkpoint": False,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_payload, indent=2))

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
