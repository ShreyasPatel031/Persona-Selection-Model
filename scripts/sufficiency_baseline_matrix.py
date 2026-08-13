#!/usr/bin/env python3
"""
Sufficiency + baseline matrix: test whether candidate sparse sets can elicit good
(Chen M.3.2 residual-add on neg_sys) vs dense DiffMean vs pos-prompt baseline.

All conditions evaluated on the same eval questions with robust judge retries.
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
from scripts.trait_sae_config import (
    SAE_RELEASE,
    check_override,
    hidden_state_index,
    resolve_trait,
    sae_id_for_layer,
)


def save_checkpoint(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    logger.info("Checkpoint saved %s (%d conditions)", out, len(payload.get("conditions") or []))


def load_checkpoint(out: Path) -> dict | None:
    if not out.exists():
        return None
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Ignoring corrupt checkpoint %s", out)
        return None


def encode_ids(tok, sys_prompt: str, prompt: str, dev):
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
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


def judge_one_score(judge_instr, sys_prompt, prompt, reply) -> int | None:
    if len(reply.strip()) < MIN_REPLY_CHARS:
        return None
    return int(score_transcript(judge_instr, sys_prompt, prompt, reply).score)


def judge_batch(judge_instr, sys_prompt, prompts, replies, workers=4) -> list[int | None]:
    scores: list[int | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))
    for round_num in range(JUDGE_MAX_ROUNDS):
        if not pending:
            break
        if round_num > 0:
            wait = JUDGE_RETRY_BASE_SEC * (2 ** min(round_num - 1, 5))
            logger.warning("Judge retry round %d for %d questions", round_num + 1, len(pending))
            time.sleep(wait)
        failed: list[int] = []
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
            futures = {
                pool.submit(judge_one_score, judge_instr, sys_prompt, prompts[i], replies[i]): i
                for i in pending
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    scores[i] = fut.result()
                except Exception as exc:
                    logger.warning("Q%d judge failed: %s", i, exc)
                    failed.append(i)
        pending = failed
    if pending:
        raise RuntimeError(f"Judge failed after {JUDGE_MAX_ROUNDS} rounds: {pending}")
    return scores


def mean_score(scores):
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def load_ssv_k100(sae_dir: Path, k: int = 100) -> tuple[list[int], list[float]]:
    d = json.loads((sae_dir / "sae_ssv_full_sweep_262k_l16.json").read_text())
    for row in d.get("results") or []:
        if int(row.get("k", 0)) == k:
            fids = [int(x) for x in row["feature_ids"]]
            weights = [float(x) for x in row["feature_weights"]]
            return fids, weights
    return [], []


def load_omp_top(sae_dir: Path, k: int) -> tuple[list[int], list[float]]:
    d = json.loads((sae_dir / "omp_decomposition_262k_l16.json").read_text())
    rows = sorted(
        d.get("decomposition") or [],
        key=lambda r: abs(float(r.get("coefficient", 0))),
        reverse=True,
    )[:k]
    return [int(r["feature_id"]) for r in rows], [float(r["coefficient"]) for r in rows]


def load_feature_sets(convergence_path: Path, sae_dir: Path) -> dict[str, dict]:
    d = json.loads(convergence_path.read_text(encoding="utf-8"))
    sets: dict[str, dict] = {}

    core = (d.get("method_invariant_core") or {}).get("2") or []
    if core:
        sets["core_votes2"] = {
            "feature_ids": [int(c["feature_id"]) for c in core],
            "weights": None,
        }

    for name in ("ssv_k100", "omp_vdense", "gradsae_output"):
        fids = (d.get("selectors") or {}).get(name) or []
        if fids:
            sets[f"{name}_top10"] = {
                "feature_ids": [int(x) for x in fids[:10]],
                "weights": None,
            }

    ssv_fids, ssv_w = load_ssv_k100(sae_dir, k=100)
    if ssv_fids:
        sets["ssv_k100_weighted"] = {"feature_ids": ssv_fids, "weights": ssv_w}
        sets["ssv_k100_top10_weighted"] = {
            "feature_ids": ssv_fids[:10],
            "weights": ssv_w[:10],
        }

    omp_fids, omp_w = load_omp_top(sae_dir, k=10)
    if omp_fids:
        sets["omp_top10_weighted"] = {"feature_ids": omp_fids, "weights": omp_w}

    return sets


def build_residual_direction(
    W_dec: torch.Tensor,
    fids: list[int],
    dense_norm: float,
    dev,
    dtype,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """SSV-style: v_res = W_dec.T @ v_sparse, scaled to dense inject norm (He SSV / Chen M.3.2)."""
    d_sae = W_dec.shape[0]
    v_sparse = torch.zeros(d_sae, dtype=torch.float32)
    if weights is None:
        for fid in fids:
            v_sparse[int(fid)] = 1.0
    else:
        for fid, w in zip(fids, weights):
            v_sparse[int(fid)] = float(w)
    v_res = (W_dec.T @ v_sparse).float()
    raw = float(v_res.norm())
    if raw > 1e-8:
        v_res = v_res * (dense_norm / raw)
    return v_res.to(device=dev, dtype=dtype).view(1, 1, -1)


def eval_condition(
    *,
    label: str,
    model,
    tok,
    layers,
    pad_id,
    layer,
    dev,
    dtype,
    judge_instr,
    sys_prompt,
    eval_qs,
    hook_fn,
    workers,
) -> dict:
    prompts, replies = [], []
    for prompt in eval_qs:
        ids, attn = encode_ids(tok, sys_prompt, prompt, dev)
        gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id, layer)
        replies.append(gen_text(hook_fn))
        prompts.append(prompt)
    scores = judge_batch(judge_instr, sys_prompt, prompts, replies, workers=workers)
    return {
        "label": label,
        "sys_prompt": "pos" if "pos" in label else "neg",
        "mean_trait": mean_score(scores),
        "scores": scores,
        "samples": [
            {"prompt_idx": i, "score": scores[i], "reply": replies[i][:300]}
            for i in range(len(eval_qs))
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--sae-dir", default=None)
    ap.add_argument("--convergence-json", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--judge-workers", type=int, default=2)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    alpha = float(args.alpha if args.alpha is not None else cfg["alpha"])
    n_questions = args.n_questions if args.n_questions is not None else int(cfg["n_questions"])
    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha, cli_nq=args.n_questions)
    tag = f"l{layer}"
    sae_dir = Path(args.sae_dir or cfg["sae_dir"])
    convergence_json = Path(
        args.convergence_json or sae_dir / f"feature_convergence_{tag}_k20.json"
    )
    out = Path(args.out or sae_dir / f"sufficiency_baseline_matrix_{tag}.json")
    run_id = cfg["run_id"]

    bundle_path = Path(f"persona_runs/{run_id}/artifacts/trait_bundle.json")
    vectors_path = Path(f"persona_runs/{run_id}/vectors/persona_vectors.pt")
    bundle = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    pos_sys = bundle.pos_system_prompt
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:n_questions]

    v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()
    dense_norm = float((alpha * v_layer).norm().item())

    feature_sets = load_feature_sets(convergence_json, sae_dir)

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev,
        release=SAE_RELEASE,
        sae_id=sae_id_for_layer(layer),
        hidden_state_index=hidden_state_index(layer),
    )
    W_dec = _get_decoder_columns(sae)
    direction_dense = (alpha * v_layer).to(device=dev, dtype=dtype).view(1, 1, -1)
    dense_hook = _steering_hook_fn(1.0, direction_dense, steer_last_token_only=False, hook_calls=[0])

    existing = load_checkpoint(out) if args.resume else None
    conditions: list[dict] = list((existing or {}).get("conditions") or [])
    done_labels = {c["label"] for c in conditions}

    def run_if_missing(label: str, fn) -> None:
        if label in done_labels:
            logger.info("Skip (checkpoint): %s", label)
            return
        row = fn()
        conditions.append(row)
        done_labels.add(label)
        payload = {
            "method": "sufficiency_baseline_matrix",
            "layer": layer,
            "alpha_dense": alpha,
            "dense_inject_norm": round(dense_norm, 3),
            "n_questions": len(eval_qs),
            "feature_sets": feature_sets,
            "conditions": conditions,
            "steering_method": "W_dec.T @ v_sparse scaled to dense_norm (SSV/He et al.)",
            "paper_refs": [
                "He et al. SSV (sparse vector residual steering)",
                "Chen et al. Persona Vectors Appendix M.3.2",
                "AxBench Wu 2025 (DiffMean + prompting baselines)",
            ],
        }
        save_checkpoint(out, payload)
        logger.info("%s mean=%s", label, row.get("mean_trait"))

    if not args.skip_judge:
        run_if_missing(
            "neg_baseline",
            lambda: eval_condition(
                label="neg_baseline",
                model=model, tok=tok, layers=layers, pad_id=pad_id, layer=layer,
                dev=dev, dtype=dtype, judge_instr=judge_instr, sys_prompt=neg_sys,
                eval_qs=eval_qs, hook_fn=None, workers=args.judge_workers,
            ),
        )
        run_if_missing(
            "pos_prompt_baseline",
            lambda: eval_condition(
                label="pos_prompt_baseline",
                model=model, tok=tok, layers=layers, pad_id=pad_id, layer=layer,
                dev=dev, dtype=dtype, judge_instr=judge_instr, sys_prompt=pos_sys,
                eval_qs=eval_qs, hook_fn=None, workers=args.judge_workers,
            ),
        )
        run_if_missing(
            f"dense_diffmean_a{alpha}",
            lambda: eval_condition(
                label=f"dense_diffmean_a{alpha}",
                model=model, tok=tok, layers=layers, pad_id=pad_id, layer=layer,
                dev=dev, dtype=dtype, judge_instr=judge_instr, sys_prompt=neg_sys,
                eval_qs=eval_qs, hook_fn=dense_hook, workers=args.judge_workers,
            ),
        )

        for set_name, spec in feature_sets.items():
            fids = spec["feature_ids"]
            weights = spec.get("weights")
            if not fids:
                continue
            label = f"residual_{set_name}"
            direction = build_residual_direction(
                W_dec, fids, dense_norm, dev, dtype, weights=weights
            )
            hook = _steering_hook_fn(1.0, direction, steer_last_token_only=False, hook_calls=[0])

            def _run_sparse(lbl=label, h=hook, w=weights, f=fids):
                return eval_condition(
                    label=lbl,
                    model=model, tok=tok, layers=layers, pad_id=pad_id, layer=layer,
                    dev=dev, dtype=dtype, judge_instr=judge_instr, sys_prompt=neg_sys,
                    eval_qs=eval_qs, hook_fn=h, workers=args.judge_workers,
                )

            run_if_missing(label, _run_sparse)
            logger.info(
                "%s weighted=%s fids=%s",
                set_name,
                weights is not None,
                fids[:8],
            )

    payload = {
        "method": "sufficiency_baseline_matrix",
        "layer": layer,
        "alpha_dense": alpha,
        "dense_inject_norm": round(dense_norm, 3),
        "n_questions": len(eval_qs),
        "feature_sets": feature_sets,
        "conditions": conditions,
        "steering_method": "W_dec.T @ v_sparse scaled to dense_norm (SSV/He et al.)",
        "paper_refs": [
            "He et al. SSV (sparse vector residual steering)",
            "Chen et al. Persona Vectors Appendix M.3.2",
            "AxBench Wu 2025 (DiffMean + prompting baselines)",
        ],
    }

    save_checkpoint(out, payload)
    print(f"Saved {out}", flush=True)
    for c in conditions:
        print(f"  {c['label']}: mean={c['mean_trait']}", flush=True)


if __name__ == "__main__":
    main()
