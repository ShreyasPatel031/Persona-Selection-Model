#!/usr/bin/env python3
"""Minimal test: dense CAA at L15 alpha=2.0 on all 20 eval questions.

Answers the question: does the dense vector actually work on the full eval?
If this doesn't score 90+, the whole SAE decomposition framework is pointless.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from scripts.trait_sae_config import resolve_trait


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    alpha = float(args.alpha if args.alpha is not None else cfg["alpha"])
    nq = args.n_questions

    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:nq]

    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype
    direction = (alpha * v_layer).to(device=dev, dtype=dtype).view(1, 1, -1)
    hook = _steering_hook_fn(1.0, direction, steer_last_token_only=False, hook_calls=[0])

    print(f"=== Dense CAA test: trait={cfg['trait']} layer={layer} alpha={alpha} nq={nq} ===", flush=True)
    print(f"Direction norm: {direction.norm().item():.2f}", flush=True)

    baseline_scores = []
    dense_scores = []
    results = []

    for qi, prompt in enumerate(eval_qs):
        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        with torch.no_grad():
            gen_bl = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, pad_token_id=pad_id, use_cache=True)
        reply_bl = tok.decode(gen_bl[0, ids.shape[-1]:], skip_special_tokens=True).strip()

        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            gen_st = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, pad_token_id=pad_id, use_cache=True)
        handle.remove()
        reply_st = tok.decode(gen_st[0, ids.shape[-1]:], skip_special_tokens=True).strip()

        bl_score = None
        st_score = None
        for attempt in range(5):
            try:
                if bl_score is None and len(reply_bl.strip()) >= 20:
                    bl_score = int(score_transcript(judge_instr, neg_sys, prompt, reply_bl).score)
            except Exception:
                pass
            try:
                if st_score is None and len(reply_st.strip()) >= 20:
                    st_score = int(score_transcript(judge_instr, neg_sys, prompt, reply_st).score)
            except Exception:
                pass
            if bl_score is not None and st_score is not None:
                break
            if attempt < 4:
                time.sleep(2 ** attempt)

        baseline_scores.append(bl_score)
        dense_scores.append(st_score)
        results.append({
            "q_idx": qi,
            "question": prompt[:120],
            "baseline_score": bl_score,
            "dense_score": st_score,
            "baseline_reply": reply_bl[:300],
            "dense_reply": reply_st[:300],
        })
        print(f"  Q{qi+1:2d} baseline={bl_score:>4}  dense={st_score:>4}  [{prompt[:60]}]", flush=True)

    valid_bl = [s for s in baseline_scores if s is not None]
    valid_st = [s for s in dense_scores if s is not None]
    bl_mean = round(sum(valid_bl) / len(valid_bl), 1) if valid_bl else None
    st_mean = round(sum(valid_st) / len(valid_st), 1) if valid_st else None

    print(f"\n{'='*80}", flush=True)
    print(f"BASELINE (neg_sys, no steering): {bl_mean}  scores={baseline_scores}", flush=True)
    print(f"DENSE CAA (L{layer} alpha={alpha}):    {st_mean}  scores={dense_scores}", flush=True)
    print(f"{'='*80}", flush=True)

    payload = {
        "test": "dense_caa_20q",
        "trait": cfg["trait"],
        "layer": layer,
        "alpha": alpha,
        "n_questions": nq,
        "baseline_mean": bl_mean,
        "dense_mean": st_mean,
        "baseline_scores": baseline_scores,
        "dense_scores": dense_scores,
        "results": results,
    }

    out = Path(args.out or cfg["sae_dir"] / f"dense_caa_20q_l{layer}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
