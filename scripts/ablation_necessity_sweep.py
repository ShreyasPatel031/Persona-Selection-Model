#!/usr/bin/env python3
"""
Phase 1: Necessity profile — ablate top-M SAE features during dense-CAA generation.

Ranks features by mean |z| on dense-CAA prompt forward passes, then subtracts their
decoder contributions (weighted by prompt-time activations) during steered generation.
This avoids per-token 262k SAE encode/decode in hooks (too slow on CPU).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, check_override, hidden_state_index, resolve_trait, sae_id_for_layer

ALPHA = 1.5
LAYER = 16
SAE_ID = "layer_16_width_262k_l0_small"
SAE_HS_INDEX = 17
RUN_BUNDLE = Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json")
RUN_VECTORS = Path("persona_runs/dnd_good_scale/vectors/persona_vectors.pt")


def coherence_heuristic(text: str) -> dict:
    t = text.strip()
    if len(t) < 20:
        return {"coherent": False, "reason": "too_short", "score": 0}
    dot_frac = t.count(".") / max(len(t), 1)
    if dot_frac > 0.5:
        return {"coherent": False, "reason": "dot_collapse", "score": 10}
    words = t.split()
    if len(words) >= 6:
        from collections import Counter

        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        top = Counter(trigrams).most_common(1)[0]
        if top[1] >= 5:
            return {"coherent": False, "reason": "trigram_repeat", "score": 20}
    if re.search(r"(.)\1{20,}", t):
        return {"coherent": False, "reason": "char_repeat", "score": 15}
    uniq_ratio = len(set(words)) / max(len(words), 1)
    score = min(100, int(uniq_ratio * 100))
    return {"coherent": uniq_ratio > 0.35, "reason": "ok", "score": score}


def capture_hidden(model, layers, layer_idx, ids, attn, hook_fn=None):
    captured = []

    def cap(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor) and h.dim() == 3:
            captured.append(h.detach().clone())
        return output

    handles = []
    if hook_fn is not None:
        handles.append(layers[layer_idx].register_forward_hook(hook_fn))
    handles.append(layers[layer_idx].register_forward_hook(cap))
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attn, use_cache=False)
    for hh in handles:
        hh.remove()
    return captured[0]


def build_ablation_vector(W_dec: torch.Tensor, fids: list[int], z_mean: torch.Tensor) -> torch.Tensor:
    """v_ablate = sum_i z_mean[i] * W_dec[i] for ablated features."""
    if not fids:
        return torch.zeros(W_dec.shape[1])
    v = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for fid in fids:
        v += float(z_mean[fid].item()) * W_dec[fid].float()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--m-values", default="0,10,50,100,200,500")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    global LAYER, SAE_ID, SAE_HS_INDEX, RUN_BUNDLE, RUN_VECTORS
    cfg = resolve_trait(args.trait)
    LAYER = int(args.layer if args.layer is not None else cfg["layer"])
    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha)
    SAE_ID = cfg["sae_id"]
    SAE_HS_INDEX = cfg["hs_index"]
    RUN_BUNDLE = Path(cfg["bundle"])
    RUN_VECTORS = Path(cfg["vectors"])
    alpha = float(args.alpha if args.alpha is not None else cfg["alpha"])
    tag = f"l{LAYER}"
    out_path = Path(args.out or cfg["sae_dir"] / f"ablation_necessity_262k_{tag}.json")

    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]

    bundle = PersonaTraitArtifact.model_validate_json(RUN_BUNDLE.read_text())
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    v_full = torch.load(RUN_VECTORS, map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[LAYER].float()

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype
    direction_dense = v_layer.to(device=dev, dtype=dtype).view(1, 1, -1)

    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev,
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        hidden_state_index=SAE_HS_INDEX,
    )
    W_dec = sae.W_dec.detach().float().cpu()

    def encode_ids(prompt: str):
        msgs = [
            {"role": "system", "content": neg_sys},
            {"role": "user", "content": prompt},
        ]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)
        return ids, attn

    steer_hook = _steering_hook_fn(
        alpha, direction_dense, steer_last_token_only=False, hook_calls=[0]
    )

    print("=== Ranking features on dense-CAA prompt activations ===", flush=True)
    z_sum = torch.zeros(int(sae.cfg.d_sae), dtype=torch.float64)
    per_q_rankings: list[dict] = []
    per_q_z_mean: list[torch.Tensor] = []

    for qi, prompt in enumerate(eval_qs):
        ids, attn = encode_ids(prompt)
        h = capture_hidden(model, layers, LAYER, ids, attn, steer_hook)
        with torch.no_grad():
            z = sae.encode(h.to(sae_dev)).float()
        z_mean = z[0].mean(0).cpu()
        per_q_z_mean.append(z_mean)
        z_sum += z_mean.abs().double()
        top10 = torch.topk(z_mean.abs(), k=10).indices.tolist()
        per_q_rankings.append({"q_idx": qi, "top10_by_abs_z": top10})
        print(f"  Q{qi+1} top-3 features: {top10[:3]}", flush=True)

    z_rank = z_sum / max(len(eval_qs), 1)
    ranked_fids = torch.argsort(z_rank, descending=True).tolist()
    global_z_mean = torch.stack(per_q_z_mean).mean(0)

    def gen_with_hooks(ids, attn, extra_hook=None) -> str:
        handles = [layers[LAYER].register_forward_hook(steer_hook)]
        if extra_hook is not None:
            handles.append(layers[LAYER].register_forward_hook(extra_hook))
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
        for hh in handles:
            hh.remove()
        return tok.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True).strip()

    def judge_reply(prompt: str, reply: str):
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

    sweep_rows = []
    for m in m_values:
        fids = ranked_fids[:m] if m > 0 else []
        v_ablate = build_ablation_vector(W_dec, fids, global_z_mean)
        ablate_dir = v_ablate.to(device=dev, dtype=dtype).view(1, 1, -1)
        ablate_hook = None
        if m > 0 and float(v_ablate.norm()) > 1e-8:
            ablate_hook = _steering_hook_fn(
                -1.0, ablate_dir, steer_last_token_only=False, hook_calls=[0]
            )

        print(f"\n=== M={m} (subtract {len(fids)} feature contributions) ===", flush=True)
        scores = []
        coherences = []
        samples = []
        for qi, prompt in enumerate(eval_qs):
            ids, attn = encode_ids(prompt)
            reply = gen_with_hooks(ids, attn, ablate_hook)
            coh = coherence_heuristic(reply)
            s = judge_reply(prompt, reply)
            scores.append(s)
            coherences.append(coh)
            samples.append(
                {
                    "q_idx": qi,
                    "score": s,
                    "coherence": coh,
                    "reply": reply[:300],
                }
            )
            print(
                f"  Q{qi+1} trait={s} coherent={coh['coherent']} ({coh['reason']})",
                flush=True,
            )
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        mean_coh = round(
            sum(c["score"] for c in coherences) / len(coherences), 1
        )
        frac_incoherent = round(
            sum(1 for c in coherences if not c["coherent"]) / len(coherences), 3
        )
        print(f"  MEAN trait={mean} coherence_heuristic={mean_coh}", flush=True)
        sweep_rows.append(
            {
                "m_ablated": m,
                "n_features_ablated": len(fids),
                "ablation_norm": round(float(v_ablate.norm()), 3),
                "ablated_feature_ids_sample": fids[:15],
                "mean_trait": mean,
                "scores": scores,
                "mean_coherence_heuristic": mean_coh,
                "frac_incoherent": frac_incoherent,
                "samples": samples,
            }
        )

    result = {
        "method": "ablation_necessity_sweep",
        "ablation_mode": "fixed_decoder_subtraction_weighted_by_prompt_z",
        "layer": LAYER,
        "alpha": alpha,
        "max_new_tokens": args.max_new_tokens,
        "n_questions": len(eval_qs),
        "ranking": {
            "criterion": "mean_abs_z_on_dense_caa_prompt",
            "top20_features": ranked_fids[:20],
            "per_question_top10": per_q_rankings,
        },
        "sweep": sweep_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
