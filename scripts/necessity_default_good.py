#!/usr/bin/env python3
"""
Necessity under default-good (positive system prompt): ablate candidate feature sets
during unsteered generation and measure good-trait degradation.

Unlike ablation_necessity_sweep.py (neg_sys + dense-CAA), this tests whether sparse
features are necessary for natural good behavior under pos_system_prompt.
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
from scripts.trait_sae_config import (
    SAE_RELEASE,
    check_override,
    hidden_state_index,
    resolve_trait,
    sae_id_for_layer,
)



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
    v = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for fid in fids:
        v += float(z_mean[fid].item()) * W_dec[fid].float()
    return v


def load_feature_sets(convergence_path: Path) -> dict[str, list[int]]:
    d = json.loads(convergence_path.read_text(encoding="utf-8"))
    sets: dict[str, list[int]] = {}
    core = (d.get("method_invariant_core") or {}).get("2") or []
    if core:
        sets["core_votes2"] = [int(c["feature_id"]) for c in core]
    for name, fids in (d.get("selectors") or {}).items():
        sets[f"{name}_top20"] = [int(x) for x in fids[:20]]
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--convergence-json", default=None)
    ap.add_argument("--feature-sets-json", default=None, help="Override {name: [fids]} JSON")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(args.layer if args.layer is not None else cfg["layer"])
    n_questions = args.n_questions if args.n_questions is not None else int(cfg["n_questions"])
    check_override(cfg, cli_layer=args.layer, cli_nq=args.n_questions)
    tag = f"l{layer}"
    convergence_json = Path(
        args.convergence_json or cfg["sae_dir"] / f"feature_convergence_{tag}_k20.json"
    )
    out = Path(args.out or cfg["sae_dir"] / f"necessity_default_good_{tag}.json")
    run_id = cfg["run_id"]
    paths = {
        "bundle": Path(f"persona_runs/{run_id}/artifacts/trait_bundle.json"),
        "vectors": Path(f"persona_runs/{run_id}/vectors/persona_vectors.pt"),
    }

    bundle = PersonaTraitArtifact.model_validate_json(paths["bundle"].read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    pos_sys = bundle.pos_system_prompt
    eval_qs = bundle.eval_questions[:n_questions]

    if args.feature_sets_json:
        feature_sets = {
            k: [int(x) for x in v]
            for k, v in json.loads(Path(args.feature_sets_json).read_text()).items()
        }
    else:
        feature_sets = load_feature_sets(convergence_json)

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
    W_dec = sae.W_dec.detach().float().cpu()

    def encode_ids(prompt: str):
        msgs = [
            {"role": "system", "content": pos_sys},
            {"role": "user", "content": prompt},
        ]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)
        return ids, attn

    print("=== Capturing prompt-time SAE activations on pos_system_prompt ===", flush=True)
    per_q_z_mean: list[torch.Tensor] = []
    for qi, prompt in enumerate(eval_qs):
        ids, attn = encode_ids(prompt)
        h = capture_hidden(model, layers, layer, ids, attn)
        with torch.no_grad():
            z = sae.encode(h.to(sae_dev)).float()
        z_mean = z[0].mean(0).cpu()
        per_q_z_mean.append(z_mean)
        print(f"  Q{qi+1} top-3 |z|: {torch.topk(z_mean.abs(), 3).indices.tolist()}", flush=True)
    global_z_mean = torch.stack(per_q_z_mean).mean(0)

    def gen_with_hook(ids, attn, hook_fn=None) -> str:
        handles = []
        if hook_fn is not None:
            handles.append(layers[layer].register_forward_hook(hook_fn))
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
            return int(score_transcript(judge_instr, pos_sys, prompt, reply).score)
        except Exception as exc:
            print(f"  judge error: {exc}", flush=True)
            return None

    rows = []
    # Baseline: no ablation
    feature_sets_ordered = {"baseline_none": []}
    feature_sets_ordered.update(feature_sets)

    for set_name, fids in feature_sets_ordered.items():
        v_ablate = build_ablation_vector(W_dec, fids, global_z_mean)
        ablate_hook = None
        if fids and float(v_ablate.norm()) > 1e-8:
            ablate_dir = v_ablate.to(device=dev, dtype=dtype).view(1, 1, -1)
            ablate_hook = _steering_hook_fn(
                -1.0, ablate_dir, steer_last_token_only=False, hook_calls=[0]
            )

        print(f"\n=== {set_name} (ablate n={len(fids)}) ===", flush=True)
        scores, coherences, samples = [], [], []
        for qi, prompt in enumerate(eval_qs):
            ids, attn = encode_ids(prompt)
            reply = gen_with_hook(ids, attn, ablate_hook)
            coh = coherence_heuristic(reply)
            s = judge_reply(prompt, reply)
            scores.append(s)
            coherences.append(coh)
            samples.append({"q_idx": qi, "score": s, "coherence": coh, "reply": reply[:300]})
            print(f"  Q{qi+1} trait={s} coherent={coh['coherent']}", flush=True)

        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  MEAN trait={mean}", flush=True)
        rows.append(
            {
                "set_name": set_name,
                "n_features": len(fids),
                "feature_ids": fids,
                "ablation_norm": round(float(v_ablate.norm()), 3),
                "mean_trait": mean,
                "scores": scores,
                "mean_coherence_heuristic": round(
                    sum(c["score"] for c in coherences) / len(coherences), 1
                ),
                "frac_incoherent": round(
                    sum(1 for c in coherences if not c["coherent"]) / len(coherences), 3
                ),
                "samples": samples,
            }
        )

    baseline_mean = next(r["mean_trait"] for r in rows if r["set_name"] == "baseline_none")
    for r in rows:
        if r["mean_trait"] is not None and baseline_mean is not None:
            r["delta_from_baseline"] = round(r["mean_trait"] - baseline_mean, 1)

    result = {
        "method": "necessity_default_good",
        "condition": "pos_system_prompt_unsteered",
        "layer": layer,
        "n_questions": len(eval_qs),
        "ablation_mode": "subtract_weighted_decoder_on_pos_prompt_z",
        "sets": rows,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
