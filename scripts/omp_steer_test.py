#!/usr/bin/env python3
"""
Steer with OMP-decomposed vectors at different K, judge via Vertex.
Tests whether the OMP sparse approximation of v_dense actually steers the trait.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import DEFAULT_ALPHA, DEFAULT_KS, SAE_RELEASE, check_override, resolve_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default=None, help="good|evil|lawful|chaotic")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--sae-id", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--ks", default=DEFAULT_KS)
    ap.add_argument("--decomp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-ref", action="store_true", help="skip baseline and dense CAA")
    args = ap.parse_args()

    if args.trait:
        cfg = resolve_trait(args.trait)
    else:
        cfg = resolve_trait("good")
        if args.run_id:
            cfg["run_id"] = args.run_id
        if args.layer is not None:
            cfg["layer"] = args.layer
            from scripts.trait_sae_config import hidden_state_index, run_paths, sae_id_for_layer

            cfg["sae_id"] = args.sae_id or sae_id_for_layer(cfg["layer"])
            cfg["hs_index"] = hidden_state_index(cfg["layer"])
            p = run_paths(cfg["run_id"], cfg["layer"])
            cfg.update(p)

    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha)
    layer = int(cfg["layer"])
    vectors_path = Path(args.vectors or cfg["vectors"])
    bundle_path = Path(args.bundle or cfg["bundle"])
    decomp_path = Path(args.decomp or cfg["decomp"])
    out_path = Path(args.out or cfg["steer"])
    sae_id = args.sae_id or cfg["sae_id"]
    hs_index = cfg["hs_index"]
    alpha = args.alpha if args.alpha is not None else float(cfg["alpha"])

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    bundle = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
    v_layer = v_full[layer].float()

    omp = json.loads(decomp_path.read_text())

    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=sae_id,
        hidden_state_index=hs_index,
    )
    W = sae.W_dec.detach().float()

    def build_omp_vec(k):
        v = torch.zeros(W.shape[1])
        for entry in omp["decomposition"][:k]:
            v += entry["coefficient"] * W[entry["feature_id"]]
        return v

    def gen_text(ids, attn, hook_fn=None):
        handle = None
        if hook_fn is not None:
            handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True).strip()

    def judge_reply(prompt, reply):
        if len(reply.strip()) < 20:
            return None
        try:
            js = score_transcript(judge_instr, neg_sys, prompt, reply)
            return int(js.score)
        except Exception:
            return None

    def run_condition(label, direction, qs):
        d = direction.to(device=dev, dtype=dtype).view(1, 1, -1)
        scores = []
        for qi, prompt in enumerate(qs):
            msgs = [
                {"role": "system", "content": neg_sys},
                {"role": "user", "content": prompt},
            ]
            enc = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook = _steering_hook_fn(
                alpha, d, steer_last_token_only=False, hook_calls=[0]
            )
            reply = gen_text(ids, attn, hook)
            s = judge_reply(prompt, reply)
            scores.append(s)
            print(f"  [{label}] Q{qi+1} score={s}")
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean}")
        return {"label": label, "mean": mean, "scores": scores}

    print(f"=== OMP steer trait={cfg.get('trait', cfg['run_id'])} layer={layer} alpha={alpha} ===")

    results = []

    if not args.skip_ref:
        print("=== BASELINE ===")
        base_scores = []
        for qi, prompt in enumerate(eval_qs):
            msgs = [
                {"role": "system", "content": neg_sys},
                {"role": "user", "content": prompt},
            ]
            enc = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            reply = gen_text(ids, attn)
            s = judge_reply(prompt, reply)
            base_scores.append(s)
            print(f"  [BASELINE] Q{qi+1} score={s}")
        valid = [s for s in base_scores if s is not None]
        base_mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [BASELINE] MEAN={base_mean}")
        results.append({"label": "BASELINE", "mean": base_mean, "scores": base_scores})

        print("=== DENSE_CAA ===")
        results.append(run_condition("DENSE_CAA", v_layer, eval_qs))

    for K in ks:
        print(f"=== OMP_K{K} ===")
        v_omp = build_omp_vec(K)
        cos = torch.nn.functional.cosine_similarity(
            v_layer.unsqueeze(0), v_omp.unsqueeze(0)
        ).item()
        print(
            f"  cos(v_omp, v_dense) = {cos:.4f}, "
            f"norm ratio = {v_omp.norm()/v_layer.norm():.2f}"
        )
        row = run_condition(f"OMP_K{K}", v_omp, eval_qs)
        row["cosine"] = round(cos, 4)
        row["k"] = K
        results.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trait": cfg.get("trait"),
        "run_id": cfg["run_id"],
        "layer": layer,
        "alpha": alpha,
        "sae_id": sae_id,
        "decomp": str(decomp_path),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['label']:>12s}  mean={r['mean']}  scores={r['scores']}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
