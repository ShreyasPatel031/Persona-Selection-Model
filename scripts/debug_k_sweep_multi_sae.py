#!/usr/bin/env python3
"""
Multi-SAE K-sweep: compare 16k vs 262k at layer 16 with finer K grid + judge scores.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()

SAE_CONFIGS = [
    ("16k", "layer_16_width_16k_l0_small"),
    ("262k", "layer_16_width_262k_l0_small"),
]

DEFAULT_KS = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]


def sae_recon_metrics(sae, v: torch.Tensor) -> dict:
    from app.persona.sae_causality import _sae_device

    dev = _sae_device(sae)
    x = v.unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        z = sae.encode(x)
        recon = sae.decode(z)[0, 0].float()
    v_f = v.float().to(recon.device)
    cos = torch.nn.functional.cosine_similarity(v_f.unsqueeze(0), recon.unsqueeze(0)).item()
    n_active = int((z[0, 0].abs() > 1e-6).sum().item())
    return {
        "cosine": round(cos, 4),
        "orig_norm": round(float(v.float().norm().item()), 1),
        "recon_norm": round(float(recon.norm().item()), 1),
        "n_active": n_active,
        "d_sae": int(sae.cfg.d_sae),
    }


def run_sweep(
    *,
    sae_label: str,
    sae_id: str,
    eval_qs: list[str],
    neg_sys: str,
    model,
    tok,
    dev,
    layers,
    sae,
    v16: torch.Tensor,
    alpha: float,
    ks: list[int | str],
    pad_id: int,
    score: bool,
    judge_instr: str,
    q_index_map: list[int],
) -> dict:
    from app.persona.sae_causality import sae_feature_clamp_hook_fn, _sae_device
    from app.persona.steering_demo import _steering_hook_fn

    dtype = next(model.parameters()).dtype
    sae_dev = _sae_device(sae)
    direction_dense = v16.to(device=dev, dtype=dtype).view(1, 1, -1)

    recon = sae_recon_metrics(sae, v16.to(sae_dev))
    logger.info("[%s] Good vector SAE recon: %s", sae_label, recon)

    results: dict = {"sae_label": sae_label, "sae_id": sae_id, "recon": recon, "questions": []}

    if score:
        from app.persona.judge_vertex import score_transcript

    for qi, prompt in enumerate(eval_qs):
        q_num = q_index_map[qi] + 1
        q_out: dict = {"q_idx": q_index_map[qi], "prompt": prompt, "conditions": {}}
        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)
        attn = torch.ones_like(ids, dtype=torch.long, device=dev)

        def gen_text(hook_fn=None) -> str:
            handle = None
            if hook_fn is not None:
                handle = layers[16].register_forward_hook(hook_fn)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids, attention_mask=attn, max_new_tokens=200,
                    do_sample=False, pad_token_id=pad_id, use_cache=True,
                )
            if handle is not None:
                handle.remove()
            return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()

        def capture_h(steer: bool):
            captured = []
            def cap(_m, _inp, output):
                h = output if isinstance(output, torch.Tensor) else output[0]
                if isinstance(h, torch.Tensor) and h.dim() == 3:
                    captured.append(h.detach().clone())
                return output
            handles = []
            if steer:
                hc = [0]
                handles.append(layers[16].register_forward_hook(
                    _steering_hook_fn(alpha, direction_dense, steer_last_token_only=False, hook_calls=hc)))
            handles.append(layers[16].register_forward_hook(cap))
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            for hh in handles:
                hh.remove()
            return captured[0]

        def score_reply(label: str, reply: str) -> int | None:
            if not score or len(reply.strip()) < 20:
                return None
            try:
                js = score_transcript(judge_instr, neg_sys, prompt, reply)
                return int(js.score)
            except Exception as e:
                logger.warning("Judge failed %s Q%d: %s", label, q_num, e)
                return None

        baseline = gen_text()
        q_out["conditions"]["BASELINE"] = {
            "reply": baseline[:500],
            "score": score_reply("BASELINE", baseline),
        }
        print(f"\n[{sae_label}] Q{q_num} BASELINE score={q_out['conditions']['BASELINE']['score']}")

        dense = gen_text(_steering_hook_fn(
            alpha, direction_dense, steer_last_token_only=False, hook_calls=[0]))
        q_out["conditions"]["DENSE_CAA"] = {
            "reply": dense[:500],
            "score": score_reply("DENSE_CAA", dense),
        }
        print(f"[{sae_label}] Q{q_num} DENSE_CAA score={q_out['conditions']['DENSE_CAA']['score']}")

        h_base = capture_h(False)
        h_steer = capture_h(True)
        with torch.no_grad():
            z_base = sae.encode(h_base.to(sae_dev))
            z_steer = sae.encode(h_steer.to(sae_dev))
        z_diff = z_steer[0].float().mean(0) - z_base[0].float().mean(0)
        z_steer_mean = z_steer[0].float().mean(0)
        sorted_idx = torch.argsort(z_diff.abs(), descending=True)
        total_features = int((z_diff.abs() > 1e-6).sum().item())
        q_out["n_features_delta"] = total_features

        for K in ks:
            if K == "ALL":
                k_val = min(total_features, 10000)
                label = f"K={k_val}" if k_val < total_features else f"K=ALL({k_val})"
            else:
                k_val = min(int(K), total_features)
                label = f"K={K}"
            fids = sorted_idx[:k_val].tolist()
            clamp_vals = [float(z_steer_mean[f].item()) for f in fids]
            hc = [0]
            hook = sae_feature_clamp_hook_fn(
                sae, fids, clamp_vals, hc,
                mode="additive_delta", steer_last_token_only=False,
            )
            reply = gen_text(hook)
            sc = score_reply(label, reply)
            q_out["conditions"][label] = {"reply": reply[:500], "score": sc, "k": k_val}
            print(f"[{sae_label}] Q{q_num} {label} score={sc}")

        results["questions"].append(q_out)

    # Aggregate scores by condition
    agg: dict[str, list[int]] = {}
    for q in results["questions"]:
        for label, row in q["conditions"].items():
            if row.get("score") is not None:
                agg.setdefault(label, []).append(int(row["score"]))
    summary = {
        lbl: {"mean": round(sum(v) / len(v), 1), "n": len(v), "scores": v}
        for lbl, v in sorted(agg.items())
    }
    results["summary"] = summary
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument(
        "--questions",
        default="",
        help="1-based question numbers to run, e.g. '4,5'. Overrides --n-questions.",
    )
    parser.add_argument(
        "--ks",
        default="1000,2000",
        help="Comma-separated K values (default: 1000,2000)",
    )
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--sae", choices=["16k", "262k", "both"], default="both")
    parser.add_argument("--out", default="logs/debug_k_sweep_multi_sae.json")
    args = parser.parse_args()

    ks: list[int | str] = []
    for part in args.ks.split(","):
        part = part.strip()
        if not part:
            continue
        ks.append("ALL" if part.upper() == "ALL" else int(part))
    if not ks:
        ks = DEFAULT_KS

    from app.persona.activations import load_model_and_tokenizer
    from app.persona.judge_vertex import judge_rubric_to_instructions
    from app.persona.steering_demo import _language_model_layers
    from app.phase2 import load_sae_for_layer
    from app.persona.schemas import PersonaTraitArtifact

    bundle = PersonaTraitArtifact.model_validate_json(
        Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = bundle.neg_system_prompt
    all_eval_qs = bundle.eval_questions
    if args.questions.strip():
        q_indices = [int(x.strip()) - 1 for x in args.questions.split(",") if x.strip()]
        eval_qs = [all_eval_qs[i] for i in q_indices if 0 <= i < len(all_eval_qs)]
        q_index_map = [i for i in q_indices if 0 <= i < len(all_eval_qs)]
    else:
        eval_qs = all_eval_qs[: args.n_questions]
        q_index_map = list(range(len(eval_qs)))
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    v_full = torch.load(
        "persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
        map_location="cpu", weights_only=False,
    )["v"]
    v16 = v_full[16].float()
    alpha = 1.5

    configs = [c for c in SAE_CONFIGS if args.sae in ("both", c[0])]
    all_results = []

    for sae_label, sae_id in configs:
        logger.info("Loading SAE %s (%s) on %s...", sae_label, sae_id, dev)
        sae, _ = load_sae_for_layer(
            dev, release="gemma-scope-2-4b-it-res-all",
            sae_id=sae_id, hidden_state_index=17,
        )
        res = run_sweep(
            sae_label=sae_label,
            sae_id=sae_id,
            eval_qs=eval_qs,
            neg_sys=neg_sys,
            model=model,
            tok=tok,
            dev=dev,
            layers=layers,
            sae=sae,
            v16=v16,
            alpha=alpha,
            ks=ks,
            pad_id=pad_id,
            score=not args.no_score,
            judge_instr=judge_instr,
            q_index_map=q_index_map,
        )
        all_results.append(res)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for res in all_results:
        print(f"\n--- {res['sae_label']} (recon cos={res['recon']['cosine']}) ---")
        for label in ["BASELINE", "DENSE_CAA"] + [f"K={k}" for k in ks if k != "ALL"]:
            if label in res.get("summary", {}):
                s = res["summary"][label]
                print(f"  {label:12s}  mean={s['mean']:5.1f}  scores={s['scores']}")
        for label, s in res.get("summary", {}).items():
            if label.startswith("K=") and label not in [f"K={k}" for k in ks if k != "ALL"]:
                print(f"  {label:12s}  mean={s['mean']:5.1f}  scores={s['scores']}")

    print(f"\nWrote {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
