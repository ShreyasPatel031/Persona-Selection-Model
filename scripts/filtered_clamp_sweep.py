#!/usr/bin/env python3
"""
Filtered clamp sweep: select features by |z_diff * dec_cos_good| instead of raw |z_diff|.
Also try: only Good-aligned features (dec_cos > 0), only anti-Good (dec_cos < 0, clamp down).
Compare against raw |z_diff| baseline.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/filtered_clamp_sweep.json")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
    from app.persona.sae_causality import sae_feature_clamp_hook_fn, _sae_device
    from app.phase2 import load_sae_for_layer
    from app.persona.schemas import PersonaTraitArtifact

    if not args.no_score:
        from app.persona.judge_vertex import score_transcript, judge_rubric_to_instructions

    bundle = PersonaTraitArtifact.model_validate_json(
        Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:5]
    if not args.no_score:
        judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    v_full = torch.load(
        "persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
        map_location="cpu", weights_only=False,
    )["v"]
    v16 = v_full[16].float()
    alpha = 1.5
    direction_dense = v16.to(device=dev, dtype=dtype).view(1, 1, -1)

    logger.info("Loading SAE 262k on %s...", dev)
    sae, _ = load_sae_for_layer(
        dev, release="gemma-scope-2-4b-it-res-all",
        sae_id="layer_16_width_262k_l0_small", hidden_state_index=17,
    )
    sae_dev = _sae_device(sae)

    # Decoder cosine with Good vector (CPU to avoid OOM)
    W_dec = sae.W_dec.data.float().cpu()
    v_unit = v16.cpu() / v16.norm()
    dec_cos = torch.nn.functional.cosine_similarity(W_dec, v_unit.unsqueeze(0), dim=1)

    SELECTION_MODES = [
        ("raw_delta", "Top-K by |z_diff| (baseline)"),
        ("weighted", "Top-K by |z_diff * dec_cos_good|"),
        ("pos_only", "Top-K by z_diff * dec_cos (only Good-aligned, dec_cos>0.05)"),
        ("neg_suppress", "Top-K by |z_diff * dec_cos| (only anti-Good, dec_cos<-0.05, clamp DOWN)"),
        ("combined", "pos_only + neg_suppress merged"),
    ]

    KS = [10, 20, 50, 100, 200, 500]

    def score_reply(label: str, q_num: int, reply: str) -> int | None:
        if args.no_score or len(reply.strip()) < 20:
            return None
        try:
            js = score_transcript(judge_instr, neg_sys, eval_qs[q_num - 1], reply)
            return int(js.score)
        except Exception as e:
            logger.warning("Judge failed %s Q%d: %s", label, q_num, e)
            return None

    all_results = []

    for qi, prompt in enumerate(eval_qs):
        q_num = qi + 1
        logger.info("=== Q%d ===", q_num)
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

        # Baseline + Dense
        baseline = gen_text()
        baseline_sc = score_reply("BASELINE", q_num, baseline)
        print(f"\n[Q{q_num}] BASELINE score={baseline_sc}")

        dense = gen_text(_steering_hook_fn(
            alpha, direction_dense, steer_last_token_only=False, hook_calls=[0]))
        dense_sc = score_reply("DENSE_CAA", q_num, dense)
        print(f"[Q{q_num}] DENSE_CAA score={dense_sc}")

        # Compute feature deltas
        h_base = capture_h(False)
        h_steer = capture_h(True)
        with torch.no_grad():
            z_base = sae.encode(h_base.to(sae_dev))
            z_steer = sae.encode(h_steer.to(sae_dev))

        z_diff = (z_steer[0].float().mean(0) - z_base[0].float().mean(0)).cpu()
        z_steer_mean = z_steer[0].float().mean(0).cpu()

        q_result = {
            "q_idx": qi,
            "prompt": prompt[:100],
            "baseline": {"score": baseline_sc, "reply": baseline[:300]},
            "dense_caa": {"score": dense_sc, "reply": dense[:300]},
            "conditions": {},
        }

        for mode_id, mode_desc in SELECTION_MODES:
            # Compute ranking score per feature
            if mode_id == "raw_delta":
                rank_score = z_diff.abs()
            elif mode_id == "weighted":
                rank_score = (z_diff * dec_cos).abs()
            elif mode_id == "pos_only":
                mask = (dec_cos > 0.05) & (z_diff > 0)
                rank_score = (z_diff * dec_cos) * mask.float()
            elif mode_id == "neg_suppress":
                mask = (dec_cos < -0.05) & (z_diff < 0)
                rank_score = (z_diff * dec_cos).abs() * mask.float()
            elif mode_id == "combined":
                pos_mask = (dec_cos > 0.05) & (z_diff > 0)
                neg_mask = (dec_cos < -0.05) & (z_diff < 0)
                rank_score = (z_diff * dec_cos).abs() * (pos_mask | neg_mask).float()
            else:
                continue

            sorted_idx = torch.argsort(rank_score, descending=True)
            n_nonzero = int((rank_score > 1e-6).sum().item())

            for K in KS:
                k_val = min(K, n_nonzero)
                if k_val == 0:
                    continue
                label = f"{mode_id}_K{K}"
                fids = sorted_idx[:k_val].tolist()
                clamp_vals = [float(z_steer_mean[f].item()) for f in fids]

                # Stats about selected features
                sel_cos = dec_cos[sorted_idx[:k_val]]
                mean_abs_cos = float(sel_cos.abs().mean().item())
                mean_delta = float(z_diff[sorted_idx[:k_val]].abs().mean().item())

                hc = [0]
                hook = sae_feature_clamp_hook_fn(
                    sae, fids, clamp_vals, hc,
                    mode="additive_delta", steer_last_token_only=False,
                )
                reply = gen_text(hook)
                sc = score_reply(label, q_num, reply)
                q_result["conditions"][label] = {
                    "score": sc,
                    "reply": reply[:300],
                    "k_actual": k_val,
                    "mean_abs_dec_cos": round(mean_abs_cos, 4),
                    "mean_abs_delta": round(mean_delta, 2),
                }
                print(f"[Q{q_num}] {label:25s} score={sc}  k={k_val}  "
                      f"mean|cos|={mean_abs_cos:.3f}  mean|delta|={mean_delta:.1f}")

        all_results.append(q_result)

    # Summary table
    print("\n=== SUMMARY (mean scores across questions) ===")
    agg: dict[str, list[int]] = {}
    for qr in all_results:
        if qr["baseline"]["score"] is not None:
            agg.setdefault("BASELINE", []).append(qr["baseline"]["score"])
        if qr["dense_caa"]["score"] is not None:
            agg.setdefault("DENSE_CAA", []).append(qr["dense_caa"]["score"])
        for label, row in qr["conditions"].items():
            if row.get("score") is not None:
                agg.setdefault(label, []).append(row["score"])

    for label in ["BASELINE", "DENSE_CAA"]:
        if label in agg:
            vals = agg[label]
            print(f"  {label:25s}  mean={sum(vals)/len(vals):5.1f}  scores={vals}")

    for mode_id, _ in SELECTION_MODES:
        for K in KS:
            label = f"{mode_id}_K{K}"
            if label in agg:
                vals = agg[label]
                print(f"  {label:25s}  mean={sum(vals)/len(vals):5.1f}  scores={vals}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
