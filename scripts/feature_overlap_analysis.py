#!/usr/bin/env python3
"""
Analyze: are the top SAE features (by steering delta) the same across questions?
For each question pair, compute Jaccard overlap at various K.
Also: decoder cosine with Good vector for top features per question.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger()


def main():
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
    from app.persona.sae_causality import _sae_device
    from app.phase2 import load_sae_for_layer
    from app.persona.schemas import PersonaTraitArtifact

    bundle = PersonaTraitArtifact.model_validate_json(
        Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[:5]

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

    # Decoder cosine with Good vector (on CPU to avoid OOM)
    W_dec = sae.W_dec.data.float().cpu()  # [d_sae, d_model]
    v_unit = v16.cpu() / v16.norm()
    dec_cos = torch.nn.functional.cosine_similarity(
        W_dec, v_unit.unsqueeze(0), dim=1
    )  # [d_sae]

    per_q_data = []

    for qi, prompt in enumerate(eval_qs):
        logger.info("Processing Q%d...", qi + 1)
        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        ids = ids.to(dev)

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

        h_base = capture_h(False)
        h_steer = capture_h(True)

        with torch.no_grad():
            z_base = sae.encode(h_base.to(sae_dev))
            z_steer = sae.encode(h_steer.to(sae_dev))

        z_diff = z_steer[0].float().mean(0) - z_base[0].float().mean(0)
        z_steer_mean = z_steer[0].float().mean(0)
        z_base_mean = z_base[0].float().mean(0)

        sorted_idx = torch.argsort(z_diff.abs(), descending=True).cpu()
        n_active = int((z_diff.abs() > 1e-6).sum().item())
        z_diff = z_diff.cpu()
        z_steer_mean = z_steer_mean.cpu()
        z_base_mean = z_base_mean.cpu()

        # Top features at various K
        top_feats = {}
        for K in [100, 200, 500, 1000, 2000]:
            k_val = min(K, n_active)
            fids = sorted_idx[:k_val].tolist()
            top_feats[K] = set(fids)

        # Decoder cosine stats for top-1000 features
        top1k = sorted_idx[:min(1000, n_active)]
        top1k_cos = dec_cos[top1k]
        pos_aligned = int((top1k_cos > 0.05).sum().item())
        neg_aligned = int((top1k_cos < -0.05).sum().item())
        neutral = len(top1k) - pos_aligned - neg_aligned

        # Per-feature detail for top-20
        top20_detail = []
        for rank, fi in enumerate(sorted_idx[:20].tolist()):
            top20_detail.append({
                "rank": rank,
                "feature_id": fi,
                "z_diff": round(float(z_diff[fi].item()), 4),
                "z_steer": round(float(z_steer_mean[fi].item()), 4),
                "z_base": round(float(z_base_mean[fi].item()), 4),
                "dec_cos_good": round(float(dec_cos[fi].item()), 4),
            })

        per_q_data.append({
            "q_idx": qi,
            "prompt_short": prompt[:80],
            "n_active_delta": n_active,
            "top_feats": top_feats,
            "top1k_dec_cos": {
                "mean": round(float(top1k_cos.mean().item()), 4),
                "pos_aligned": pos_aligned,
                "neg_aligned": neg_aligned,
                "neutral": neutral,
                "max": round(float(top1k_cos.max().item()), 4),
                "min": round(float(top1k_cos.min().item()), 4),
            },
            "top20": top20_detail,
        })

    # Compute pairwise Jaccard overlap
    print("\n=== PAIRWISE JACCARD OVERLAP (top-K features) ===")
    for K in [100, 200, 500, 1000, 2000]:
        print(f"\n--- K={K} ---")
        header = "     " + "".join(f"  Q{j+1}  " for j in range(5))
        print(header)
        for i in range(5):
            row = f"Q{i+1}  "
            for j in range(5):
                si = per_q_data[i]["top_feats"][K]
                sj = per_q_data[j]["top_feats"][K]
                if len(si | sj) == 0:
                    jac = 0.0
                else:
                    jac = len(si & sj) / len(si | sj)
                row += f" {jac:.3f} "
            print(row)

    # Common features across ALL questions
    print("\n=== FEATURES IN COMMON ACROSS ALL 5 QUESTIONS ===")
    for K in [200, 500, 1000, 2000]:
        common = per_q_data[0]["top_feats"][K]
        for q in per_q_data[1:]:
            common = common & q["top_feats"][K]
        print(f"K={K}: {len(common)} features in common out of {K}")
        if common and K <= 500:
            common_list = sorted(common)[:30]
            cos_vals = [float(dec_cos[f].item()) for f in common_list]
            print(f"  Sample common features: {common_list[:15]}")
            print(f"  Their dec_cos_good: {[round(c, 3) for c in cos_vals[:15]]}")

    # Per-question decoder alignment
    print("\n=== DECODER ALIGNMENT WITH GOOD VECTOR (top-1000 features) ===")
    for q in per_q_data:
        stats = q["top1k_dec_cos"]
        print(f"Q{q['q_idx']+1}: mean_cos={stats['mean']:.4f}  "
              f"pos={stats['pos_aligned']}  neg={stats['neg_aligned']}  "
              f"neutral={stats['neutral']}  range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    # Top-20 feature detail per question
    print("\n=== TOP-20 FEATURES PER QUESTION ===")
    for q in per_q_data:
        print(f"\nQ{q['q_idx']+1} ({q['prompt_short']}...)")
        print(f"  {'rank':>4}  {'feat_id':>8}  {'z_diff':>8}  {'z_steer':>8}  {'z_base':>8}  {'dec_cos':>8}")
        for f in q["top20"]:
            print(f"  {f['rank']:4d}  {f['feature_id']:8d}  {f['z_diff']:8.3f}  "
                  f"{f['z_steer']:8.3f}  {f['z_base']:8.3f}  {f['dec_cos_good']:8.4f}")

    # Serialize for later use (convert sets to lists)
    out = []
    for q in per_q_data:
        qc = dict(q)
        qc["top_feats"] = {str(k): sorted(v) for k, v in qc["top_feats"].items()}
        out.append(qc)

    out_path = Path("logs/feature_overlap_262k.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
