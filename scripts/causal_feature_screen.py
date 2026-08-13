#!/usr/bin/env python3
"""
Causal feature screening for a persona trait (Good) at layer 16.

Motivation (literature):
  Input-side feature selection -- decoder cosine to v_dense, or STA activation
  difference (z_pos - z_neg) -- does NOT identify causally effective steering
  features. Three independent 2024-2025 results converge on this:
    * Gur-Arieh et al., "SAEs Are Good for Steering" (EMNLP 2025): output score
      >> input score for steering efficacy (2-3x).
    * GradSAE (EMNLP 2025): rank latents by output-side gradient attribution.
    * Chalnev/Siu/Conmy, SAE-TS: select by measured *effect*, not presence;
      features with similar effects need not share decoder directions.

This script selects features by causal effect, in three phases:

  Phase A (cheap, output-side):  GradSAE-style attribution. For each contrast
      prompt, replace the L16 residual with its SAE reconstruction so the forward
      pass depends on the latent z, then backprop a downstream trait readout
      (projection of late-layer residuals onto the persona direction) to z.
      attribution_i = mean_t( z_i * dL/dz_i ).  Ranks ALL features in ~1 pass.

  Phase B (validation, Vertex):  Take top-N attributed features. Steer with ONLY
      that one feature's decoder column (unit, scaled to match dense-CAA norm),
      generate, Vertex-judge. The judge score IS the per-feature causal effect.

  Phase C (sparsity):  Greedily add features in causal-score order, re-judge the
      cumulative steering vector, and find where trait recovery plateaus. Emits
      the final weighted decomposition Good = sum_i w_i * feature_i.

Also computes overlap between the causal set and the STA "naturally-active" set.

Run phases independently with --phase {A,B,C,all}. Artifacts are JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.trait_sae_config import check_override, hidden_state_index, resolve_trait, sae_id_for_layer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("causal_screen")

SAE_CONFIGS = {
    "16k": "layer_16_width_16k_l0_small",
    "262k": "layer_16_width_262k_l0_small",
}
LAYER_IDX = 16
SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
SAE_HS_INDEX = 17
RUN_VECTORS = Path("persona_runs/dnd_good_scale/vectors/persona_vectors.pt")


# --------------------------------------------------------------------------- #
# Shared loading
# --------------------------------------------------------------------------- #
def load_everything(sae_label: str):
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers
    from app.phase2 import load_sae_for_layer

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    sae_id = SAE_CONFIGS[sae_label]
    # 262k SAE is ~2.5GB+; keep on CPU so Gemma-4B fits on T4 GPU.
    sae_dev = torch.device("cpu") if sae_label == "262k" else dev
    logger.info("Loading SAE %s (%s) on %s ...", sae_label, sae_id, sae_dev)
    sae, _ = load_sae_for_layer(
        sae_dev, release=SAE_RELEASE, sae_id=sae_id, hidden_state_index=SAE_HS_INDEX
    )

    v_full = torch.load(RUN_VECTORS, map_location="cpu", weights_only=False)["v"]
    return model, tok, dev, layers, pad_id, sae, sae_id, v_full, sae_dev


def load_bundle():
    from app.persona.judge_vertex import judge_rubric_to_instructions
    from app.persona.schemas import PersonaTraitArtifact

    bundle = PersonaTraitArtifact.model_validate_json(
        Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    return bundle, judge_instr


def encode_ids(tok, neg_sys: str, prompt: str, dev):
    msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    ids = ids.to(dev)
    attn = torch.ones_like(ids, dtype=torch.long, device=dev)
    return ids, attn


# --------------------------------------------------------------------------- #
# Phase A: GradSAE-style output-side attribution
# --------------------------------------------------------------------------- #
def _enable_grad_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on model")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not enable gradient checkpointing: %s", e)


def _freeze_params(module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def phase_a_attribution(
    *, model, tok, dev, layers, sae, sae_dev, v_full, eval_qs, neg_sys, readout_layers, out_path,
):
    """Rank every SAE feature by output-side gradient attribution.

    Forward: at L16, substitute h := decode(encode(h)) so downstream depends on z.
    Objective: mean over response positions of sum_{l in readout_layers}
               cos-projection of h_l onto the persona direction v_full[l].
    attribution_i = mean_t( z_i * dL/dz_i ).
    """
    _freeze_params(model)
    _freeze_params(sae)
    _enable_grad_checkpointing(model)

    d_sae = int(sae.cfg.d_sae)
    n_layers = len(layers)
    readout_layers = [l for l in readout_layers if 0 <= l < n_layers]
    logger.info("Phase A: readout layers = %s, d_sae=%d", readout_layers, d_sae)

    # Persona readout directions (unit) per readout layer.
    readout_dirs = {}
    for l in readout_layers:
        d = v_full[l].float().to(dev)
        readout_dirs[l] = d / (d.norm() + 1e-8)

    attr_sum = torch.zeros(d_sae, dtype=torch.float64, device="cpu")
    tok_count = 0

    for qi, prompt in enumerate(eval_qs):
        ids, attn = encode_ids(tok, neg_sys, prompt, dev)
        seq_len = ids.shape[-1]

        # One readout layer per backward pass to limit activation memory.
        for rl in readout_layers:
            z_holder: dict = {}
            readout_holder: dict = {}

            def l16_hook(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                h_sae = h.to(sae_dev)
                z = sae.encode(h_sae)
                z.retain_grad()
                z_holder["z"] = z
                h_recon = sae.decode(z).to(device=h.device, dtype=h.dtype)
                if isinstance(output, tuple):
                    return (h_recon,) + tuple(output[1:])
                return h_recon

            def readout_hook(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                readout_holder["h"] = h
                return output

            handles = [
                layers[LAYER_IDX].register_forward_hook(l16_hook),
                layers[rl].register_forward_hook(readout_hook),
            ]

            model.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            model(input_ids=ids, attention_mask=attn, use_cache=False)

            h = readout_holder["h"].float()
            proj = (h * readout_dirs[rl].view(1, 1, -1)).sum(-1)
            loss = proj.mean()
            loss.backward()

            z = z_holder["z"]
            g = z.grad
            contrib = (z.detach().float() * g.float())[0].sum(0).cpu().to(torch.float64)
            attr_sum += contrib
            tok_count += seq_len

            for hh in handles:
                hh.remove()
            del z, g, loss, h, readout_holder, z_holder
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        logger.info("  [A] Q%d done (T=%d)", qi + 1, seq_len)

    attribution = attr_sum / max(tok_count, 1)

    # Decoder cosine to v_dense@L16 for comparison (input-side baseline).
    from app.persona.sae_common import _get_decoder_columns
    W_dec = _get_decoder_columns(sae)  # (d_sae, d_in) cpu float
    v16 = v_full[LAYER_IDX].float().cpu()
    v_unit = v16 / (v16.norm() + 1e-8)
    dec_norms = W_dec.norm(dim=1) + 1e-8
    dec_cos = (W_dec @ v_unit) / dec_norms  # (d_sae,)

    order = torch.argsort(attribution, descending=True)
    top_pos = order[:200].tolist()
    top_neg = order[-200:].tolist()

    def pack(fids):
        return [
            {
                "feature_id": int(f),
                "attribution": round(float(attribution[f]), 6),
                "dec_cos": round(float(dec_cos[f]), 4),
                "dec_norm": round(float(dec_norms[f] - 1e-8), 4),
            }
            for f in fids
        ]

    result = {
        "phase": "A",
        "method": "gradsae_output_attribution",
        "readout_layers": readout_layers,
        "n_features": d_sae,
        "n_questions": len(eval_qs),
        "top_positive": pack(top_pos),
        "top_negative": pack(top_neg),
        # spearman-ish diagnostic: correlation of attribution rank vs dec_cos rank
        "attr_vs_deccos_pearson": round(
            float(torch.corrcoef(torch.stack([attribution, dec_cos]))[0, 1].item()), 4
        ),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Phase A wrote %s", out_path)
    logger.info("  attribution vs dec_cos pearson = %s", result["attr_vs_deccos_pearson"])
    logger.info("  top-10 positive features: %s",
                [(d["feature_id"], d["attribution"], d["dec_cos"]) for d in result["top_positive"][:10]])
    return result


# --------------------------------------------------------------------------- #
# Steering + judging helpers (Phase B/C)
# --------------------------------------------------------------------------- #
def make_gen_fn(model, tok, layers, ids, attn, pad_id, max_new_tokens=200):
    def gen_text(hook_fn=None) -> str:
        handle = None
        if hook_fn is not None:
            handle = layers[LAYER_IDX].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_id, use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return gen_text


def judge(score_transcript, judge_instr, neg_sys, prompt, reply):
    if len(reply.strip()) < 20:
        return None
    try:
        js = score_transcript(judge_instr, neg_sys, prompt, reply)
        return int(js.score)
    except Exception as e:  # noqa: BLE001
        logger.warning("Judge failed: %s", e)
        return None


def mean_score(scores):
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


# --------------------------------------------------------------------------- #
# Phase B: single-feature causal validation
# --------------------------------------------------------------------------- #
def phase_b_single_feature(
    *, model, tok, dev, layers, pad_id, sae, v_full, eval_qs, neg_sys,
    judge_instr, candidate_fids, alpha_dense, n_questions, out_path,
):
    from app.persona.steering_demo import _steering_hook_fn
    from app.persona.judge_vertex import score_transcript
    from app.persona.sae_common import _get_decoder_columns

    dtype = next(model.parameters()).dtype
    W_dec = _get_decoder_columns(sae)  # cpu float (d_sae, d_in)
    v16 = v_full[LAYER_IDX].float()
    dense_norm = float((alpha_dense * v16).norm().item())  # match injected magnitude
    direction_dense = v16.to(device=dev, dtype=dtype).view(1, 1, -1)

    qs = eval_qs[:n_questions]
    prepared = []
    for prompt in qs:
        ids, attn = encode_ids(tok, neg_sys, prompt, dev)
        prepared.append((prompt, ids, attn))

    # Reference: baseline + dense per question
    refs = {"BASELINE": [], "DENSE_CAA": []}
    for prompt, ids, attn in prepared:
        gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id)
        base = gen_text()
        refs["BASELINE"].append(judge(score_transcript, judge_instr, neg_sys, prompt, base))
        dense = gen_text(_steering_hook_fn(alpha_dense, direction_dense,
                                           steer_last_token_only=False, hook_calls=[0]))
        refs["DENSE_CAA"].append(judge(score_transcript, judge_instr, neg_sys, prompt, dense))
    logger.info("[B] BASELINE mean=%s  DENSE_CAA mean=%s",
                mean_score(refs["BASELINE"]), mean_score(refs["DENSE_CAA"]))

    feature_rows = []
    for fi, fid in enumerate(candidate_fids):
        col = W_dec[fid].float()
        col_norm = float(col.norm().item())
        if col_norm < 1e-8:
            continue
        unit = (col / col_norm).to(device=dev, dtype=dtype).view(1, 1, -1)
        # alpha so that injected norm == dense_norm
        alpha_eff = dense_norm
        per_q = []
        for prompt, ids, attn in prepared:
            gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id)
            reply = gen_text(_steering_hook_fn(alpha_eff, unit,
                                               steer_last_token_only=False, hook_calls=[0]))
            per_q.append({
                "prompt_idx": qs.index(prompt),
                "score": judge(score_transcript, judge_instr, neg_sys, prompt, reply),
                "reply": reply[:300],
            })
        row = {
            "feature_id": int(fid),
            "dec_norm": round(col_norm, 4),
            "causal_score": mean_score([p["score"] for p in per_q]),
            "scores": [p["score"] for p in per_q],
            "samples": per_q,
        }
        feature_rows.append(row)
        logger.info("[B] %d/%d feat %d causal_score=%s scores=%s",
                    fi + 1, len(candidate_fids), fid, row["causal_score"], row["scores"])

    feature_rows.sort(key=lambda r: (r["causal_score"] is not None, r["causal_score"] or -1),
                      reverse=True)
    result = {
        "phase": "B",
        "alpha_dense": alpha_dense,
        "dense_inject_norm": round(dense_norm, 3),
        "n_questions": len(qs),
        "reference": {k: {"mean": mean_score(v), "scores": v} for k, v in refs.items()},
        "features": feature_rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Phase B wrote %s", out_path)
    logger.info("[B] top causal features: %s",
                [(r["feature_id"], r["causal_score"]) for r in feature_rows[:10]])
    return result


# --------------------------------------------------------------------------- #
# Phase C: greedy sparse build
# --------------------------------------------------------------------------- #
def phase_c_greedy(
    *, model, tok, dev, layers, pad_id, sae, v_full, eval_qs, neg_sys,
    judge_instr, ranked_fids, ranked_scores, alpha_dense, n_questions,
    max_features, out_path,
):
    from app.persona.steering_demo import _steering_hook_fn
    from app.persona.judge_vertex import score_transcript
    from app.persona.sae_common import _get_decoder_columns

    dtype = next(model.parameters()).dtype
    W_dec = _get_decoder_columns(sae)  # cpu (d_sae, d_in)
    v16 = v_full[LAYER_IDX].float()
    dense_norm = float((alpha_dense * v16).norm().item())

    qs = eval_qs[:n_questions]
    prepared = []
    for prompt in qs:
        ids, attn = encode_ids(tok, neg_sys, prompt, dev)
        prepared.append((prompt, ids, attn))

    # weight each feature by its causal score (clip negatives to 0)
    weights = [max(s or 0.0, 0.0) for s in ranked_scores]
    fids = list(ranked_fids)[:max_features]
    weights = weights[:max_features]

    steps = []
    cum_vec = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for k in range(1, len(fids) + 1):
        fid = fids[k - 1]
        cum_vec = cum_vec + weights[k - 1] * W_dec[fid].float()
        cn = float(cum_vec.norm().item())
        if cn < 1e-8:
            continue
        unit = (cum_vec / cn).to(device=dev, dtype=dtype).view(1, 1, -1)
        per_q = []
        for prompt, ids, attn in prepared:
            gen_text = make_gen_fn(model, tok, layers, ids, attn, pad_id)
            reply = gen_text(_steering_hook_fn(dense_norm, unit,
                                               steer_last_token_only=False, hook_calls=[0]))
            per_q.append(judge(score_transcript, judge_instr, neg_sys, prompt, reply))
        ms = mean_score(per_q)
        steps.append({"n_features": k, "added_feature": int(fid),
                      "cum_mean_score": ms, "scores": per_q})
        logger.info("[C] k=%d (+feat %d) cum_mean=%s", k, fid, ms)

    wsum = sum(weights) or 1.0
    decomposition = [
        {"feature_id": int(fid), "causal_score": ranked_scores[i],
         "weight": round(weights[i] / wsum, 4)}
        for i, fid in enumerate(fids)
    ]
    result = {
        "phase": "C",
        "n_questions": len(qs),
        "greedy_curve": steps,
        "decomposition": decomposition,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Phase C wrote %s", out_path)
    return result


# --------------------------------------------------------------------------- #
# STA overlap
# --------------------------------------------------------------------------- #
def sta_overlap(causal_fids, sta_path, out_path):
    p = Path(sta_path)
    if not p.exists():
        logger.warning("STA file %s not found; skipping overlap", sta_path)
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    sta_fids = set()
    # tolerate a few shapes
    def collect(obj):
        if isinstance(obj, dict):
            if "feature_id" in obj:
                sta_fids.add(int(obj["feature_id"]))
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
    collect(data)
    causal = set(int(f) for f in causal_fids)
    inter = causal & sta_fids
    union = causal | sta_fids
    res = {
        "n_causal": len(causal),
        "n_sta": len(sta_fids),
        "intersection": sorted(inter),
        "n_intersection": len(inter),
        "jaccard": round(len(inter) / max(len(union), 1), 4),
    }
    Path(out_path).write_text(json.dumps(res, indent=2), encoding="utf-8")
    logger.info("STA overlap: jaccard=%s intersection=%d/%d causal",
                res["jaccard"], len(inter), len(causal))
    return res


# --------------------------------------------------------------------------- #
# Neuronpedia interpretation (262k L16: 16-gemmascope-2-res-262k)
# --------------------------------------------------------------------------- #
def interpret_decomposition(*, c_path: Path, sae_id: str, out_path: Path, top_k: int = 15):
    from app.persona.sae_autointerp import (
        DEFAULT_NEURONPEDIA_MODEL,
        explanation_from_neuronpedia,
        fetch_neuronpedia_feature,
        neuronpedia_feature_url,
        neuronpedia_source_set,
    )

    c_res = json.loads(c_path.read_text(encoding="utf-8"))
    source = neuronpedia_source_set(SAE_RELEASE, sae_id)
    curve = c_res.get("greedy_curve", [])
    best_k = max(curve, key=lambda s: s.get("cum_mean_score") or -1) if curve else None
    decomp = c_res.get("decomposition", [])[:top_k]

    labeled = []
    for row in decomp:
        fid = int(row["feature_id"])
        expl = None
        url = neuronpedia_feature_url(DEFAULT_NEURONPEDIA_MODEL, source, fid) if source else None
        if source:
            doc = fetch_neuronpedia_feature(DEFAULT_NEURONPEDIA_MODEL, source, fid)
            if doc:
                expl = explanation_from_neuronpedia(doc)
        labeled.append({
            **row,
            "neuronpedia_source": source,
            "neuronpedia_url": url,
            "explanation": expl or "(no Neuronpedia entry)",
        })

    report = {
        "sae_id": sae_id,
        "neuronpedia_source": source,
        "best_greedy_step": best_k,
        "decomposition_labeled": labeled,
        "formula": " + ".join(
            f"feat_{r['feature_id']}({r['weight']})" for r in labeled[:8]
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Interpretation report: %s", out_path)
    return report


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--phase", choices=["A", "B", "C", "all", "interpret"], default="A")
    ap.add_argument("--sae", choices=["16k", "262k"], default="262k",
                    help="SAE width (default 262k)")
    ap.add_argument("--n-questions", type=int, default=3, help="questions for B/C judging")
    ap.add_argument("--n-questions-a", type=int, default=5, help="contrast prompts for attribution")
    ap.add_argument("--top-n", type=int, default=60, help="features to validate in phase B")
    ap.add_argument("--max-features", type=int, default=25, help="greedy cap in phase C")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--readout-layers", default="26", help="comma-separated readout layers (default: 26 only for GPU memory)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    global LAYER_IDX, SAE_HS_INDEX, SAE_CONFIGS, RUN_VECTORS
    cfg = resolve_trait(args.trait)
    LAYER_IDX = int(args.layer if args.layer is not None else cfg["layer"])
    check_override(cfg, cli_layer=args.layer, cli_alpha=args.alpha)
    SAE_HS_INDEX = hidden_state_index(LAYER_IDX)
    SAE_CONFIGS = {
        "262k": sae_id_for_layer(LAYER_IDX, "262k"),
        "16k": sae_id_for_layer(LAYER_IDX, "16k"),
    }
    RUN_VECTORS = Path(cfg["vectors"])
    alpha = float(args.alpha if args.alpha is not None else cfg["alpha"])
    outdir = Path(args.outdir or cfg["sae_dir"])
    tag = f"{args.sae}_l{LAYER_IDX}"
    a_path = outdir / f"causal_screen_A_{tag}.json"
    b_path = outdir / f"causal_screen_B_{tag}.json"
    c_path = outdir / f"causal_screen_C_{tag}.json"
    overlap_path = outdir / f"causal_screen_overlap_{tag}.json"
    report_path = outdir / f"causal_screen_report_{tag}.json"

    bundle, judge_instr = load_bundle()
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions

    if args.phase == "interpret":
        interpret_decomposition(c_path=c_path, sae_id=SAE_CONFIGS[args.sae], out_path=report_path)
        print("=== DONE ===")
        return

    model, tok, dev, layers, pad_id, sae, sae_id, v_full, sae_dev = load_everything(args.sae)

    readout_layers = [int(x) for x in args.readout_layers.split(",") if x.strip()]

    if args.phase in ("A", "all"):
        phase_a_attribution(
            model=model, tok=tok, dev=dev, layers=layers, sae=sae, sae_dev=sae_dev,
            v_full=v_full, eval_qs=eval_qs[: args.n_questions_a], neg_sys=neg_sys,
            readout_layers=readout_layers, out_path=a_path,
        )

    if args.phase in ("B", "all"):
        a_res = json.loads(a_path.read_text(encoding="utf-8"))
        candidate_fids = [d["feature_id"] for d in a_res["top_positive"][: args.top_n]]
        phase_b_single_feature(
            model=model, tok=tok, dev=dev, layers=layers, pad_id=pad_id, sae=sae,
            v_full=v_full, eval_qs=eval_qs, neg_sys=neg_sys, judge_instr=judge_instr,
            candidate_fids=candidate_fids, alpha_dense=alpha,
            n_questions=args.n_questions, out_path=b_path,
        )

    if args.phase in ("C", "all"):
        b_res = json.loads(b_path.read_text(encoding="utf-8"))
        ranked = [r for r in b_res["features"] if r["causal_score"] is not None]
        ranked_fids = [r["feature_id"] for r in ranked]
        ranked_scores = [r["causal_score"] for r in ranked]
        phase_c_greedy(
            model=model, tok=tok, dev=dev, layers=layers, pad_id=pad_id, sae=sae,
            v_full=v_full, eval_qs=eval_qs, neg_sys=neg_sys, judge_instr=judge_instr,
            ranked_fids=ranked_fids, ranked_scores=ranked_scores, alpha_dense=alpha,
            n_questions=args.n_questions, max_features=args.max_features, out_path=c_path,
        )
        sta_overlap(
            ranked_fids[: args.max_features],
            outdir / f"sta_validation_l{LAYER_IDX}_v2.json",
            overlap_path,
        )
        interpret_decomposition(c_path=c_path, sae_id=sae_id, out_path=report_path)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
