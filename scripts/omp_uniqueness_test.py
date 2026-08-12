#!/usr/bin/env python3
"""
Phase 0: Prove OMP decomposition non-uniqueness and random-subspace baselines.

  0a) Re-run OMP on v_dense with original top-K features banned -> disjoint set B.
  0b) Least-squares fit v_dense onto random feature subsets; steer + judge.

Compares steering efficacy when cos(v, v_dense) is matched but feature identity differs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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

ALPHA = 1.5
LAYER = 16
SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
SAE_ID = "layer_16_width_262k_l0_small"
SAE_HS_INDEX = 17


def omp_decompose(
    W: torch.Tensor,
    target: torch.Tensor,
    k: int,
    *,
    banned: set[int] | None = None,
) -> tuple[list[int], list[float], torch.Tensor]:
    """Greedy OMP; optionally exclude feature ids."""
    banned = banned or set()
    W_unit = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    residual = target.clone()
    selected: list[int] = []
    coefs: list[float] = []

    for _ in range(k):
        dots = W_unit @ residual
        if banned:
            mask = torch.zeros(dots.shape[0], dtype=torch.bool)
            for b in banned:
                if 0 <= b < dots.numel():
                    mask[b] = True
            dots = dots.masked_fill(mask, 0.0)
        best = int(dots.abs().argmax().item())
        if dots[best].abs().item() < 1e-12:
            break
        c = float((W[best] @ residual) / (W[best] @ W[best]))
        selected.append(best)
        coefs.append(c)
        residual = residual - c * W[best]

    if not selected:
        recon = torch.zeros_like(target)
    else:
        recon = sum(coefs[i] * W[selected[i]] for i in range(len(selected)))
    return selected, coefs, recon


def lsq_fit(W_sub: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve min ||W_sub^T c - target|| for c. W_sub: (k, d_in)."""
    # torch.linalg.lstsq expects A @ x = b with A (m,n), b (m,)
    A = W_sub.T  # (d_in, k)
    sol = torch.linalg.lstsq(A, target.unsqueeze(-1)).solution.squeeze(-1)
    recon = W_sub.T @ sol
    return sol, recon


def vec_metrics(target: torch.Tensor, recon: torch.Tensor) -> dict:
    cos = float(
        torch.nn.functional.cosine_similarity(
            target.unsqueeze(0), recon.unsqueeze(0)
        ).item()
    )
    return {
        "cosine": round(cos, 4),
        "norm_ratio": round(float(recon.norm() / target.norm()), 4),
        "recon_norm": round(float(recon.norm()), 2),
        "target_norm": round(float(target.norm()), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--decomp",
        default="persona_runs/dnd_good_scale/sae/omp_decomposition_262k_l16.json",
    )
    ap.add_argument(
        "--out",
        default="persona_runs/dnd_good_scale/sae/omp_uniqueness_262k_l16.json",
    )
    ap.add_argument("--k", type=int, default=750, help="features for banned OMP rerun")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--random-k", type=int, default=750)
    ap.add_argument("--match-cos", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--conditions",
        default="all",
        help="comma list: ref,omp_a,omp_b,random_lsq,random_match or 'all'",
    )
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="Re-judge replies already saved in --out JSON (no model load)",
    )
    args = ap.parse_args()

    if args.rescore:
        out = Path(args.out)
        result = json.loads(out.read_text())
        bundle = PersonaTraitArtifact.model_validate_json(
            Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text()
        )
        judge_instr = judge_rubric_to_instructions(
            bundle.judge_rubric, trait_label=bundle.trait_label
        )
        neg_sys = bundle.neg_system_prompt
        eval_qs = bundle.eval_questions[: args.n_questions]

        from app.persona.judge_vertex import score_transcript

        for block in result.get("steering", []):
            scores = []
            for sample in block.get("samples", []):
                qi = sample["q_idx"]
                prompt = eval_qs[qi]
                reply = sample.get("reply", "")
                try:
                    if len(reply.strip()) < 20:
                        s = None
                    else:
                        s = int(
                            score_transcript(
                                judge_instr, neg_sys, prompt, reply
                            ).score
                        )
                except Exception as exc:
                    print(f"  judge error [{block['label']}] Q{qi+1}: {exc}")
                    s = None
                sample["score"] = s
                scores.append(s)
                print(f"  [{block['label']}] Q{qi+1} score={s}")
            valid = [s for s in scores if s is not None]
            block["scores"] = scores
            block["mean"] = round(sum(valid) / len(valid), 1) if valid else None
            print(f"  [{block['label']}] MEAN={block['mean']}")
        out.write_text(json.dumps(result, indent=2))
        print(f"Rescored {out}")
        return

    rng = random.Random(args.seed)
    conds = (
        ["ref", "omp_a", "omp_b", "random_lsq", "random_match"]
        if args.conditions == "all"
        else [c.strip() for c in args.conditions.split(",") if c.strip()]
    )

    bundle = PersonaTraitArtifact.model_validate_json(
        Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text()
    )
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    v_full = torch.load(
        "persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )["v"]
    v16 = v_full[LAYER].float()

    decomp_path = Path(args.decomp)
    if not decomp_path.exists():
        raise FileNotFoundError(f"OMP decomposition not found: {decomp_path}")
    omp_data = json.loads(decomp_path.read_text())
    omp_a_fids = [int(e["feature_id"]) for e in omp_data["decomposition"][: args.k]]
    omp_a_coefs = [float(e["coefficient"]) for e in omp_data["decomposition"][: args.k]]

    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        hidden_state_index=SAE_HS_INDEX,
    )
    W = sae.W_dec.detach().float()
    d_sae = W.shape[0]

    v_omp_a = sum(omp_a_coefs[i] * W[omp_a_fids[i]] for i in range(len(omp_a_fids)))
    metrics_a = vec_metrics(v16, v_omp_a)

    banned = set(omp_a_fids)
    omp_b_fids, omp_b_coefs, v_omp_b = omp_decompose(W, v16, args.k, banned=banned)
    overlap = len(set(omp_b_fids) & banned)
    metrics_b = vec_metrics(v16, v_omp_b)

    print(f"OMP_A: K={len(omp_a_fids)} cos={metrics_a['cosine']} norm_ratio={metrics_a['norm_ratio']}")
    print(
        f"OMP_B: K={len(omp_b_fids)} cos={metrics_b['cosine']} "
        f"overlap_with_A={overlap} norm_ratio={metrics_b['norm_ratio']}"
    )

    # Random 750 LSQ
    random_k = min(args.random_k, d_sae)
    rand_ids = rng.sample(range(d_sae), random_k)
    W_rand = W[rand_ids]
    _, v_rand_lsq = lsq_fit(W_rand, v16)
    metrics_rand = vec_metrics(v16, v_rand_lsq)
    print(f"RANDOM_LSQ: K={random_k} cos={metrics_rand['cosine']}")

    # Random features until cos >= match_cos
    pool = list(range(d_sae))
    rng.shuffle(pool)
    match_ids: list[int] = []
    v_match = torch.zeros_like(v16)
    match_cos = 0.0
    for fid in pool:
        if fid in banned:
            continue
        match_ids.append(fid)
        W_sub = W[match_ids]
        _, v_match = lsq_fit(W_sub, v16)
        match_cos = float(
            torch.nn.functional.cosine_similarity(
                v16.unsqueeze(0), v_match.unsqueeze(0)
            ).item()
        )
        if match_cos >= args.match_cos:
            break
    metrics_match = vec_metrics(v16, v_match)
    print(
        f"RANDOM_MATCH: K={len(match_ids)} cos={metrics_match['cosine']} "
        f"(target>={args.match_cos})"
    )

    vectors: dict[str, torch.Tensor] = {
        "DENSE_CAA": v16,
        "OMP_A": v_omp_a,
        "OMP_B": v_omp_b,
        "RANDOM_LSQ": v_rand_lsq,
        "RANDOM_MATCH": v_match,
    }
    cond_map = {
        "ref": ["BASELINE", "DENSE_CAA"],
        "omp_a": ["OMP_A"],
        "omp_b": ["OMP_B"],
        "random_lsq": ["RANDOM_LSQ"],
        "random_match": ["RANDOM_MATCH"],
    }
    to_run: list[str] = []
    for c in conds:
        to_run.extend(cond_map.get(c, [c.upper()]))

    result: dict = {
        "method": "omp_uniqueness_test",
        "layer": LAYER,
        "alpha": ALPHA,
        "k_banned_omp": args.k,
        "seed": args.seed,
        "geometry": {
            "OMP_A": {
                "n_features": len(omp_a_fids),
                "feature_ids_sample": omp_a_fids[:20],
                **metrics_a,
            },
            "OMP_B": {
                "n_features": len(omp_b_fids),
                "overlap_with_A": overlap,
                "jaccard_disjoint": round(
                    1.0 - overlap / max(len(set(omp_b_fids) | banned), 1), 4
                ),
                "feature_ids_sample": omp_b_fids[:20],
                **metrics_b,
            },
            "RANDOM_LSQ": {
                "n_features": random_k,
                "feature_ids_sample": rand_ids[:20],
                **metrics_rand,
            },
            "RANDOM_MATCH": {
                "n_features": len(match_ids),
                "target_cos": args.match_cos,
                "feature_ids_sample": match_ids[:20],
                **metrics_match,
            },
        },
        "steering": [],
    }

    if args.skip_judge:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"Saved geometry-only {out}")
        return

    model, tok, dev = load_model_and_tokenizer()
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype

    def gen_text(ids, attn, hook_fn=None) -> str:
        handle = None
        if hook_fn is not None:
            handle = layers[LAYER].register_forward_hook(hook_fn)
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

    def judge_reply(prompt: str, reply: str):
        if len(reply.strip()) < 20:
            return None
        try:
            js = score_transcript(judge_instr, neg_sys, prompt, reply)
            return int(js.score)
        except Exception as exc:
            print(f"  judge error: {exc}")
            return None

    def run_label(label: str, direction: torch.Tensor | None = None):
        scores = []
        samples = []
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
            hook = None
            if direction is not None:
                d = direction.to(device=dev, dtype=dtype).view(1, 1, -1)
                hook = _steering_hook_fn(
                    ALPHA, d, steer_last_token_only=False, hook_calls=[0]
                )
            reply = gen_text(ids, attn, hook)
            s = judge_reply(prompt, reply)
            scores.append(s)
            samples.append({"q_idx": qi, "score": s, "reply": reply[:300]})
            print(f"  [{label}] Q{qi+1} score={s}")
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean}")
        return {
            "label": label,
            "mean": mean,
            "scores": scores,
            "samples": samples,
        }

    if "BASELINE" in to_run:
        print("=== BASELINE ===")
        result["steering"].append(run_label("BASELINE", None))
    if "DENSE_CAA" in to_run:
        print("=== DENSE_CAA ===")
        result["steering"].append(run_label("DENSE_CAA", vectors["DENSE_CAA"]))

    for label in ["OMP_A", "OMP_B", "RANDOM_LSQ", "RANDOM_MATCH"]:
        if label in to_run:
            print(f"=== {label} ===")
            result["steering"].append(run_label(label, vectors[label]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\n=== SUMMARY ===")
    for row in result["steering"]:
        print(f"  {row['label']:>14s}  mean={row['mean']}  scores={row['scores']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
