#!/usr/bin/env python3
"""
Cluster SSV features by decoder similarity, label clusters from corpus
interpretations, and run per-cluster causal steering validation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SSV = REPO / "persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json"
DEFAULT_INTERP = REPO / "persona_runs/dnd_good_scale/sae/ssv_corpus_interp.json"
DEFAULT_OUT = REPO / "persona_runs/dnd_good_scale/sae/ssv_cluster_report.json"


def load_feature_set(ssv_path: Path, k: int) -> tuple[list[int], list[float]]:
    doc = json.loads(ssv_path.read_text(encoding="utf-8"))
    for row in doc.get("results", []):
        if row.get("k") == k:
            fids = [int(f) for f in row.get("feature_ids", [])]
            wts = row.get("feature_weights") or [1.0] * len(fids)
            return fids, [float(w) for w in wts]
    raise KeyError(f"K={k} not found in {ssv_path}")


def hierarchical_cluster(W: np.ndarray, n_clusters: int) -> np.ndarray:
    """Simple agglomerative clustering on cosine distance."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if W.shape[0] <= n_clusters:
        return np.arange(W.shape[0])
    dist = pdist(W, metric="cosine")
    Z = linkage(dist, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels - 1  # 0-indexed


def cluster_label(interps: dict[str, dict], fids: list[int]) -> str:
    """Most common interpretation theme in cluster (first 3 words)."""
    themes: dict[str, int] = {}
    for fid in fids:
        entry = interps.get(str(fid), {})
        text = entry.get("interpretation", "unknown")
        key = " ".join(text.split()[:3]).lower()
        themes[key] = themes.get(key, 0) + 1
    if not themes:
        return "unlabeled"
    return max(themes, key=themes.get)


def build_cluster_vector(
    W_dec: torch.Tensor,
    fids: list[int],
    weights: list[float],
    steer_norm: float,
) -> torch.Tensor:
    cols = W_dec[fids].float()
    w = torch.tensor(weights, dtype=torch.float32)
    v = (cols.T @ w).float()
    n = float(v.norm())
    if n > 1e-8:
        v = v * (steer_norm / n)
    return v


def run_cluster_steering(
    model,
    tok,
    dev,
    layer: int,
    neg_sys: str,
    judge_instr: str,
    eval_qs: list[str],
    v: torch.Tensor,
    alpha: float,
    label: str,
    max_new_tokens: int = 200,
) -> dict:
    layers = _language_model_layers(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    dtype = next(model.parameters()).dtype
    d = v.to(device=dev, dtype=dtype).view(1, 1, -1)
    scores = []

    for qi, prompt in enumerate(eval_qs):
        msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
        attn = torch.ones_like(ids)
        hook = _steering_hook_fn(alpha, d, steer_last_token_only=False, hook_calls=[0])
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id, use_cache=True,
            )
        handle.remove()
        reply = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        s = None
        if len(reply.strip()) >= 20:
            try:
                s = int(score_transcript(judge_instr, neg_sys, prompt, reply).score)
            except Exception as exc:
                logger.warning("Judge failed Q%d: %s", qi, exc)
        scores.append(s)
        logger.info("  [%s] Q%d score=%s", label, qi + 1, s)

    valid = [s for s in scores if s is not None]
    mean = round(sum(valid) / len(valid), 1) if valid else None
    return {"label": label, "mean": mean, "scores": scores}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", default="good")
    ap.add_argument("--ssv", type=Path, default=DEFAULT_SSV)
    ap.add_argument("--interp", type=Path, default=DEFAULT_INTERP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=100, help="SSV K level for feature set")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--skip-causal", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = cfg["layer"]
    sae_id = cfg["sae_id"]
    bundle = PersonaTraitArtifact.model_validate_json(cfg["bundle"].read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    fids, weights = load_feature_set(args.ssv, args.k)
    logger.info("Clustering %d features from SSV K=%d", len(fids), args.k)

    interps = {}
    if args.interp.is_file():
        interps = json.loads(args.interp.read_text(encoding="utf-8")).get("features", {})

    sae, _ = load_sae_for_layer(
        torch.device("cpu"), release=SAE_RELEASE, sae_id=sae_id, hidden_state_index=cfg["hs_index"],
    )
    W_dec = sae.W_dec.detach().float().cpu()
    W_norm = F.normalize(W_dec[fids], dim=1).numpy()

    n_clusters = min(args.n_clusters, len(fids))
    labels = hierarchical_cluster(W_norm, n_clusters)

    clusters: dict[int, list[int]] = {}
    for i, fid in enumerate(fids):
        clusters.setdefault(int(labels[i]), []).append(fid)

    cluster_info = []
    for cid, cfids in sorted(clusters.items()):
        idxs = [fids.index(f) for f in cfids]
        cw = [weights[i] for i in idxs]
        label = cluster_label(interps, cfids)
        cluster_info.append({
            "cluster_id": cid,
            "label": label,
            "feature_ids": cfids,
            "weights": [round(w, 6) for w in cw],
            "n_features": len(cfids),
        })
        logger.info("Cluster %d (%s): %d features %s", cid, label, len(cfids), cfids[:5])

    causal_results = []
    forward_rows: list[dict] = []
    if not args.skip_causal:
        vectors_path = cfg["vectors"]
        steer_norm = 1.0
        if vectors_path.exists():
            v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
            steer_norm = float(v_full[layer].norm())

        model, tok, dev = load_model_and_tokenizer()

        # Full K=100 vector
        full_v = build_cluster_vector(W_dec, fids, weights, steer_norm)
        causal_results.append(run_cluster_steering(
            model, tok, dev, layer, neg_sys, judge_instr, eval_qs,
            full_v, args.alpha, f"SSV_K{args.k}_FULL", args.max_new_tokens,
        ))

        # Per-cluster sufficiency
        for c in cluster_info:
            v_c = build_cluster_vector(W_dec, c["feature_ids"], c["weights"], steer_norm)
            row = run_cluster_steering(
                model, tok, dev, layer, neg_sys, judge_instr, eval_qs,
                v_c, args.alpha, f"CLUSTER_{c['cluster_id']}", args.max_new_tokens,
            )
            row["cluster_id"] = c["cluster_id"]
            row["cluster_label"] = c["label"]
            row["n_features"] = c["n_features"]
            causal_results.append(row)

        # Greedy forward cluster selection
        selected: list[int] = []
        remaining = list(range(len(cluster_info)))
        forward_rows = []
        best_mean = 0.0
        for step in range(len(cluster_info)):
            best_cid = None
            best_step_mean = -1.0
            for cid in remaining:
                trial_fids = []
                trial_w = []
                for sel in selected + [cid]:
                    c = cluster_info[sel]
                    trial_fids.extend(c["feature_ids"])
                    trial_w.extend(c["weights"])
                v_t = build_cluster_vector(W_dec, trial_fids, trial_w, steer_norm)
                row = run_cluster_steering(
                    model, tok, dev, layer, neg_sys, judge_instr, eval_qs,
                    v_t, args.alpha, f"FORWARD_{step+1}", args.max_new_tokens,
                )
                if row["mean"] is not None and row["mean"] > best_step_mean:
                    best_step_mean = row["mean"]
                    best_cid = cid
            if best_cid is None:
                break
            selected.append(best_cid)
            remaining.remove(best_cid)
            forward_rows.append({
                "step": step + 1,
                "added_cluster": best_cid,
                "cluster_label": cluster_info[best_cid]["label"],
                "mean_trait": best_step_mean,
                "n_clusters": len(selected),
            })
            logger.info("Forward step %d: cluster %d -> mean=%.1f", step + 1, best_cid, best_step_mean)

    report = {
        "meta": {
            "trait": args.trait,
            "layer": layer,
            "sae_id": sae_id,
            "ssv_k": args.k,
            "n_clusters": n_clusters,
            "n_features": len(fids),
        },
        "clusters": cluster_info,
        "causal": causal_results,
        "forward_selection": forward_rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
