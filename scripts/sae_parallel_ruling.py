#!/usr/bin/env python3
"""
Parallel ruling experiments while Option C runs (prep + analysis + inject variants).

Tasks:
  prep-clamp     — p95 clamp targets from latents (CPU)
  encode-vdense  — encode dense persona vector; compare to STA atoms (CPU SAE)
  output-score   — rank features by decoder·W_U alignment (CPU SAE + model embed)
  aligned-inject — sum-and-inject aligned vs misaligned vs full (GPU)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

PERSONA_RUNS = Path(os.environ.get("PERSONA_RUNS", Path.home() / "gemma-chat/persona_runs"))
DEFAULT_RUN = "dnd_good_scale"
DEFAULT_LAYER = 16
DEFAULT_SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
DEFAULT_SAE_ID = "layer_16_width_16k_l0_small"
DEFAULT_LATENTS = "sae/sae_latents_l16_v2.pt"
DEFAULT_STEERED_KEY = "1.5"


def _run_dir(run_id: str) -> Path:
    return PERSONA_RUNS / run_id


def _load_sta_atoms(run_dir: Path) -> list[dict[str, Any]]:
    sta_path = run_dir / "sae/sta_validation_l16_exp1a.json"
    if not sta_path.is_file():
        sta_path = run_dir / "sae/sta_validation_l16_v2.json"
    sta = json.loads(sta_path.read_text(encoding="utf-8"))
    post_ids = set(sta.get("sta_positive_atom_ids") or [])
    atoms = sta["sta_attribution"]["sta_positive_atoms"]
    if post_ids:
        atoms = [a for a in atoms if a["feature_id"] in post_ids]
    with_cos = {a["feature_id"]: a.get("decoder_cosine") for a in sta.get("sta_positive_atoms_with_cosine") or []}
    for a in atoms:
        if a["feature_id"] in with_cos and with_cos[a["feature_id"]] is not None:
            a["decoder_cosine"] = with_cos[a["feature_id"]]
    return atoms


def task_prep_clamp(
    run_dir: Path,
    out_json: Path,
    *,
    latents_pt: Path,
    steered_alpha_key: str,
    layer_idx: int,
) -> Path:
    """Task 1: p95 per-feature activation on steered spans."""
    ckpt = torch.load(latents_pt, map_location="cpu", weights_only=False)
    questions = ckpt.get("questions") or []
    per_fid: dict[int, list[float]] = {}

    for qd in questions:
        z = qd.get("z_steered", {}).get(steered_alpha_key)
        if z is None:
            keys = sorted(qd.get("z_steered", {}).keys(), key=float)
            z = qd["z_steered"][keys[-1]] if keys else None
        if z is None:
            continue
        z = z.float()
        active = (z.abs() > 1e-8).nonzero(as_tuple=True)[0]
        for i in active.tolist():
            per_fid.setdefault(int(i), []).append(float(z[i].item()))

    rows: list[dict[str, Any]] = []
    for fid, vals in sorted(per_fid.items(), key=lambda x: -max(x[1])):
        t = torch.tensor(vals)
        rows.append(
            {
                "feature_id": fid,
                "n_samples": len(vals),
                "mean": float(t.mean()),
                "p95": float(torch.quantile(t, 0.95)),
                "max": float(t.max()),
            }
        )

    atoms = _load_sta_atoms(run_dir)
    by_cos = sorted(
        [a for a in atoms if a.get("decoder_cosine") is not None],
        key=lambda a: a["decoder_cosine"],
        reverse=True,
    )
    p95_map = {r["feature_id"]: r["p95"] for r in rows}

    def _pack(atom_list: list[dict], k: int | None = None) -> list[dict]:
        out = []
        for a in (atom_list[:k] if k else atom_list):
            fid = a["feature_id"]
            out.append(
                {
                    "feature_id": fid,
                    "decoder_cosine": a.get("decoder_cosine"),
                    "mean_signed_delta": a.get("mean_signed_delta"),
                    "clamp_p95": p95_map.get(fid),
                }
            )
        return out

    doc = {
        "task": "prep-clamp",
        "run_id": run_dir.name,
        "layer": layer_idx,
        "steered_alpha_key": steered_alpha_key,
        "latents_pt": str(latents_pt.resolve()),
        "n_questions": len(questions),
        "n_features_with_steered_activations": len(rows),
        "top_by_steered_activation": rows[:30],
        "option_a_single": _pack(by_cos, 1),
        "option_b_aligned_k": {
            str(k): _pack(by_cos, k) for k in (5, 10, 20, 50)
        },
        "option_b_attribution_k": {
            str(k): _pack(sorted(atoms, key=lambda a: -a["mean_signed_delta"]), k)
            for k in (5, 10, 20, 50)
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def task_encode_vdense(
    run_dir: Path,
    out_json: Path,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    device: torch.device,
) -> Path:
    """Task 2: SAE encoding of dense persona vector vs STA selection."""
    from app.persona.sae_causality import encode_vector
    from app.phase2 import load_sae_for_layer

    vectors_pt = run_dir / "vectors/persona_vectors.pt"
    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_dense = ckpt["v"].float()[layer_idx]

    sae, sae_info = load_sae_for_layer(device, release=sae_release, sae_id=sae_id)
    z = encode_vector(sae, v_dense.to(device))

    vals, idx = torch.topk(z.abs(), k=min(50, z.numel()))
    encoded_top = [
        {"feature_id": int(i), "activation": float(z[i]), "abs": float(z[i].abs())}
        for v, i in zip(vals.tolist(), idx.tolist())
    ]

    atoms = _load_sta_atoms(run_dir)
    sta_top_ids = {a["feature_id"] for a in sorted(atoms, key=lambda a: -a["mean_signed_delta"])[:50]}
    enc_top_ids = {r["feature_id"] for r in encoded_top}
    overlap = sorted(sta_top_ids & enc_top_ids)

    sta_by_fid = {a["feature_id"]: a for a in atoms}

    doc = {
        "task": "encode-vdense",
        "run_id": run_dir.name,
        "layer": layer_idx,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "dense_norm": float(v_dense.norm()),
        "n_active_in_encode": int((z.abs() > 1e-8).sum()),
        "encoded_top50": encoded_top,
        "sta_top50_overlap": {
            "n_overlap": len(overlap),
            "feature_ids": overlap,
            "encoded_only_top50": sorted(enc_top_ids - sta_top_ids)[:20],
            "sta_only_top50": sorted(sta_top_ids - enc_top_ids)[:20],
        },
        "encoded_top_with_sta_meta": [
            {
                **row,
                "in_sta_top50": row["feature_id"] in sta_top_ids,
                "sta_decoder_cosine": sta_by_fid.get(row["feature_id"], {}).get("decoder_cosine"),
                "sta_mean_delta": sta_by_fid.get(row["feature_id"], {}).get("mean_signed_delta"),
            }
            for row in encoded_top[:25]
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (overlap=%d/50)", out_json, len(overlap))
    return out_json


def task_output_score(
    run_dir: Path,
    out_json: Path,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    device: torch.device,
    top_k: int = 50,
) -> Path:
    """Task 5: rank STA atoms by |W_dec[f] · W_U[good-ish]| proxy (Arad-style output relevance)."""
    from app.persona.activations import load_model_and_tokenizer
    from app.phase2 import load_sae_for_layer

    model, tokenizer, dev = load_model_and_tokenizer(device=device)
    sae, sae_info = load_sae_for_layer(dev, release=sae_release, sae_id=sae_id)

    W_dec = sae.W_dec.detach().float().cpu()  # (d_sae, d_in)
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu()
    elif hasattr(model, "model") and hasattr(model.model, "lm_head"):
        W_U = model.model.lm_head.weight.detach().float().cpu()
    else:
        raise RuntimeError("Could not find lm_head for output score")

    # Chunked proxy: max |W_dec[f] @ W_U.T| — avoids OOM on full matmul
    scores = torch.zeros(W_dec.shape[0], dtype=torch.float32)
    chunk = 256
    for start in range(0, W_dec.shape[0], chunk):
        end = min(start + chunk, W_dec.shape[0])
        block = W_dec[start:end] @ W_U.T
        scores[start:end] = block.abs().max(dim=1).values
        del block

    atoms = _load_sta_atoms(run_dir)
    for a in atoms:
        fid = a["feature_id"]
        a["output_score_proxy"] = float(scores[fid].item()) if 0 <= fid < scores.shape[0] else 0.0

    by_attr = sorted(atoms, key=lambda a: -a["mean_signed_delta"])[:top_k]
    by_out = sorted(atoms, key=lambda a: -a["output_score_proxy"])[:top_k]
    by_cos = sorted(
        [a for a in atoms if a.get("decoder_cosine") is not None],
        key=lambda a: -a["decoder_cosine"],
    )[:top_k]

    attr_ids = {a["feature_id"] for a in by_attr}
    out_ids = {a["feature_id"] for a in by_out}
    cos_ids = {a["feature_id"] for a in by_cos}

    doc = {
        "task": "output-score",
        "run_id": run_dir.name,
        "layer": layer_idx,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "top_k": top_k,
        "overlap_attribution_vs_output": sorted(attr_ids & out_ids),
        "overlap_attribution_vs_decoder_cos": sorted(attr_ids & cos_ids),
        "overlap_output_vs_decoder_cos": sorted(out_ids & cos_ids),
        "top_by_attribution": [
            {k: a[k] for k in ("feature_id", "mean_signed_delta", "decoder_cosine", "output_score_proxy")}
            for a in by_attr[:15]
        ],
        "top_by_output_score": [
            {k: a[k] for k in ("feature_id", "mean_signed_delta", "decoder_cosine", "output_score_proxy")}
            for a in by_out[:15]
        ],
        "top_by_decoder_cosine": [
            {k: a[k] for k in ("feature_id", "mean_signed_delta", "decoder_cosine", "output_score_proxy")}
            for a in by_cos[:15]
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def _build_subset_vector(sae, atoms: list[dict], device, dtype):
    from app.persona.sae_common import build_sta_steering_vector
    return build_sta_steering_vector(sae, atoms, device, dtype)


def task_aligned_inject(
    run_dir: Path,
    out_json: Path,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    steer_alpha: float,
    limit: int,
    skip_judge: bool,
    project_id: str | None,
    device: torch.device | None,
) -> Path:
    """Task 3: aligned-only vs misaligned-only vs full sum-and-inject."""
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.quality_gates import score_coherence
    from app.persona.response_style import with_paragraph_cap
    from app.persona.sae_experiment import _generate_steered_reply
    from app.persona.schemas import PersonaTraitArtifact
    from app.phase2 import load_sae_for_layer

    artifact = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric, trait_label=artifact.trait_label)
    questions = artifact.eval_questions[:limit]

    ckpt = torch.load(run_dir / "vectors/persona_vectors.pt", map_location="cpu", weights_only=False)
    u_dense = ckpt["v"].float()[layer_idx]

    model, tokenizer, dev = load_model_and_tokenizer(device=device)
    sae, sae_info = load_sae_for_layer(dev, release=sae_release, sae_id=sae_id)
    dtype = next(model.parameters()).dtype

    atoms = _load_sta_atoms(run_dir)
    aligned = [a for a in atoms if (a.get("decoder_cosine") or 0) >= 0.5]
    misaligned = [a for a in atoms if (a.get("decoder_cosine") or 0) < 0.1]

    variants = {
        "full": atoms,
        "aligned_only": aligned,
        "misaligned_only": misaligned,
    }
    vectors = {name: _build_subset_vector(sae, grp, dev, dtype) for name, grp in variants.items()}

    dense_norm = float(u_dense.norm())
    variant_meta = {}
    for name, u in vectors.items():
        n = float(u.float().norm())
        matched = steer_alpha * (dense_norm / max(n, 1e-8))
        cos = float(torch.dot(u_dense / u_dense.norm(), u.float().cpu() / (u.float().cpu().norm() + 1e-8)))
        variant_meta[name] = {
            "n_atoms": len(variants[name]),
            "norm": n,
            "cosine_to_dense": cos,
            "matched_alpha": matched,
            "raw_alpha": steer_alpha,
        }

    pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    rows: list[dict[str, Any]] = []
    for qi, q in enumerate(questions):
        row: dict[str, Any] = {"question_index": qi, "question": q}
        dense_reply = _generate_steered_reply(
            model, tokenizer, dev, neg_sys, q, layer_idx, u_dense, steer_alpha
        )
        row["dense_reply"] = dense_reply
        if not skip_judge:
            row["dense_trait"] = int(
                score_transcript(judge_instr, neg_sys, q, dense_reply, project_id=pid).score
            )
            row["dense_coherence"] = int(score_coherence(dense_reply, project_id=pid))

        for name, u in vectors.items():
            meta = variant_meta[name]
            alpha = meta["matched_alpha"]
            reply = _generate_steered_reply(model, tokenizer, dev, neg_sys, q, layer_idx, u, alpha)
            row[f"{name}_reply"] = reply
            row[f"{name}_alpha"] = alpha
            if not skip_judge:
                row[f"{name}_trait"] = int(
                    score_transcript(judge_instr, neg_sys, q, reply, project_id=pid).score
                )
                row[f"{name}_coherence"] = int(score_coherence(reply, project_id=pid))
        rows.append(row)
        logger.info("aligned-inject %s/%s", qi + 1, len(questions))

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key, -1) >= 0]
        return sum(vals) / len(vals) if vals else None

    doc = {
        "task": "aligned-inject",
        "run_id": run_dir.name,
        "layer": layer_idx,
        "steer_alpha_dense": steer_alpha,
        "variants": variant_meta,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "skip_judge": skip_judge,
        "comparisons": rows,
        "mean_dense_trait": _mean("dense_trait"),
        "mean_full_trait": _mean("full_trait"),
        "mean_aligned_only_trait": _mean("aligned_only_trait"),
        "mean_misaligned_only_trait": _mean("misaligned_only_trait"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def task_ablation_groups(
    run_dir: Path,
    out_json: Path,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    steer_alpha: float,
    limit: int,
    skip_judge: bool,
    project_id: str | None,
    device: torch.device | None,
) -> Path:
    """Task 4: ablate aligned vs misaligned feature groups during dense steer."""
    import tempfile

    atoms = _load_sta_atoms(run_dir)
    aligned = sorted(
        [a for a in atoms if (a.get("decoder_cosine") or 0) >= 0.5],
        key=lambda a: -a["decoder_cosine"],
    )[:20]
    misaligned = sorted(
        [a for a in atoms if (a.get("decoder_cosine") or 0) < 0.1],
        key=lambda a: -a["mean_signed_delta"],
    )[:20]

    from app.persona.sae_causality import run_ablation_validation

    results: dict[str, Any] = {}
    for label, group in [("aligned_top20", aligned), ("misaligned_top20", misaligned)]:
        attr = {
            "top_positive_features": [
                {
                    "feature_id": a["feature_id"],
                    "mean_delta_steered": a["mean_signed_delta"],
                }
                for a in group
            ]
        }
        tmp = Path(tempfile.mkdtemp()) / f"attr_{label}.json"
        tmp.write_text(json.dumps(attr, indent=2), encoding="utf-8")
        out = out_json.parent / f"ablation_{label}.json"
        run_ablation_validation(
            run_dir,
            tmp,
            out,
            layer_idx=layer_idx,
            steer_alpha=steer_alpha,
            top_k=len(group),
            limit=limit,
            sae_release=sae_release,
            sae_id=sae_id,
            device=device,
            skip_judge=skip_judge,
            project_id=project_id,
            use_eval_questions=True,
        )
        doc = json.loads(out.read_text(encoding="utf-8"))
        results[label] = {
            "n_features": len(group),
            "feature_ids": [a["feature_id"] for a in group],
            "mean_dense_trait": doc.get("mean_dense_trait_score"),
            "mean_ablated_trait": doc.get("mean_ablated_trait_score"),
            "trait_drop": (
                (doc.get("mean_dense_trait_score") or 0) - (doc.get("mean_ablated_trait_score") or 0)
                if doc.get("mean_dense_trait_score") is not None
                else None
            ),
            "out_json": str(out),
        }

    summary = {"task": "ablation-groups", "run_id": run_dir.name, "layer": layer_idx, "groups": results}
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def main() -> int:
    p = argparse.ArgumentParser(description="Parallel SAE ruling experiments")
    p.add_argument("task", choices=[
        "prep-clamp", "encode-vdense", "output-score", "aligned-inject", "ablation-groups", "all-cpu", "all-gpu",
    ])
    p.add_argument("--run-id", default=DEFAULT_RUN)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE)
    p.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    p.add_argument("--latents-pt", default="")
    p.add_argument("--steered-alpha-key", default=DEFAULT_STEERED_KEY)
    p.add_argument("--steer-alpha", type=float, default=1.5)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out-dir", default="")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--project", default="")
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    run_dir = _run_dir(args.run_id)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "sae/parallel_ruling"
    out_dir.mkdir(parents=True, exist_ok=True)
    latents = Path(args.latents_pt) if args.latents_pt else run_dir / DEFAULT_LATENTS

    from app.persona.activations import _pick_device
    cpu = torch.device("cpu")
    gpu = _pick_device()
    sae_dev = cpu if args.force_cpu else gpu

    tasks = [args.task]
    if args.task == "all-cpu":
        tasks = ["prep-clamp", "encode-vdense"]
    elif args.task == "all-gpu":
        tasks = ["aligned-inject", "ablation-groups", "output-score"]

    for task in tasks:
        logger.info("=== TASK %s ===", task)
        if task == "prep-clamp":
            task_prep_clamp(
                run_dir, out_dir / "01_prep_clamp.json",
                latents_pt=latents, steered_alpha_key=args.steered_alpha_key, layer_idx=args.layer,
            )
        elif task == "encode-vdense":
            task_encode_vdense(
                run_dir, out_dir / "02_encode_vdense.json",
                layer_idx=args.layer, sae_release=args.sae_release, sae_id=args.sae_id, device=cpu,
            )
        elif task == "output-score":
            task_output_score(
                run_dir, out_dir / "05_output_score.json",
                layer_idx=args.layer, sae_release=args.sae_release, sae_id=args.sae_id, device=gpu,
            )
        elif task == "aligned-inject":
            task_aligned_inject(
                run_dir, out_dir / "03_aligned_inject.json",
                layer_idx=args.layer, sae_release=args.sae_release, sae_id=args.sae_id,
                steer_alpha=args.steer_alpha, limit=args.limit, skip_judge=args.skip_judge,
                project_id=args.project or None, device=gpu,
            )
        elif task == "ablation-groups":
            task_ablation_groups(
                run_dir, out_dir / "04_ablation_groups.json",
                layer_idx=args.layer, sae_release=args.sae_release, sae_id=args.sae_id,
                steer_alpha=args.steer_alpha, limit=args.limit, skip_judge=args.skip_judge,
                project_id=args.project or None, device=gpu,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
