#!/usr/bin/env python3
"""
SAE feature emergence loop for Good vector.

Phase A: small data steps → step-c/d → calibrate → SAE generate/encode/attribute per checkpoint.
Phase B: fixed best vector → alpha sweep on 5 eval Qs → SAE feature trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PHASE_A_SCHEDULE = [
    {"checkpoint": 1, "pairs": 1, "rollouts_per_q": 1},
    {"checkpoint": 2, "pairs": 1, "rollouts_per_q": 2},
    {"checkpoint": 3, "pairs": 1, "rollouts_per_q": 3},
    {"checkpoint": 4, "pairs": 2, "rollouts_per_q": 2},
    {"checkpoint": 5, "pairs": 2, "rollouts_per_q": 3},
    {"checkpoint": 6, "pairs": 3, "rollouts_per_q": 3},
    {"checkpoint": 7, "pairs": 3, "rollouts_per_q": 5},
    {"checkpoint": 8, "pairs": 4, "rollouts_per_q": 5},
    {"checkpoint": 9, "pairs": 5, "rollouts_per_q": 5},
    {"checkpoint": 10, "pairs": 5, "rollouts_per_q": 10},
]

PHASE_B_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
PHASE_B_EVAL_N = 5

BASELINE_REFERENCE = {
    "run_id": "dnd_good",
    "pairs": 1,
    "rollouts_per_q": 1,
    "training_questions": 18,
    "eval_questions": 9,
    "mean_dense_trait": 21.7,
    "note": "Old scenario re-extract with eval/train overlap",
}

TARGET_DENSE = 75.0
MIN_CHECKPOINTS_BEFORE_EARLY_STOP = 5
LAYER = 31
QUESTIONS_SOURCE = "extraction"
N_EVAL = 20
COHERENCE_FLOOR = 80.0
CALIB_STEP = 0.3
CALIB_MAX_ALPHA = 4.0
SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
SAE_ID = "layer_31_width_16k_l0_small"
TOP_K = 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runs_dir() -> Path:
    from app.persona.config import PERSONA_RUNS_DIR

    return PERSONA_RUNS_DIR


def _run(cmd: list[str], *, check: bool = True) -> None:
    logger.info("RUN %s", " ".join(cmd))
    subprocess.run(cmd, check=check, env=os.environ.copy())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _jaccard(a: set[int], b: set[int]) -> float | None:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def _feature_ids_from_attr(attr: dict[str, Any]) -> set[int]:
    return {int(r["feature_id"]) for r in (attr.get("top_positive_features") or [])}


def _top_features_rows(attr: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in attr.get("top_positive_features") or []:
        rows.append(
            {
                "feature_id": int(r["feature_id"]),
                "magnitude": float(r.get("shared_magnitude", 0)),
                "direction": "positive",
            }
        )
    return rows


def _rollout_stats(rollouts_json: Path) -> dict[str, Any]:
    doc = _load_json(rollouts_json)
    stats = doc.get("stats") or {}
    return {
        "pos_kept": int(stats.get("pos_kept") or 0),
        "neg_kept": int(stats.get("neg_kept") or 0),
        "question_count": int(doc.get("question_count") or 0),
    }


def _split_half(summary_json: Path) -> float | None:
    if not summary_json.is_file():
        return None
    sh = (_load_json(summary_json).get("split_half_cosine") or {}).get(
        "mean_cosine_at_argmax_norm"
    )
    return float(sh) if sh is not None else None


def _trait_at_alpha(cal_doc: dict[str, Any], alpha: float) -> tuple[float | None, float | None]:
    rows = ((cal_doc.get("alpha_sweep") or {}).get("rows")) or []
    for row in rows:
        if abs(float(row.get("alpha", -1)) - float(alpha)) < 1e-6:
            return float(row.get("mean_trait")), float(row.get("mean_coherence"))
    return None, None


def _start_uvicorn(gemma_url: str) -> None:
    _run(["pkill", "-f", "uvicorn app.main:app"], check=False)
    time.sleep(2)
    repo = _repo_root()
    env_file = repo / ".hf.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    subprocess.Popen(
        [str(repo / ".venv" / "bin" / "uvicorn"), "app.main:app", "--host", "127.0.0.1", "--port", "8080"],
        cwd=str(repo),
        stdout=open("/tmp/gemma-uvicorn.log", "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    import urllib.request

    for _ in range(90):
        try:
            with urllib.request.urlopen(f"{gemma_url.rstrip('/')}/health", timeout=5) as r:
                if b'"model_loaded":true' in r.read():
                    logger.info("uvicorn ready")
                    return
        except OSError:
            pass
        time.sleep(5)
    raise RuntimeError("uvicorn failed to start or model not loaded")


def _stop_uvicorn() -> None:
    subprocess.run(["pkill", "-f", "uvicorn app.main:app"], check=False)
    time.sleep(3)


def _calibrate(
    run_dir: Path,
    *,
    project_id: str,
    out_json: Path,
    python: str,
) -> dict[str, Any]:
    """Run calibrate in a subprocess so GPU memory is freed before SAE steps."""
    cfg = run_dir.parent / "dnd_config_good_scale.json"
    if not cfg.is_file():
        bundle = str((run_dir / "artifacts" / "trait_bundle.json").resolve())
        vectors = str((run_dir / "vectors" / "persona_vectors.pt").resolve())
        _save_json(
            cfg,
            {
                "good": {
                    "bundle": bundle,
                    "vectors": vectors,
                    "layer": LAYER,
                }
            },
        )
    _run(
        [
            python,
            "-m",
            "app.persona.vector_compose",
            "calibrate",
            "--config-json",
            str(cfg),
            "--traits-filter",
            "good",
            "--n-questions",
            str(N_EVAL),
            "--step",
            str(CALIB_STEP),
            "--max-alpha",
            str(CALIB_MAX_ALPHA),
            "--coherence-floor",
            str(COHERENCE_FLOOR),
            "--out-json",
            str(out_json),
        ]
    )
    cal_doc = _load_json(out_json)
    if "good" in cal_doc and "scale_recommended" not in cal_doc:
        return cal_doc["good"]
    return cal_doc


def _run_sae_pipeline(
    *,
    run_id: str,
    ckpt_dir: Path,
    alpha: float,
    python: str,
    project_id: str,
    calibration_json: Path,
) -> dict[str, Any]:
    gen_json = ckpt_dir / "generations.json"
    latents_pt = ckpt_dir / "sae_latents.pt"
    attr_json = ckpt_dir / "feature_attribution.json"
    alpha_str = f"{float(alpha):g}"
    alphas_arg = f"0,{alpha_str}"

    _run(
        [
            python,
            "-m",
            "app.persona.sae_experiment",
            "generate",
            "--run-id",
            run_id,
            "--layer",
            str(LAYER),
            "--alphas",
            alphas_arg,
            "--out",
            str(gen_json),
        ]
    )
    _run(
        [
            python,
            "-m",
            "app.persona.sae_experiment",
            "encode",
            "--run-id",
            run_id,
            "--layer",
            str(LAYER),
            "--generations",
            str(gen_json),
            "--out-pt",
            str(latents_pt),
            "--sae-release",
            SAE_RELEASE,
            "--sae-id",
            SAE_ID,
        ]
    )
    _run(
        [
            python,
            "-m",
            "app.persona.sae_experiment",
            "attribute",
            "--run-id",
            run_id,
            "--latents-pt",
            str(latents_pt),
            "--out-json",
            str(attr_json),
            "--steered-alpha",
            alpha_str,
            "--calibration-json",
            str(calibration_json),
            "--top-k",
            str(TOP_K),
        ]
    )
    return _load_json(attr_json)


def run_checkpoint(
    *,
    run_id: str,
    spec: dict[str, int],
    gemma_url: str,
    project_id: str,
    python: str,
    prev_feature_ids: set[int] | None,
) -> tuple[dict[str, Any], set[int]]:
    ckpt_num = int(spec["checkpoint"])
    pairs = int(spec["pairs"])
    rollouts_per_q = int(spec["rollouts_per_q"])
    t0 = time.time()
    run_dir = _runs_dir() / run_id
    ckpt_dir = run_dir / "sae_checkpoints" / f"ckpt_{ckpt_num:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _start_uvicorn(gemma_url)
    try:
        _run(
            [
                python,
                "-m",
                "app.persona.run",
                "step-c",
                "--run-id",
                run_id,
                "--gemma-url",
                gemma_url,
                "--questions-source",
                QUESTIONS_SOURCE,
                "--rollouts-per-q",
                str(rollouts_per_q),
                "--max-pairs",
                str(pairs),
                "--project",
                project_id,
            ]
        )
    finally:
        _stop_uvicorn()

    _run([python, "-m", "app.persona.run", "step-d", "--run-id", run_id])

    vec_src = run_dir / "vectors" / "persona_vectors.pt"
    vec_ckpt = run_dir / "vectors" / f"persona_vectors_ckpt{ckpt_num:02d}.pt"
    shutil.copy2(vec_src, vec_ckpt)

    cal_path = ckpt_dir / "calibration.json"
    cal_doc = _calibrate(run_dir, project_id=project_id, out_json=cal_path, python=python)
    alpha = cal_doc.get("scale_recommended")
    mean_trait, mean_coh = (
        _trait_at_alpha(cal_doc, float(alpha)) if alpha is not None else (None, None)
    )

    attr = _run_sae_pipeline(
        run_id=run_id,
        ckpt_dir=ckpt_dir,
        alpha=float(alpha or 2.0),
        python=python,
        project_id=project_id,
        calibration_json=cal_path,
    )

    stats = _rollout_stats(run_dir / "rollouts" / "extraction_rollouts.json")
    feat_ids = _feature_ids_from_attr(attr)
    overlap = _jaccard(feat_ids, prev_feature_ids) if prev_feature_ids else None

    row: dict[str, Any] = {
        "checkpoint": ckpt_num,
        "pairs": pairs,
        "rollouts_per_q": rollouts_per_q,
        "training_questions": stats["question_count"],
        "raw_rollouts_per_arm": pairs * stats["question_count"] * rollouts_per_q,
        "kept_pos": stats["pos_kept"],
        "kept_neg": stats["neg_kept"],
        "split_half_cosine": _split_half(run_dir / "vectors" / "summary.json"),
        "calibrated_alpha": alpha,
        "mean_dense_trait": mean_trait,
        "mean_dense_coherence": mean_coh,
        "top_features": _top_features_rows(attr),
        "feature_overlap_with_prev": overlap,
        "mean_delta_steered_l2": attr.get("mean_delta_steered_l2"),
        "vectors_ckpt": str(vec_ckpt),
        "sae_dir": str(ckpt_dir),
        "elapsed_minutes": round((time.time() - t0) / 60.0, 1),
    }
    _save_json(ckpt_dir / "checkpoint.json", row)
    return row, feat_ids


def _pick_best_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        c
        for c in checkpoints
        if c.get("mean_dense_trait") is not None and c.get("split_half_cosine") is not None
    ]
    if not scored:
        return checkpoints[-1]
    return max(
        scored,
        key=lambda c: (
            float(c["mean_dense_trait"]),
            float(c.get("split_half_cosine") or 0),
        ),
    )


def _trait_score_eval(
    *,
    run_dir: Path,
    questions: list[str],
    alpha: float,
    project_id: str,
) -> float:
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.quality_gates import _generate_steered
    from app.persona.response_style import with_paragraph_cap
    from app.persona.schemas import PersonaTraitArtifact

    artifact = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts" / "trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)
    ckpt = __import__("torch").load(
        run_dir / "vectors" / "persona_vectors.pt", map_location="cpu", weights_only=False
    )
    direction = ckpt["v"].float()[LAYER]
    model, tokenizer, device = load_model_and_tokenizer(None, device=None)
    direction = direction.to(device=device, dtype=next(model.parameters()).dtype).view(1, 1, -1)

    scores: list[int] = []
    for q in questions:
        reply = _generate_steered(
            model,
            tokenizer,
            device,
            neg_sys,
            q,
            LAYER,
            direction,
            float(alpha),
            max_new_tokens=120,
        )
        js = score_transcript(judge_instr, neg_sys, q, reply, project_id=project_id)
        scores.append(int(js.score))
    return sum(scores) / len(scores) if scores else 0.0


def run_phase_b(
    *,
    run_id: str,
    best_ckpt: dict[str, Any],
    gemma_url: str,
    python: str,
    project_id: str,
) -> dict[str, Any]:
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.response_style import with_paragraph_cap
    from app.persona.sae_encode import assistant_hidden_span_at_layer, encode_hidden_span
    from app.persona.sae_experiment import _generate_steered_reply
    from app.persona.schemas import PersonaTraitArtifact
    from app.phase2 import load_sae_for_layer

    run_dir = _runs_dir() / run_id
    ckpt_num = int(best_ckpt["checkpoint"])
    vec_ckpt = Path(best_ckpt["vectors_ckpt"])
    shutil.copy2(vec_ckpt, run_dir / "vectors" / "persona_vectors.pt")

    eval_path = run_dir / "eval" / "eval_answers.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    _start_uvicorn(gemma_url)
    try:
        _run(
            [
                python,
                "-m",
                "app.persona.run",
                "eval-answers",
                "--run-id",
                run_id,
                "--gemma-url",
                gemma_url,
                "--out",
                str(eval_path),
                "--limit",
                str(PHASE_B_EVAL_N),
            ]
        )
    finally:
        _stop_uvicorn()

    eval_doc = _load_json(eval_path)
    eval_items = (eval_doc.get("items") or [])[:PHASE_B_EVAL_N]
    if not eval_items:
        raise ValueError(f"No eval items in {eval_path}")

    artifact = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts" / "trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys_default = with_paragraph_cap(artifact.neg_system_prompt)

    ckpt = __import__("torch").load(vec_ckpt, map_location="cpu", weights_only=False)
    direction = ckpt["v"].float()[LAYER]
    model, tokenizer, device = load_model_and_tokenizer(None, device=None)
    sae, _sae_info = load_sae_for_layer(device, release=SAE_RELEASE, sae_id=SAE_ID)

    per_alpha: list[dict[str, Any]] = []
    for alpha in PHASE_B_ALPHAS:
        logger.info("Phase B alpha=%s", alpha)
        feat_accum: dict[int, float] = {}
        for qi, item in enumerate(eval_items):
            q = item["question"]
            pos_reply = item.get("pos_reply") or ""
            neg_reply = item.get("neg_reply") or ""
            if alpha == 0.0:
                steered_reply = neg_reply
            else:
                steered_reply = _generate_steered_reply(
                    model,
                    tokenizer,
                    device,
                    neg_sys_default,
                    q,
                    LAYER,
                    direction,
                    float(alpha),
                    rng_seed=42 + qi,
                )
            _, z_neg_mean = encode_hidden_span(
                sae,
                assistant_hidden_span_at_layer(
                    model, tokenizer, device, neg_sys_default, q, neg_reply, LAYER
                )[0],
            )
            _, z_st_mean = encode_hidden_span(
                sae,
                assistant_hidden_span_at_layer(
                    model, tokenizer, device, neg_sys_default, q, steered_reply, LAYER
                )[0],
            )
            delta = (z_st_mean - z_neg_mean).abs()
            for fid in torch.topk(delta, k=min(TOP_K * 2, delta.numel())).indices:
                fid_i = int(fid.item())
                feat_accum[fid_i] = feat_accum.get(fid_i, 0.0) + float(delta[fid].item())

        top_feats = sorted(feat_accum.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        mean_trait = _trait_score_eval(
            run_dir=run_dir, questions=[it["question"] for it in eval_items], alpha=alpha, project_id=project_id
        )
        per_alpha.append(
            {
                "alpha": float(alpha),
                "mean_trait_score": mean_trait,
                "top_features": [
                    {"feature_id": fid, "magnitude": mag} for fid, mag in top_feats
                ],
                "total_sparse_norm": sum(m for _, m in top_feats),
            }
        )

    doc = {
        "vector_from_checkpoint": ckpt_num,
        "vectors_pt": str(vec_ckpt),
        "eval_questions_used": [it["question"] for it in eval_items],
        "alphas": PHASE_B_ALPHAS,
        "per_alpha": per_alpha,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = run_dir / "sae_alpha_sweep.json"
    _save_json(out, doc)
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Good SAE milestone loop")
    parser.add_argument("--run-id", default="dnd_good_scale")
    parser.add_argument("--gemma-url", default="http://127.0.0.1:8080")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--target-dense", type=float, default=TARGET_DENSE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--phase-a-only", action="store_true")
    parser.add_argument("--phase-b-only", action="store_true")
    args = parser.parse_args()

    if not args.project:
        logger.error("Set GOOGLE_CLOUD_PROJECT or pass --project")
        return 1

    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    run_dir = _runs_dir() / args.run_id
    bundle = run_dir / "artifacts" / "trait_bundle.json"
    if not bundle.is_file():
        logger.error("Missing bundle: %s", bundle)
        return 1

    results_path = run_dir / "sae_milestone_results.json"
    doc: dict[str, Any] = {
        "target_dense_trait": args.target_dense,
        "run_id": args.run_id,
        "layer": LAYER,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_reference": BASELINE_REFERENCE,
        "phase_a_checkpoints": [],
        "final_status": "running",
    }
    _save_json(results_path, doc)

    prev_ids: set[int] | None = None
    if not args.phase_b_only:
        for spec in PHASE_A_SCHEDULE:
            ckpt_num = spec["checkpoint"]
            logger.info(
                "=== Phase A checkpoint %s pairs=%s rollouts/q=%s ===",
                ckpt_num,
                spec["pairs"],
                spec["rollouts_per_q"],
            )
            try:
                row, prev_ids = run_checkpoint(
                    run_id=args.run_id,
                    spec=spec,
                    gemma_url=args.gemma_url,
                    project_id=args.project,
                    python=args.python,
                    prev_feature_ids=prev_ids,
                )
            except Exception as exc:
                logger.exception("checkpoint %s failed: %s", ckpt_num, exc)
                doc = _load_json(results_path)
                doc["final_status"] = f"failed_at_ckpt_{ckpt_num}"
                doc["error"] = str(exc)
                doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _save_json(results_path, doc)
                return 1

            doc = _load_json(results_path)
            doc["phase_a_checkpoints"].append(row)
            _save_json(results_path, doc)

            trait = row.get("mean_dense_trait")
            logger.info(
                "ckpt %s trait=%s overlap=%s features=%s elapsed=%s min",
                ckpt_num,
                trait,
                row.get("feature_overlap_with_prev"),
                [f["feature_id"] for f in row.get("top_features", [])[:5]],
                row.get("elapsed_minutes"),
            )
            n_done = len(doc["phase_a_checkpoints"])
            if (
                trait is not None
                and float(trait) >= args.target_dense
                and n_done >= MIN_CHECKPOINTS_BEFORE_EARLY_STOP
            ):
                doc["final_status"] = "target_reached"
                doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _save_json(results_path, doc)
                break
        else:
            doc = _load_json(results_path)
            doc["final_status"] = "phase_a_complete"
            _save_json(results_path, doc)

    if args.phase_a_only:
        return 0

    doc = _load_json(results_path)
    checkpoints = doc.get("phase_a_checkpoints") or []
    if not checkpoints:
        logger.error("No Phase A checkpoints; cannot run Phase B")
        return 1

    best = _pick_best_checkpoint(checkpoints)
    logger.info("Phase B using checkpoint %s trait=%s", best["checkpoint"], best.get("mean_dense_trait"))
    try:
        phase_b = run_phase_b(
            run_id=args.run_id,
            best_ckpt=best,
            gemma_url=args.gemma_url,
            python=args.python,
            project_id=args.project,
        )
    except Exception as exc:
        logger.exception("Phase B failed: %s", exc)
        doc = _load_json(results_path)
        doc["final_status"] = "phase_b_failed"
        doc["error"] = str(exc)
        doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save_json(results_path, doc)
        return 1

    doc = _load_json(results_path)
    doc["phase_b"] = phase_b
    doc["best_checkpoint"] = best["checkpoint"]
    doc["final_status"] = doc.get("final_status", "complete")
    if doc["final_status"] == "running":
        doc["final_status"] = "complete"
    doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _save_json(results_path, doc)
    logger.info("Done. Results: %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
