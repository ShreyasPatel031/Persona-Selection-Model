#!/usr/bin/env python3
"""Iterative Good vector scaling: step-c → step-d → calibrate → score → repeat."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEDULE = [
    {"pairs": 2, "rollouts_per_q": 2},
    {"pairs": 2, "rollouts_per_q": 4},
    {"pairs": 4, "rollouts_per_q": 4},
    {"pairs": 5, "rollouts_per_q": 8},
    {"pairs": 5, "rollouts_per_q": 10},
]

BASELINE_REFERENCE = {
    "run_id": "dnd_good",
    "pairs": 1,
    "rollouts_per_q": 1,
    "training_questions": 18,
    "eval_questions": 9,
    "mean_dense_trait": 21.7,
    "calibrated_alpha": 2.1,
    "note": "Old scenario re-extract with eval/train overlap — not re-run",
}

TARGET_DENSE = 75.0
LAYER = 31
QUESTIONS_SOURCE = "extraction"
N_EVAL = 20
COHERENCE_FLOOR = 80.0
CALIB_STEP = 0.3
CALIB_MAX_ALPHA = 4.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runs_dir() -> Path:
    from app.persona.config import PERSONA_RUNS_DIR

    return PERSONA_RUNS_DIR


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> None:
    logger.info("RUN %s", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=check, env=merged)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_results(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _rollout_stats(rollouts_json: Path) -> dict[str, Any]:
    doc = _load_json(rollouts_json)
    stats = doc.get("stats") or {}
    return {
        "pos_kept": int(stats.get("pos_kept") or 0),
        "neg_kept": int(stats.get("neg_kept") or 0),
        "pos_errors": int(stats.get("pos_errors") or 0),
        "neg_errors": int(stats.get("neg_errors") or 0),
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


def _calibrate(
    run_dir: Path,
    *,
    project_id: str,
    out_json: Path,
) -> dict[str, Any]:
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.coherence_alpha_sweep import run_coherence_alpha_sweep_loaded

    bundle = run_dir / "artifacts" / "trait_bundle.json"
    vectors = run_dir / "vectors" / "persona_vectors.pt"
    model, tokenizer, device = load_model_and_tokenizer(None, device=None)
    jkw: dict[str, Any] = {"project_id": project_id}

    doc = run_coherence_alpha_sweep_loaded(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bundle_path=bundle,
        vectors_pt=vectors,
        layer_idx=LAYER,
        coherence_floor=COHERENCE_FLOOR,
        step=CALIB_STEP,
        max_alpha=CALIB_MAX_ALPHA,
        n_questions=N_EVAL,
        max_new_tokens=120,
        judge_kwargs=jkw,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


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
    venv_uvicorn = repo / ".venv" / "bin" / "uvicorn"
    subprocess.Popen(
        [str(venv_uvicorn), "app.main:app", "--host", "127.0.0.1", "--port", "8080"],
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


def run_iteration(
    *,
    run_id: str,
    iter_num: int,
    pairs: int,
    rollouts_per_q: int,
    gemma_url: str,
    project_id: str,
    python: str,
) -> dict[str, Any]:
    t0 = time.time()
    run_dir = _runs_dir() / run_id
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

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

    cal_path = run_dir / "vectors" / f"calibration_iter{iter_num}.json"
    cal_doc = _calibrate(run_dir, project_id=project_id, out_json=cal_path)
    alpha = cal_doc.get("scale_recommended")
    mean_trait, mean_coh = (
        _trait_at_alpha(cal_doc, float(alpha)) if alpha is not None else (None, None)
    )

    stats = _rollout_stats(run_dir / "rollouts" / "extraction_rollouts.json")
    sh = _split_half(run_dir / "vectors" / "summary.json")
    elapsed = round((time.time() - t0) / 60.0, 1)

    return {
        "iter": iter_num,
        "pairs": pairs,
        "rollouts_per_q": rollouts_per_q,
        "training_questions": stats["question_count"],
        "raw_rollouts_per_arm": pairs * stats["question_count"] * rollouts_per_q,
        "kept_pos": stats["pos_kept"],
        "kept_neg": stats["neg_kept"],
        "split_half_cosine": sh,
        "calibrated_alpha": alpha,
        "mean_dense_trait": mean_trait,
        "mean_dense_coherence": mean_coh,
        "calibration_json": str(cal_path),
        "elapsed_minutes": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Good vector scaling loop")
    parser.add_argument("--run-id", default="dnd_good_scale")
    parser.add_argument("--gemma-url", default="http://127.0.0.1:8080")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--target-dense", type=float, default=TARGET_DENSE)
    parser.add_argument("--python", default=sys.executable)
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
        logger.error("Missing bundle: %s (run step-b first)", bundle)
        return 1

    results_path = run_dir / "scaling_results.json"
    doc: dict[str, Any] = {
        "target_dense_trait": args.target_dense,
        "run_id": args.run_id,
        "layer": LAYER,
        "questions_source": QUESTIONS_SOURCE,
        "n_eval_questions": N_EVAL,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_reference": BASELINE_REFERENCE,
        "iterations": [],
        "final_status": "running",
    }
    _save_results(results_path, doc)

    for i, spec in enumerate(SCHEDULE, start=1):
        logger.info("=== iteration %s/%s pairs=%s rollouts/q=%s ===", i, len(SCHEDULE), spec["pairs"], spec["rollouts_per_q"])
        try:
            row = run_iteration(
                run_id=args.run_id,
                iter_num=i,
                pairs=spec["pairs"],
                rollouts_per_q=spec["rollouts_per_q"],
                gemma_url=args.gemma_url,
                project_id=args.project,
                python=args.python,
            )
        except Exception as exc:
            logger.exception("iteration %s failed: %s", i, exc)
            doc = _load_json(results_path)
            doc["final_status"] = f"failed_at_iter_{i}"
            doc["error"] = str(exc)
            doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _save_results(results_path, doc)
            return 1

        doc = _load_json(results_path)
        doc["iterations"].append(row)
        _save_results(results_path, doc)

        trait = row.get("mean_dense_trait")
        logger.info(
            "iter %s done: trait=%s alpha=%s kept=%s/%s elapsed=%s min",
            i,
            trait,
            row.get("calibrated_alpha"),
            row.get("kept_pos"),
            row.get("kept_neg"),
            row.get("elapsed_minutes"),
        )
        if trait is not None and float(trait) >= args.target_dense:
            doc["final_status"] = "target_reached"
            doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _save_results(results_path, doc)
            logger.info("Target dense trait %.1f reached at iter %s", args.target_dense, i)
            return 0

    doc = _load_json(results_path)
    doc["final_status"] = "max_iterations"
    doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _save_results(results_path, doc)
    logger.info("Completed all %s iterations without reaching target", len(SCHEDULE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
