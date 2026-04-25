"""CLI for persona pipeline steps."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.persona.config import (
    DEFAULT_ARTIFACT_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_PROJECT,
    PERSONA_RUNS_DIR,
)
from app.persona.eval_answers import run_eval_answers
from app.persona.gpu_orchestrate import build_gpu_probe_parser
from app.persona.rollouts import run_step_c
from app.persona.schemas import PersonaTraitArtifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    trait_label: str,
    model_used: str,
    artifact_rel: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "step": "B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trait_label": trait_label,
        "artifact_model": model_used,
        "artifacts": {"trait_bundle": artifact_rel},
        "steps": {},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def cmd_step_b(args: argparse.Namespace) -> int:
    from app.persona.artifact_gen import generate_trait_artifact, parse_artifact_json

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PERSONA_RUNS_DIR / run_id
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    out_path = art_dir / "trait_bundle.json"

    project = args.project or DEFAULT_VERTEX_PROJECT
    location = args.location or DEFAULT_VERTEX_LOCATION
    model = args.model or DEFAULT_ARTIFACT_MODEL

    if args.from_json:
        raw = Path(args.from_json).read_text(encoding="utf-8")
        data = parse_artifact_json(raw)
        artifact = PersonaTraitArtifact.model_validate(data)
        logger.info("Loaded artifact from %s (no Vertex call).", args.from_json)
    else:
        if not project:
            logger.error(
                "Missing project: set GOOGLE_CLOUD_PROJECT or pass --project."
            )
            return 1
        try:
            artifact = generate_trait_artifact(
                args.trait,
                args.trait_description,
                project_id=project,
                location=location,
                model_name=model,
                temperature=args.temperature,
            )
        except Exception as e:
            logger.exception("Artifact generation failed: %s", e)
            return 1

    out_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    try:
        rel = str(out_path.relative_to(PERSONA_RUNS_DIR))
    except ValueError:
        rel = str(out_path)
    _write_manifest(
        run_dir,
        run_id=run_id,
        trait_label=args.trait,
        model_used=model,
        artifact_rel=rel,
    )
    print(out_path.resolve())
    return 0


def _merge_manifest_step_d(
    run_dir: Path,
    vectors_rel: str,
    *,
    layer_recommendation_v1: dict | None = None,
) -> None:
    mpath = run_dir / "manifest.json"
    if not mpath.is_file():
        return
    data = json.loads(mpath.read_text(encoding="utf-8"))
    step_d: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "persona_vectors": vectors_rel,
    }
    if layer_recommendation_v1:
        step_d["layer_recommendation_v1"] = layer_recommendation_v1
        rec = layer_recommendation_v1.get("recommended_layer")
        if rec is not None:
            step_d["recommended_layer"] = rec
    data.setdefault("steps", {})["D"] = step_d
    mpath.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _merge_manifest_sanity_eval_projection(run_dir: Path, report_rel: str) -> None:
    """Plan testing §4 output — not plan Step E (Appendix B.4 lives under Step D)."""
    mpath = run_dir / "manifest.json"
    if not mpath.is_file():
        return
    data = json.loads(mpath.read_text(encoding="utf-8"))
    steps = data.setdefault("steps", {})
    steps.pop("E", None)  # remove mistaken plan-Step-E key if present
    steps["sanity_eval_projection"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report": report_rel,
    }
    mpath.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _merge_manifest_step_c(
    run_dir: Path,
    rollouts_rel: str,
    *,
    jsonl_rel: str | None = None,
) -> None:
    mpath = run_dir / "manifest.json"
    if not mpath.is_file():
        return
    data = json.loads(mpath.read_text(encoding="utf-8"))
    step_c = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rollouts_extraction": rollouts_rel,
    }
    if jsonl_rel:
        step_c["rollouts_jsonl"] = jsonl_rel
    data.setdefault("steps", {})["C"] = step_c
    mpath.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cmd_eval_answers(args: argparse.Namespace) -> int:
    if args.bundle:
        bundle_path = Path(args.bundle).resolve()
    else:
        if not args.run_id:
            logger.error("Provide --bundle PATH or --run-id RUN_ID.")
            return 1
        bundle_path = (
            PERSONA_RUNS_DIR / args.run_id / "artifacts" / "trait_bundle.json"
        ).resolve()
    if not bundle_path.is_file():
        logger.error("Trait bundle not found: %s", bundle_path)
        return 1

    if args.out:
        out_path = Path(args.out).resolve()
    elif bundle_path.parent.name == "artifacts":
        out_path = (bundle_path.parent.parent / "eval" / "eval_answers.json").resolve()
    else:
        out_path = (bundle_path.parent / "eval_answers.json").resolve()

    limit = args.limit if args.limit > 0 else 0
    try:
        written = run_eval_answers(
            bundle_path,
            args.gemma_url,
            out_path,
            limit=limit,
            timeout=args.timeout,
            paragraph_cap=not args.no_paragraph_cap,
        )
    except Exception as e:
        logger.exception("eval-answers failed: %s", e)
        return 1
    print(written.resolve())
    return 0


def cmd_step_c(args: argparse.Namespace) -> int:
    if args.bundle:
        bundle_path = Path(args.bundle).resolve()
    else:
        if not args.run_id:
            logger.error("Provide --bundle PATH or --run-id RUN_ID.")
            return 1
        bundle_path = (
            PERSONA_RUNS_DIR / args.run_id / "artifacts" / "trait_bundle.json"
        ).resolve()
    if not bundle_path.is_file():
        logger.error("Trait bundle not found: %s", bundle_path)
        return 1

    if args.out:
        out_path = Path(args.out).resolve()
    elif bundle_path.parent.name == "artifacts":
        out_path = (
            bundle_path.parent.parent / "rollouts" / "extraction_rollouts.json"
        ).resolve()
    else:
        out_path = (bundle_path.parent / "extraction_rollouts.json").resolve()

    jsonl_path = Path(args.jsonl_out).resolve() if args.jsonl_out else None

    from_rollouts = Path(args.from_rollouts).resolve() if args.from_rollouts else None
    if from_rollouts and not from_rollouts.is_file():
        logger.error("Missing --from-rollouts file: %s", from_rollouts)
        return 1

    skip_judge = args.skip_judge
    if not skip_judge:
        proj = args.project or DEFAULT_VERTEX_PROJECT
        if not proj:
            logger.error(
                "Vertex judge requires GOOGLE_CLOUD_PROJECT or --project "
                "(or use --skip-judge for rollouts only)."
            )
            return 1

    limit = args.limit if args.limit > 0 else 0
    try:
        written, jsonl_written = run_step_c(
            bundle_path,
            args.gemma_url,
            out_path,
            jsonl_path=jsonl_path,
            limit=limit,
            timeout=args.timeout,
            paragraph_cap=not args.no_paragraph_cap,
            skip_judge=skip_judge,
            from_rollouts_json=from_rollouts,
            project_id=args.project or DEFAULT_VERTEX_PROJECT,
            location=args.location or DEFAULT_VERTEX_LOCATION,
            judge_model=args.judge_model or None,
            pos_threshold=(
                args.pos_threshold if args.pos_threshold >= 0 else None
            ),
            neg_threshold=(
                args.neg_threshold if args.neg_threshold >= 0 else None
            ),
            rollouts_per_q=max(1, int(args.rollouts_per_q)),
            sampling_temperature=float(args.sampling_temperature),
        )
    except Exception as e:
        logger.exception("step-c failed: %s", e)
        return 1

    if bundle_path.parent.name == "artifacts":
        run_dir = bundle_path.parent.parent
        try:
            rel = str(written.relative_to(PERSONA_RUNS_DIR))
        except ValueError:
            rel = str(written)
        jsonl_rel = None
        if jsonl_written:
            try:
                jsonl_rel = str(jsonl_written.relative_to(PERSONA_RUNS_DIR))
            except ValueError:
                jsonl_rel = str(jsonl_written)
        _merge_manifest_step_c(run_dir, rel, jsonl_rel=jsonl_rel)

    print(written.resolve())
    if jsonl_written:
        print(jsonl_written.resolve())
    return 0


def cmd_step_d(args: argparse.Namespace) -> int:
    if not args.run_id:
        logger.error("step-d requires --run-id.")
        return 1
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"

    from app.persona.activations import run_step_d

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    jsonl = (
        Path(args.rollouts_jsonl).resolve()
        if args.rollouts_jsonl
        else run_dir / "rollouts" / "rollouts.jsonl"
    )
    if not jsonl.is_file():
        logger.error("Missing rollouts jsonl: %s", jsonl)
        return 1

    vec_dir = run_dir / "vectors"
    out_pt = Path(args.out_pt).resolve() if args.out_pt else vec_dir / "persona_vectors.pt"
    summary = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else vec_dir / "summary.json"
    )

    try:
        run_step_d(
            jsonl,
            out_pt,
            summary,
            model_id=args.model_id or None,
            device=None,
        )
    except Exception as e:
        logger.exception("step-d failed: %s", e)
        return 1

    try:
        rel = str(out_pt.relative_to(PERSONA_RUNS_DIR))
    except ValueError:
        rel = str(out_pt)
    layer_v1 = None
    if summary.is_file():
        try:
            layer_v1 = json.loads(summary.read_text(encoding="utf-8")).get(
                "layer_recommendation_v1"
            )
        except (json.JSONDecodeError, OSError):
            pass
    _merge_manifest_step_d(run_dir, rel, layer_recommendation_v1=layer_v1)
    print(out_pt.resolve())
    return 0


def cmd_sanity_eval_projection(args: argparse.Namespace) -> int:
    """Plan doc 'Testing / exit criteria' §4 — not plan Step E (Appendix B.4)."""
    if getattr(args, "_deprecated_step_e_alias", False):
        logger.warning(
            "CLI `step-e` is a deprecated alias. Use `sanity-eval-projection`. "
            "In the pipeline plan, Step E is Appendix B.4 layer selection (v1 in Step D "
            "summary/manifest; v2 steering sweep not implemented)."
        )
    if not args.run_id:
        logger.error("sanity-eval-projection requires --run-id.")
        return 1
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    bundle_path = (
        Path(args.bundle).resolve()
        if args.bundle
        else (run_dir / "artifacts" / "trait_bundle.json").resolve()
    )
    vectors_pt = (
        Path(args.vectors_pt).resolve()
        if args.vectors_pt
        else (run_dir / "vectors" / "persona_vectors.pt").resolve()
    )
    eval_path = (
        Path(args.eval_json).resolve()
        if args.eval_json
        else (run_dir / "eval" / "eval_answers.json").resolve()
    )
    out_path = (
        Path(args.out).resolve()
        if args.out
        else (run_dir / "eval" / "sanity_eval_projection.json").resolve()
    )

    if not bundle_path.is_file():
        logger.error("Trait bundle not found: %s", bundle_path)
        return 1
    if not vectors_pt.is_file():
        logger.error("persona_vectors.pt not found: %s", vectors_pt)
        return 1

    if not eval_path.is_file():
        if args.refresh_eval:
            try:
                run_eval_answers(
                    bundle_path,
                    args.gemma_url,
                    eval_path,
                    limit=args.limit if args.limit > 0 else 0,
                    timeout=args.timeout,
                    paragraph_cap=not args.no_paragraph_cap,
                )
            except Exception as e:
                logger.exception("sanity-eval-projection --refresh-eval failed: %s", e)
                return 1
        else:
            logger.error(
                "Missing eval answers: %s. Run eval-answers first, or pass "
                "--refresh-eval with a working --gemma-url.",
                eval_path,
            )
            return 1

    from app.persona.vector_probe import run_sanity_eval_projection

    default_layer = args.layer if args.layer is not None else None
    try:
        written = run_sanity_eval_projection(
            bundle_path,
            vectors_pt,
            eval_path,
            out_path,
            model_id=args.model_id or None,
            device=None,
            default_layer=default_layer,
            limit=args.limit if args.limit > 0 else 0,
        )
    except Exception as e:
        logger.exception("sanity-eval-projection failed: %s", e)
        return 1

    try:
        rel = str(written.relative_to(PERSONA_RUNS_DIR))
    except ValueError:
        rel = str(written)
    _merge_manifest_sanity_eval_projection(run_dir, rel)
    print(written.resolve())
    return 0


def _attach_sanity_eval_projection_args(
    p: argparse.ArgumentParser, *, deprecated_alias: bool
) -> None:
    p.add_argument(
        "--run-id",
        required=True,
        help="Run id under persona_runs/ (bundle, vectors, eval paths).",
    )
    p.add_argument(
        "--bundle",
        default="",
        help="trait_bundle.json path (default: <run-id>/artifacts/trait_bundle.json).",
    )
    p.add_argument(
        "--vectors-pt",
        default="",
        help="persona_vectors.pt path (default: <run-id>/vectors/persona_vectors.pt).",
    )
    p.add_argument(
        "--eval-json",
        default="",
        help="eval_answers.json from eval-answers (default: <run-id>/eval/eval_answers.json).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output JSON (default: <run-id>/eval/sanity_eval_projection.json).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index for summary fraction (default: middle layer L//2).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max eval questions (0 = all); also passed to --refresh-eval.",
    )
    p.add_argument(
        "--gemma-url",
        default="http://127.0.0.1:8080",
        help="Gemma /chat base URL (only with --refresh-eval).",
    )
    p.add_argument(
        "--refresh-eval",
        action="store_true",
        help="If eval JSON is missing, run eval-answers against --gemma-url first.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=720,
        help="Per-request HTTP timeout when using --refresh-eval.",
    )
    p.add_argument(
        "--no-paragraph-cap",
        action="store_true",
        help="Do not append one-paragraph suffix when using --refresh-eval.",
    )
    p.add_argument(
        "--model-id",
        default="",
        help="HF model id for teacher forwards (default: GEMMA_MODEL_ID).",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set PERSONA_FORCE_CPU=1 for this process.",
    )
    p.set_defaults(
        func=cmd_sanity_eval_projection,
        _deprecated_step_e_alias=deprecated_alias,
    )


def cmd_steering_ramp(args: argparse.Namespace) -> int:
    if not args.run_id:
        logger.error("steering-ramp requires --run-id.")
        return 1
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    bundle_path = (
        Path(args.bundle).resolve()
        if args.bundle
        else (run_dir / "artifacts" / "trait_bundle.json").resolve()
    )
    vectors_pt = (
        Path(args.vectors_pt).resolve()
        if args.vectors_pt
        else (run_dir / "vectors" / "persona_vectors.pt").resolve()
    )
    out_path = (
        Path(args.out).resolve()
        if args.out
        else (run_dir / "eval" / "steering_ramp.json").resolve()
    )

    if not bundle_path.is_file():
        logger.error("Trait bundle not found: %s", bundle_path)
        return 1
    if not vectors_pt.is_file():
        logger.error("persona_vectors.pt not found: %s", vectors_pt)
        return 1

    layer_idx = args.layer
    if layer_idx is None:
        mpath = run_dir / "manifest.json"
        if mpath.is_file():
            try:
                man = json.loads(mpath.read_text(encoding="utf-8"))
                layer_idx = (
                    (man.get("steps") or {}).get("D") or {}
                ).get("recommended_layer")
            except (json.JSONDecodeError, OSError):
                layer_idx = None
        if layer_idx is None:
            sum_path = run_dir / "vectors" / "summary.json"
            if sum_path.is_file():
                try:
                    layer_idx = (
                        json.loads(sum_path.read_text(encoding="utf-8"))
                        .get("layer_recommendation_v1") or {}
                    ).get("recommended_layer")
                except (json.JSONDecodeError, OSError):
                    layer_idx = None
        if layer_idx is None:
            layer_idx = 22
            logger.info("No recommended_layer in manifest/summary; using default 22.")

    if args.question:
        question = args.question
    else:
        raw = bundle_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        eq = data.get("eval_questions") or []
        ei = min(max(args.eval_index, 0), len(eq) - 1) if eq else -1
        if ei < 0:
            logger.error("No eval_questions in bundle; pass --question TEXT.")
            return 1
        question = eq[ei]

    from app.persona.steering_demo import run_steering_ramp

    try:
        written = run_steering_ramp(
            bundle_path,
            vectors_pt,
            out_path,
            question=question,
            layer_idx=int(layer_idx),
            model_id=args.model_id or None,
            device=None,
            n_steps=args.steps,
            alpha_max=args.alpha_max,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            use_cache=not args.no_kv_cache,
            steer_last_token_only=not getattr(args, "steer_all_tokens", False),
            rng_seed=(args.rng_seed if args.do_sample else None),
        )
    except Exception as e:
        logger.exception("steering-ramp failed: %s", e)
        return 1
    print(written.resolve())
    return 0


def cmd_steering_ab(args: argparse.Namespace) -> int:
    if not args.run_id:
        logger.error("steering-ab requires --run-id.")
        return 1
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    bundle_path = (
        Path(args.bundle).resolve()
        if args.bundle
        else (run_dir / "artifacts" / "trait_bundle.json").resolve()
    )
    vectors_pt = (
        Path(args.vectors_pt).resolve()
        if args.vectors_pt
        else (run_dir / "vectors" / "persona_vectors.pt").resolve()
    )
    out_path = (
        Path(args.out).resolve()
        if args.out
        else (run_dir / "eval" / "steering_ab_raw_v.json").resolve()
    )

    if not bundle_path.is_file():
        logger.error("Trait bundle not found: %s", bundle_path)
        return 1
    if not vectors_pt.is_file():
        logger.error("persona_vectors.pt not found: %s", vectors_pt)
        return 1

    layer_idx = args.layer
    if layer_idx is None:
        mpath = run_dir / "manifest.json"
        if mpath.is_file():
            try:
                man = json.loads(mpath.read_text(encoding="utf-8"))
                layer_idx = (
                    (man.get("steps") or {}).get("D") or {}
                ).get("recommended_layer")
            except (json.JSONDecodeError, OSError):
                layer_idx = None
        if layer_idx is None:
            sum_path = run_dir / "vectors" / "summary.json"
            if sum_path.is_file():
                try:
                    layer_idx = (
                        json.loads(sum_path.read_text(encoding="utf-8"))
                        .get("layer_recommendation_v1") or {}
                    ).get("recommended_layer")
                except (json.JSONDecodeError, OSError):
                    layer_idx = None
        if layer_idx is None:
            layer_idx = 22
            logger.info("No recommended_layer in manifest/summary; using default 22.")

    if args.question:
        question = args.question
    else:
        raw = bundle_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        eq = data.get("eval_questions") or []
        ei = min(max(args.eval_index, 0), len(eq) - 1) if eq else -1
        if ei < 0:
            logger.error("No eval_questions in bundle; pass --question TEXT.")
            return 1
        question = eq[ei]

    from app.persona.steering_demo import run_steering_ab_compare

    try:
        alphas_arg = (getattr(args, "alphas", None) or "").strip()
        steering_alphas = (
            [float(x.strip()) for x in alphas_arg.split(",") if x.strip()]
            if alphas_arg
            else None
        )

        written = run_steering_ab_compare(
            bundle_path,
            vectors_pt,
            out_path,
            question=question,
            layer_idx=int(layer_idx),
            model_id=args.model_id or None,
            device=None,
            alpha=float(args.alpha),
            steering_alphas=steering_alphas,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            use_cache=not args.no_kv_cache,
            steer_last_token_only=args.steer_last_token_only,
            rng_seed=(args.rng_seed if args.do_sample else None),
            include_pos_baseline=args.with_pos_baseline,
        )
    except Exception as e:
        logger.exception("steering-ab failed: %s", e)
        return 1
    print(written.resolve())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if not args.run_id:
        logger.error("validate requires --run-id.")
        return 1
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"

    run_dir = (PERSONA_RUNS_DIR / args.run_id).resolve()
    bundle_path = (
        Path(args.bundle).resolve()
        if args.bundle
        else (run_dir / "artifacts" / "trait_bundle.json").resolve()
    )
    vectors_pt = (
        Path(args.vectors_pt).resolve()
        if args.vectors_pt
        else (run_dir / "vectors" / "persona_vectors.pt").resolve()
    )
    rollouts_jsonl = (
        Path(args.rollouts_jsonl).resolve()
        if args.rollouts_jsonl
        else (run_dir / "rollouts" / "rollouts.jsonl")
    )
    sanity_json = (
        Path(args.sanity_json).resolve()
        if args.sanity_json
        else (run_dir / "eval" / "sanity_eval_projection.json")
    )
    out_path = (
        Path(args.out).resolve()
        if args.out
        else (run_dir / "eval" / "validation_report.json").resolve()
    )

    if not bundle_path.is_file():
        logger.error("Trait bundle not found: %s", bundle_path)
        return 1
    if not vectors_pt.is_file():
        logger.error("persona_vectors.pt not found: %s", vectors_pt)
        return 1

    alphas_str = (args.alphas or "1,2,3,4,5,6,7,8,9,10").strip()
    alphas = tuple(float(x.strip()) for x in alphas_str.split(",") if x.strip())

    sweep_stop: int | None
    if getattr(args, "no_sweep_coherence_stop", False):
        sweep_stop = None
    else:
        sweep_stop = int(getattr(args, "sweep_coherence_stop", 15))

    steering_replies_out: Path | None = None
    if getattr(args, "steering_replies_out", "").strip():
        steering_replies_out = Path(args.steering_replies_out.strip()).resolve()

    from app.persona.quality_gates import run_validation

    try:
        written = run_validation(
            bundle_path,
            vectors_pt,
            out_path,
            run_id=args.run_id,
            rollouts_jsonl=rollouts_jsonl if rollouts_jsonl.is_file() else None,
            sanity_json=sanity_json if sanity_json.is_file() else None,
            model_id=args.model_id or None,
            device=None,
            n_candidate_layers=args.n_candidate_layers,
            n_questions=args.n_questions,
            alphas=alphas,
            sweep_stop_coherence_below=sweep_stop,
            judge_project=args.project or DEFAULT_VERTEX_PROJECT,
            judge_location=args.location or DEFAULT_VERTEX_LOCATION,
            judge_model=args.judge_model or None,
            skip_model_gates=args.skip_model_gates,
            steering_replies_out=steering_replies_out,
        )
    except Exception as e:
        logger.exception("validate failed: %s", e)
        return 1
    print(written.resolve())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persona pipeline CLI",
        prog="python -m app.persona.run",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_b = sub.add_parser(
        "step-b",
        help="Generate validated trait artifact (contrast prompts, Q lists, judge rubric) via Vertex Gemini.",
    )
    p_b.add_argument(
        "--trait",
        required=True,
        help="Short trait label (e.g. ability to find the worst in every situation).",
    )
    p_b.add_argument(
        "--trait-description",
        required=True,
        help="Author notes: when the trait applies, style rules, humor, etc.",
    )
    p_b.add_argument(
        "--run-id",
        default="",
        help="Subdirectory under persona_runs/ (default: UTC timestamp).",
    )
    p_b.add_argument(
        "--project",
        default="",
        help="GCP project id (default: GOOGLE_CLOUD_PROJECT).",
    )
    p_b.add_argument(
        "--location",
        default="",
        help="Vertex region (default: VERTEX_LOCATION or us-central1).",
    )
    p_b.add_argument(
        "--model",
        default="",
        help="Gemini model id (default: PERSONA_ARTIFACT_MODEL or PERSONA_JUDGE_MODEL).",
    )
    p_b.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature for artifact generation.",
    )
    p_b.add_argument(
        "--from-json",
        default="",
        metavar="PATH",
        help="Skip Vertex: validate existing JSON file and write trait_bundle.json.",
    )
    p_b.set_defaults(func=cmd_step_b)

    p_ev = sub.add_parser(
        "eval-answers",
        help="For each eval_question, POST /chat on Gemma with pos vs neg system prompts; save JSON.",
    )
    p_ev.add_argument(
        "--bundle",
        default="",
        help="Path to trait_bundle.json (default: derive from --run-id).",
    )
    p_ev.add_argument(
        "--run-id",
        default="",
        help="Run id under persona_runs/ (uses .../artifacts/trait_bundle.json).",
    )
    p_ev.add_argument(
        "--gemma-url",
        default="http://127.0.0.1:8080",
        help="Base URL of Gemma FastAPI (no trailing slash).",
    )
    p_ev.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: <run>/eval/eval_answers.json next to bundle).",
    )
    p_ev.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of eval questions (0 = all).",
    )
    p_ev.add_argument(
        "--timeout",
        type=int,
        default=720,
        help="Per-request HTTP timeout seconds.",
    )
    p_ev.add_argument(
        "--no-paragraph-cap",
        action="store_true",
        help="Do not append the one-paragraph reply constraint to system prompts.",
    )
    p_ev.set_defaults(func=cmd_eval_answers)

    p_c = sub.add_parser(
        "step-c",
        help="Step C §2.2: Gemma extraction rollouts + Vertex judge (0–100 JSON) + filter; writes extraction_rollouts.json + rollouts.jsonl.",
    )
    p_c.add_argument("--bundle", default="", help="Path to trait_bundle.json.")
    p_c.add_argument(
        "--run-id",
        default="",
        help="Run id under persona_runs/ (uses .../artifacts/trait_bundle.json).",
    )
    p_c.add_argument(
        "--gemma-url",
        default="http://127.0.0.1:8080",
        help="Base URL of Gemma FastAPI.",
    )
    p_c.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: <run>/rollouts/extraction_rollouts.json).",
    )
    p_c.add_argument(
        "--jsonl-out",
        default="",
        help="Path for rollouts.jsonl (default: same dir as --out / rollouts.jsonl).",
    )
    p_c.add_argument(
        "--from-rollouts",
        default="",
        metavar="PATH",
        help="Skip Gemma: load prior extraction_rollouts.json items; still need --bundle for judge rubric.",
    )
    p_c.add_argument(
        "--skip-judge",
        action="store_true",
        help="Rollouts only (no Vertex); no rollouts.jsonl.",
    )
    p_c.add_argument(
        "--project",
        default="",
        help="GCP project for Vertex judge (default: GOOGLE_CLOUD_PROJECT).",
    )
    p_c.add_argument(
        "--location",
        default="",
        help="Vertex region (default: VERTEX_LOCATION).",
    )
    p_c.add_argument(
        "--judge-model",
        default="",
        help=f"Vertex model for judge (default: PERSONA_JUDGE_MODEL or {DEFAULT_JUDGE_MODEL!r}).",
    )
    p_c.add_argument(
        "--pos-threshold",
        type=int,
        default=-1,
        help="Keep pos arm if score > this (default: env PERSONA_JUDGE_POS_MIN or 50).",
    )
    p_c.add_argument(
        "--neg-threshold",
        type=int,
        default=-1,
        help="Keep neg arm if score < this (default: env PERSONA_JUDGE_NEG_MAX or 50).",
    )
    p_c.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max extraction questions (0 = all).",
    )
    p_c.add_argument(
        "--rollouts-per-q",
        type=int,
        default=1,
        help="Samples per (contrast pair, extraction question); paper §2.2 uses 10 (set Gemma /chat do_sample).",
    )
    p_c.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="Temperature for Gemma when --rollouts-per-q > 1.",
    )
    p_c.add_argument(
        "--timeout",
        type=int,
        default=720,
        help="Per-request HTTP timeout seconds.",
    )
    p_c.add_argument(
        "--no-paragraph-cap",
        action="store_true",
        help="Do not append the one-paragraph reply constraint.",
    )
    p_c.set_defaults(func=cmd_step_c)

    build_gpu_probe_parser(sub)

    p_d = sub.add_parser(
        "step-d",
        help="Step D: in-process Gemma teacher-forwards on kept rollouts; mean assistant-token "
        "hidden states per layer; save v = mean_pos - mean_neg to vectors/persona_vectors.pt.",
    )
    p_d.add_argument(
        "--run-id",
        required=True,
        help="Run id under persona_runs/ (default jsonl: .../rollouts/rollouts.jsonl).",
    )
    p_d.add_argument(
        "--rollouts-jsonl",
        default="",
        help="Path to rollouts.jsonl (default: <run-id>/rollouts/rollouts.jsonl).",
    )
    p_d.add_argument(
        "--out-pt",
        default="",
        help="Output .pt path (default: <run-id>/vectors/persona_vectors.pt).",
    )
    p_d.add_argument(
        "--summary-json",
        default="",
        help="Metadata JSON path (default: <run-id>/vectors/summary.json).",
    )
    p_d.add_argument(
        "--model-id",
        default="",
        help="HF model id (default: GEMMA_MODEL_ID or google/gemma-3-4b-it).",
    )
    p_d.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set PERSONA_FORCE_CPU=1 for this process.",
    )
    p_d.set_defaults(func=cmd_step_d)

    p_sanity = sub.add_parser(
        "sanity-eval-projection",
        help="Plan testing §4: eval pos/neg activations projected onto v_ℓ (not plan Step E).",
    )
    _attach_sanity_eval_projection_args(p_sanity, deprecated_alias=False)
    p_step_e_legacy = sub.add_parser(
        "step-e",
        help="Deprecated: use sanity-eval-projection. Plan Step E = Appendix B.4 (Step D output).",
    )
    _attach_sanity_eval_projection_args(p_step_e_legacy, deprecated_alias=True)

    p_sr = sub.add_parser(
        "steering-ramp",
        help="5-step demo: neg system + add α·(v_ℓ/||v_ℓ||) at layer ℓ during generate (α sweeps 0→alpha-max).",
    )
    p_sr.add_argument("--run-id", required=True, help="Run id under persona_runs/.")
    p_sr.add_argument(
        "--bundle",
        default="",
        help="trait_bundle.json (default: <run-id>/artifacts/trait_bundle.json).",
    )
    p_sr.add_argument(
        "--vectors-pt",
        default="",
        help="persona_vectors.pt (default: <run-id>/vectors/persona_vectors.pt).",
    )
    p_sr.add_argument(
        "--out",
        default="",
        help="Output JSON (default: <run-id>/eval/steering_ramp.json).",
    )
    p_sr.add_argument(
        "--layer",
        type=int,
        default=None,
        metavar="N",
        help="Decoder layer index for hook (default: manifest/summary recommended_layer, else 22).",
    )
    p_sr.add_argument(
        "--question",
        default="",
        help="User message (default: eval_questions[--eval-index]).",
    )
    p_sr.add_argument(
        "--eval-index",
        type=int,
        default=0,
        help="Which eval_questions[] to use if --question omitted (default: 0).",
    )
    p_sr.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of α values from 0 to alpha-max inclusive (default: 5).",
    )
    p_sr.add_argument(
        "--alpha-max",
        type=float,
        default=6.0,
        help="Max α on unit direction û=v/||v|| (default: 6). Tune if replies collapse or barely change.",
    )
    p_sr.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation budget per iteration.",
    )
    p_sr.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample instead of greedy decode.",
    )
    p_sr.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Only used with --do-sample.",
    )
    p_sr.add_argument(
        "--model-id",
        default="",
        help="HF model id (default: GEMMA_MODEL_ID).",
    )
    p_sr.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set PERSONA_FORCE_CPU=1.",
    )
    p_sr.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="generate(use_cache=False): much slower but can help if steering seems no-op.",
    )
    p_sr.add_argument(
        "--steer-all-tokens",
        action="store_true",
        help="Add α·û at every position (default: last position only, usually stronger for decode).",
    )
    p_sr.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Base seed for --do-sample (per step seed += step*10007) so five samples can differ.",
    )
    p_sr.set_defaults(func=cmd_steering_ramp, steer_all_tokens=False)

    p_sab = sub.add_parser(
        "steering-ab",
        help="A/B: neg only vs neg + α·v_ℓ (raw v). Default α=1 is one full v_ℓ. "
        "Use --with-pos-baseline for a third reply: true jester via pos system prompt.",
    )
    p_sab.add_argument("--run-id", required=True, help="Run id under persona_runs/.")
    p_sab.add_argument(
        "--bundle",
        default="",
        help="trait_bundle.json (default: <run-id>/artifacts/trait_bundle.json).",
    )
    p_sab.add_argument(
        "--vectors-pt",
        default="",
        help="persona_vectors.pt (default: <run-id>/vectors/persona_vectors.pt).",
    )
    p_sab.add_argument(
        "--out",
        default="",
        help="Output JSON (default: <run-id>/eval/steering_ab_raw_v.json).",
    )
    p_sab.add_argument(
        "--layer",
        type=int,
        default=None,
        metavar="N",
        help="Decoder layer index (default: manifest/summary recommended_layer, else 22).",
    )
    p_sab.add_argument(
        "--question",
        default="",
        help="User message (default: eval_questions[--eval-index]).",
    )
    p_sab.add_argument(
        "--eval-index",
        type=int,
        default=0,
        help="Which eval_questions[] if --question omitted (default: 0).",
    )
    p_sab.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Used when --alphas is omitted: single steered run with this α.",
    )
    p_sab.add_argument(
        "--alphas",
        default="",
        metavar="LIST",
        help="Comma-separated α values, e.g. 1,2,5,10 — one model load, multiple steered decodes (overrides --alpha for steering).",
    )
    p_sab.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation budget per arm.",
    )
    p_sab.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample instead of greedy decode.",
    )
    p_sab.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Only used with --do-sample.",
    )
    p_sab.add_argument(
        "--model-id",
        default="",
        help="HF model id (default: GEMMA_MODEL_ID).",
    )
    p_sab.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set PERSONA_FORCE_CPU=1.",
    )
    p_sab.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="generate(use_cache=False).",
    )
    p_sab.add_argument(
        "--steer-last-token-only",
        action="store_true",
        help="Add α·v only at last position (default: all positions, closer to paper §3.2).",
    )
    p_sab.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for --do-sample (same seed for A and B).",
    )
    p_sab.add_argument(
        "--with-pos-baseline",
        action="store_true",
        help="Also generate with pos (jester) system prompt only — compare to steered neg.",
    )
    p_sab.set_defaults(func=cmd_steering_ab)

    # ── validate ──────────────────────────────────────────────────────────
    p_val = sub.add_parser(
        "validate",
        help="Run automated quality gates on extracted vectors: data sufficiency, "
        "separation, layer selection (Appendix B.4), steering effectiveness + coherence. "
        "Outputs pass/fail verdict + recommended layer & alpha.",
    )
    p_val.add_argument("--run-id", required=True, help="Run id under persona_runs/.")
    p_val.add_argument(
        "--bundle",
        default="",
        help="trait_bundle.json (default: <run-id>/artifacts/trait_bundle.json).",
    )
    p_val.add_argument(
        "--vectors-pt",
        default="",
        help="persona_vectors.pt (default: <run-id>/vectors/persona_vectors.pt).",
    )
    p_val.add_argument(
        "--rollouts-jsonl",
        default="",
        help="rollouts.jsonl from step-c (default: <run-id>/rollouts/rollouts.jsonl).",
    )
    p_val.add_argument(
        "--sanity-json",
        default="",
        help="sanity_eval_projection.json (default: <run-id>/eval/sanity_eval_projection.json).",
    )
    p_val.add_argument(
        "--out",
        default="",
        help="Output JSON (default: <run-id>/eval/validation_report.json).",
    )
    p_val.add_argument(
        "--n-candidate-layers",
        type=int,
        default=8,
        help="Top-N layers (by norm) to test for layer selection (default: 8).",
    )
    p_val.add_argument(
        "--n-questions",
        type=int,
        default=3,
        help="Eval questions per gate (default: 3).",
    )
    p_val.add_argument(
        "--alphas",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated α for Gate 3 in order (default: integers 1–10).",
    )
    p_val.add_argument(
        "--sweep-coherence-stop",
        type=int,
        default=15,
        metavar="N",
        help="Stop Gate 3 after an α if mean coherence ≤ N (default: 15). Saves work past the cliff.",
    )
    p_val.add_argument(
        "--no-sweep-coherence-stop",
        action="store_true",
        help="Run every α in --alphas even if coherence has collapsed.",
    )
    p_val.add_argument(
        "--steering-replies-out",
        default="",
        metavar="PATH",
        help="Where to write Gate 2–3 steered replies JSON "
        "(default: <run-id>/eval/validate_steering_replies.json).",
    )
    p_val.add_argument(
        "--skip-model-gates",
        action="store_true",
        help="Only run Gates 0-1 (data + separation); skip layer selection and steering (no Gemma needed).",
    )
    p_val.add_argument(
        "--project",
        default="",
        help="GCP project for Vertex judge.",
    )
    p_val.add_argument(
        "--location",
        default="",
        help="Vertex region.",
    )
    p_val.add_argument(
        "--judge-model",
        default="",
        help="Vertex judge model id.",
    )
    p_val.add_argument(
        "--model-id",
        default="",
        help="HF model id for steering (default: GEMMA_MODEL_ID).",
    )
    p_val.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set PERSONA_FORCE_CPU=1.",
    )
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
