"""
Per-trait α sweep: mean trait + mean coherence vs α ≥ 0 along +v (same judges as validate Gate 3).

**Scale rule**: largest α in the sweep such that mean coherence (Vertex) across eval questions is still
≥ ``--coherence-floor`` (default 80). Only **positive** steering (+α·v) is evaluated.

Each ``per_question`` entry includes ``reply`` with the **full** steered assistant text (not truncated).
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

logger = logging.getLogger(__name__)


def _alpha_grid(
    step: float,
    max_alpha: float,
) -> list[float]:
    out: list[float] = []
    a = 0.0
    while a <= max_alpha + 1e-9:
        out.append(round(a, 6))
        a += step
    return out


_JUDGE_MAX_RETRIES = 2


def _call_trait_judge_with_retry(
    judge_instr: str,
    neg_sys: str,
    question: str,
    text: str,
    alpha: float,
    judge_kwargs: dict[str, Any],
) -> int:
    from app.persona.judge_vertex import score_transcript

    for attempt in range(_JUDGE_MAX_RETRIES + 1):
        try:
            js = score_transcript(judge_instr, neg_sys, question, text, **judge_kwargs)
            return int(js.score)
        except Exception as exc:
            if attempt < _JUDGE_MAX_RETRIES:
                logger.warning("Trait judge α=%s attempt %d failed, retrying: %s", alpha, attempt + 1, exc)
            else:
                logger.warning("Trait judge α=%s failed after %d attempts: %s", alpha, _JUDGE_MAX_RETRIES + 1, exc)
    return 0


def _call_coherence_judge_with_retry(
    text: str,
    alpha: float,
    judge_kwargs: dict[str, Any],
) -> int:
    from app.persona.quality_gates import score_coherence

    for attempt in range(_JUDGE_MAX_RETRIES + 1):
        try:
            return int(score_coherence(text, **judge_kwargs))
        except Exception as exc:
            if attempt < _JUDGE_MAX_RETRIES:
                logger.warning("Coherence judge α=%s attempt %d failed, retrying: %s", alpha, attempt + 1, exc)
            else:
                logger.warning("Coherence judge α=%s failed after %d attempts: %s", alpha, _JUDGE_MAX_RETRIES + 1, exc)
    return 0


def _sweep_positive_alphas(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    neg_sys: str,
    questions: list[str],
    layer_idx: int,
    direction: torch.Tensor,
    magnitudes: list[float],
    judge_instr: str,
    max_new_tokens: int,
    judge_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Sweep α ≥ 0 only: steering uses +α·direction."""
    from app.persona.quality_gates import _generate_steered

    rows: list[dict[str, Any]] = []
    for mag in magnitudes:
        alpha = float(mag)
        trait_scores: list[int] = []
        coh_scores: list[int] = []
        per_q: list[dict[str, Any]] = []
        for q in questions:
            text = _generate_steered(
                model,
                tokenizer,
                device,
                neg_sys,
                q,
                layer_idx=layer_idx,
                direction=direction,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
            )
            ts = _call_trait_judge_with_retry(judge_instr, neg_sys, q, text, alpha, judge_kwargs)
            trait_scores.append(ts)
            ch = _call_coherence_judge_with_retry(text, alpha, judge_kwargs)
            coh_scores.append(ch)
            per_q.append(
                {
                    "question": q,
                    "trait_score": ts,
                    "coherence_score": ch,
                    "reply": text,
                }
            )
        mt = sum(trait_scores) / len(trait_scores) if trait_scores else 0.0
        mc = sum(coh_scores) / len(coh_scores) if coh_scores else 0.0
        rows.append(
            {
                "alpha": alpha,
                "magnitude": mag,
                "mean_trait": round(mt, 2),
                "mean_coherence": round(mc, 2),
                "per_question": per_q,
            }
        )
        logger.info("α=%s → mean_trait=%.2f mean_coh=%.2f", alpha, mt, mc)

    return rows


def _max_magnitude_under_coherence(
    rows: list[dict[str, Any]],
    coherence_floor: float,
) -> float | None:
    """Rows must be sorted by increasing magnitude; each row has mean_coherence and magnitude."""
    best: float | None = None
    for r in rows:
        mag = float(r["magnitude"])
        mc = float(r["mean_coherence"])
        if mc >= coherence_floor:
            best = mag
    return best


def run_coherence_alpha_sweep_loaded(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    bundle_path: Path,
    vectors_pt: Path,
    layer_idx: int,
    coherence_floor: float,
    step: float,
    max_alpha: float,
    n_questions: int,
    max_new_tokens: int,
    judge_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Positive-α sweep only; uses an already-loaded Gemma model/tokenizer.
    """
    from app.persona.judge_vertex import judge_rubric_to_instructions
    from app.persona.response_style import with_paragraph_cap
    from app.persona.schemas import PersonaTraitArtifact

    jkw = dict(judge_kwargs or {})
    raw = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw)
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)
    questions = artifact.eval_questions[:n_questions]

    ck = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ck["v"].float()
    if not (0 <= layer_idx < v_full.shape[0]):
        raise ValueError(f"layer_idx={layer_idx} out of range for v shape {tuple(v_full.shape)}")

    dtype = next(model.parameters()).dtype
    direction = v_full[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)

    mags = _alpha_grid(step, max_alpha)

    sweep_rows = _sweep_positive_alphas(
        model=model,
        tokenizer=tokenizer,
        device=device,
        neg_sys=neg_sys,
        questions=questions,
        layer_idx=layer_idx,
        direction=direction,
        magnitudes=mags,
        judge_instr=judge_instr,
        max_new_tokens=max_new_tokens,
        judge_kwargs=jkw,
    )

    max_under = _max_magnitude_under_coherence(sweep_rows, coherence_floor)

    return {
        "trait_label": artifact.trait_label,
        "bundle_path": str(bundle_path.resolve()),
        "vectors_pt": str(vectors_pt.resolve()),
        "layer": layer_idx,
        "steering_direction": "+α·v (positive α only)",
        "coherence_floor": coherence_floor,
        "step": step,
        "max_alpha_swept": max_alpha,
        "n_questions": len(questions),
        "eval_questions_used": questions,
        "alpha_sweep": {
            "description": "α ≥ 0 along +v",
            "rows": sweep_rows,
            "max_magnitude_with_mean_coh_ge_floor": max_under,
        },
        "scale_recommended": max_under,
        "scale_rule": f"largest α with mean_coherence>={coherence_floor}",
    }


def run_coherence_alpha_sweep(
    *,
    bundle_path: Path,
    vectors_pt: Path,
    layer_idx: int,
    coherence_floor: float,
    step: float,
    max_alpha: float,
    n_questions: int,
    max_new_tokens: int,
    model_id: str | None,
    judge_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.persona.activations import load_model_and_tokenizer

    model, tokenizer, device = load_model_and_tokenizer(model_id, device=None)
    return run_coherence_alpha_sweep_loaded(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bundle_path=bundle_path,
        vectors_pt=vectors_pt,
        layer_idx=layer_idx,
        coherence_floor=coherence_floor,
        step=step,
        max_alpha=max_alpha,
        n_questions=n_questions,
        max_new_tokens=max_new_tokens,
        judge_kwargs=judge_kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Positive α sweep vs coherence (Vertex); recommend scale at coherence floor.",
    )
    p.add_argument(
        "--syc-bundle",
        type=Path,
        default=Path("persona_runs/sycophancy_iter1/artifacts/trait_bundle.json"),
    )
    p.add_argument(
        "--chaos-bundle",
        type=Path,
        default=Path("persona_runs/chaos_iter1/artifacts/trait_bundle.json"),
    )
    p.add_argument(
        "--syc-vectors",
        type=Path,
        default=Path("persona_runs/sycophancy_iter1/vectors/persona_vectors.pt"),
    )
    p.add_argument(
        "--chaos-vectors",
        type=Path,
        default=Path("persona_runs/chaos_iter1/vectors/persona_vectors.pt"),
    )
    p.add_argument("--syc-layer", type=int, default=29)
    p.add_argument("--chaos-layer", type=int, default=25)
    p.add_argument("--coherence-floor", type=float, default=80.0)
    p.add_argument("--step", type=float, default=0.1, help="α magnitude step (default 0.1).")
    p.add_argument("--max-alpha", type=float, default=4.0)
    p.add_argument("--n-questions", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument(
        "--model-id",
        default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("persona_runs/grid_coherence80_calibration.json"),
    )
    p.add_argument(
        "--trait-only",
        choices=("syc", "chaos", "both"),
        default="both",
    )
    args = p.parse_args(argv)

    jkw: dict[str, Any] = {}
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if pid:
        jkw["project_id"] = pid

    doc: dict[str, Any] = {
        "coherence_floor": args.coherence_floor,
        "step": args.step,
        "max_alpha": args.max_alpha,
    }

    if args.trait_only in ("syc", "both"):
        logger.info("Sweeping sycophancy (layer %s)...", args.syc_layer)
        doc["sycophancy"] = run_coherence_alpha_sweep(
            bundle_path=args.syc_bundle,
            vectors_pt=args.syc_vectors,
            layer_idx=args.syc_layer,
            coherence_floor=args.coherence_floor,
            step=args.step,
            max_alpha=args.max_alpha,
            n_questions=args.n_questions,
            max_new_tokens=args.max_new_tokens,
            model_id=args.model_id or None,
            judge_kwargs=jkw,
        )
    if args.trait_only in ("chaos", "both"):
        logger.info("Sweeping chaos (layer %s)...", args.chaos_layer)
        doc["chaos"] = run_coherence_alpha_sweep(
            bundle_path=args.chaos_bundle,
            vectors_pt=args.chaos_vectors,
            layer_idx=args.chaos_layer,
            coherence_floor=args.coherence_floor,
            step=args.step,
            max_alpha=args.max_alpha,
            n_questions=args.n_questions,
            max_new_tokens=args.max_new_tokens,
            model_id=args.model_id or None,
            judge_kwargs=jkw,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, indent=2))
    print(f"\nWrote {args.out_json.resolve()}", file=sys.stderr)

    ss = doc.get("sycophancy", {}).get("scale_recommended")
    cs = doc.get("chaos", {}).get("scale_recommended")
    if ss is not None and cs is not None:
        print(
            f"\nSuggested grid_nine: --scale-syc {ss:g} --scale-chaos {cs:g}",
            file=sys.stderr,
        )
    elif ss is not None:
        print(f"\nSuggested: --scale-syc {ss:g}", file=sys.stderr)
    elif cs is not None:
        print(f"\nSuggested: --scale-chaos {cs:g}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
