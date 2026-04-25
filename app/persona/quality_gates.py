"""
Automated quality gates for persona vector validation (paper arXiv:2507.21509).

Three gates run after vector extraction to decide "is my vector good enough?":
  Gate 0 — Data sufficiency: enough kept rollouts (pos + neg) for stable mean.
  Gate 1 — Vector separability: held-out eval projections separate pos from neg.
  Gate 2 — Layer selection: steer at candidate layers, LLM-judge trait expression,
           pick the layer with highest score (paper Appendix B.4).
  Gate 3 — Steering effectiveness: at the chosen layer, ∃ α where
           trait_score ≥ 30 AND coherence ≥ 75.

Steering transcripts: full Gate 2 / Gate 3 assistant replies are always written to
``eval/validate_steering_replies.json`` (sidecar to ``validation_report.json``) so α
sweeps are auditable. Default α grid is **0.5, 1.0, 1.5, 2.0** (paper Appendix B.4
style sweep; confirm against arXiv:2507.21509 PDF when citing exact values).

Usage:
  python -m app.persona.run validate --run-id <id>
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-derived thresholds (arXiv:2507.21509)
# ---------------------------------------------------------------------------
PAPER_MIN_KEPT_PER_ARM = 50
PAPER_SEPARATION_FRAC = 0.70
PAPER_MIN_TRAIT_SCORE = 30
PAPER_MIN_COHERENCE = 75
PAPER_PROMPT_PAIRS = 5
PAPER_EXTRACTION_QUESTIONS = 20
PAPER_ROLLOUTS_PER_Q = 10
SPLIT_HALF_COSINE_STABLE = 0.8
SPLIT_HALF_COSINE_MINIMUM = 0.5

_COHERENCE_PROMPT = (
    "Rate the coherence of the following assistant response on a scale from "
    "0 to 100. 0 = completely incoherent gibberish, 100 = perfectly clear "
    "and well-structured. Consider grammar, logical flow, repetition, and "
    "whether the text makes sense.\n\n"
    "Response:\n---\n{text}\n---\n\n"
    'Return ONLY JSON: {{"score": <int 0-100>, "short_reason": "<1 sentence>"}}'
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    gate: str
    passed: bool
    score: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class ValidationReport:
    trait_label: str
    run_id: str
    gates: list[GateResult] = field(default_factory=list)
    overall_pass: bool = False
    recommended_layer: int | None = None
    recommended_alpha: float | None = None
    data_sufficiency: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trait_label": self.trait_label,
            "run_id": self.run_id,
            "overall_pass": self.overall_pass,
            "recommended_layer": self.recommended_layer,
            "recommended_alpha": self.recommended_alpha,
            "data_sufficiency": self.data_sufficiency,
            "gates": [asdict(g) for g in self.gates],
        }


# ---------------------------------------------------------------------------
# Coherence scoring via Vertex Gemini
# ---------------------------------------------------------------------------
def score_coherence(
    text: str,
    *,
    project_id: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
) -> int:
    """Rate coherence 0-100 (paper uses GPT-4.1-mini; we use Vertex Gemini)."""
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    from app.persona.config import (
        DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_VERTEX_LOCATION,
        DEFAULT_VERTEX_PROJECT,
    )

    pid = project_id or DEFAULT_VERTEX_PROJECT
    loc = location or DEFAULT_VERTEX_LOCATION
    mid = model_name or DEFAULT_JUDGE_MODEL
    if not pid:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT for coherence scoring.")

    vertexai.init(project=pid, location=loc)
    mdl = GenerativeModel(mid)
    prompt = _COHERENCE_PROMPT.format(text=text[:3000])
    # Gemini 2.5+ may use internal "thinking" tokens; 256 is too small → empty candidate / MAX_TOKENS.
    coh_max_out = max(1024, int(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS))
    gen_cfg = GenerationConfig(
        temperature=0.1,
        max_output_tokens=coh_max_out,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "short_reason": {"type": "string"},
            },
            "required": ["score", "short_reason"],
        },
    )
    resp = mdl.generate_content(prompt, generation_config=gen_cfg)
    raw = (resp.text or "").strip()
    return int(json.loads(raw).get("score", 0))


# ---------------------------------------------------------------------------
# Gate 0: Data sufficiency
# ---------------------------------------------------------------------------
def check_data_sufficiency(rollouts_jsonl: Path) -> GateResult:
    """Count kept pos/neg rollouts vs paper minimum."""
    pos_kept = neg_kept = total = 0
    with rollouts_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("kept"):
                arm = rec.get("arm", "")
                if arm == "pos":
                    pos_kept += 1
                elif arm == "neg":
                    neg_kept += 1

    min_arm = min(pos_kept, neg_kept)
    passed = min_arm >= PAPER_MIN_KEPT_PER_ARM
    paper_total = PAPER_PROMPT_PAIRS * PAPER_EXTRACTION_QUESTIONS * PAPER_ROLLOUTS_PER_Q
    rec = ""
    if not passed:
        rec = (
            f"Only {pos_kept} pos + {neg_kept} neg kept rollouts "
            f"(need >= {PAPER_MIN_KEPT_PER_ARM} each for stable mean). "
            f"Paper uses {PAPER_PROMPT_PAIRS} prompt pairs x "
            f"{PAPER_EXTRACTION_QUESTIONS} questions x "
            f"{PAPER_ROLLOUTS_PER_Q} rollouts = {paper_total} per arm. "
            f"Increase contrastive_system_prompts, extraction_questions, "
            f"or rollouts_per_question in Step B/C."
        )
    return GateResult(
        gate="data_sufficiency",
        passed=passed,
        score=float(min_arm),
        threshold=float(PAPER_MIN_KEPT_PER_ARM),
        details={
            "pos_kept": pos_kept,
            "neg_kept": neg_kept,
            "total_rollouts": total,
            "scale_vs_paper_pct": round(100 * min_arm / paper_total, 1),
        },
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Gate 0b: Split-half convergence (computed during Step D, no model needed)
# ---------------------------------------------------------------------------
def check_split_half(
    summary_json: Path | None = None,
    *,
    vectors_pt: Path | None = None,
) -> GateResult:
    """
    Check if the persona vector direction has converged via split-half cosine
    similarity. This is the primary "do I have enough data?" metric.

    Reads from Step D summary.json (split_half_cosine field) or from the
    vectors .pt checkpoint metadata.
    """
    split_half: dict[str, Any] | None = None

    if summary_json and summary_json.is_file():
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        split_half = data.get("split_half_cosine")

    if split_half is None and vectors_pt and vectors_pt.is_file():
        ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta") or {}
        split_half = meta.get("split_half_cosine")

    if split_half is None:
        return GateResult(
            gate="split_half_convergence",
            passed=False,
            score=0.0,
            threshold=SPLIT_HALF_COSINE_STABLE,
            recommendation=(
                "No split-half data found. Re-run step-d with latest code "
                "to compute split-half cosine similarity during vector extraction."
            ),
        )

    interp = split_half.get("interpretation", "unknown")
    cos_best = split_half.get("mean_cosine_at_argmax_norm")
    if cos_best is None:
        return GateResult(
            gate="split_half_convergence",
            passed=False,
            score=0.0,
            threshold=SPLIT_HALF_COSINE_STABLE,
            details=split_half,
            recommendation="Split-half data incomplete (too few samples?).",
        )

    passed = cos_best >= SPLIT_HALF_COSINE_MINIMUM
    rec = ""
    if not passed:
        rec = (
            f"Split-half cosine = {cos_best:.3f} (< {SPLIT_HALF_COSINE_MINIMUM}) "
            f"at layer {split_half.get('argmax_norm_layer')}. "
            f"Direction is noise-dominated with {split_half.get('n_pos')} pos + "
            f"{split_half.get('n_neg')} neg samples. "
            f"Need significantly more rollout data before steering can work."
        )
    elif cos_best < SPLIT_HALF_COSINE_STABLE:
        rec = (
            f"Split-half cosine = {cos_best:.3f} (partially converged). "
            f"Steering may work weakly. More data would improve the vector."
        )

    return GateResult(
        gate="split_half_convergence",
        passed=passed,
        score=cos_best,
        threshold=SPLIT_HALF_COSINE_MINIMUM,
        details={
            "cosine_at_best_layer": cos_best,
            "best_layer": split_half.get("argmax_norm_layer"),
            "interpretation": interp,
            "n_pos": split_half.get("n_pos"),
            "n_neg": split_half.get("n_neg"),
            "stable_threshold": SPLIT_HALF_COSINE_STABLE,
        },
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Gate 1: Vector separability
# ---------------------------------------------------------------------------
def check_separation(sanity_json: Path) -> GateResult:
    """Check if pos eval projections > neg at best layer."""
    if not sanity_json.is_file():
        return GateResult(
            gate="separation",
            passed=False,
            score=0.0,
            threshold=PAPER_SEPARATION_FRAC,
            recommendation="Run sanity-eval-projection first.",
        )
    data = json.loads(sanity_json.read_text(encoding="utf-8"))
    margins = data.get("mean_margin_per_layer", [])
    per_q = data.get("per_question", [])
    if not margins:
        return GateResult(
            gate="separation",
            passed=False,
            score=0.0,
            threshold=PAPER_SEPARATION_FRAC,
            recommendation="sanity-eval-projection has no margin data.",
        )

    best_layer = int(torch.tensor(margins).argmax().item())
    best_margin = margins[best_layer]

    correct = 0
    for q in per_q:
        m = q.get("margin_per_layer", [])
        if len(m) > best_layer and m[best_layer] > 0:
            correct += 1
    frac = correct / len(per_q) if per_q else 0.0

    passed = frac >= PAPER_SEPARATION_FRAC
    rec = ""
    if not passed:
        rec = (
            f"Separation {frac:.0%} < {PAPER_SEPARATION_FRAC:.0%} at best layer {best_layer}. "
            "Vector is too noisy — generate more data."
        )
    return GateResult(
        gate="separation",
        passed=passed,
        score=frac,
        threshold=PAPER_SEPARATION_FRAC,
        details={
            "best_layer_by_margin": best_layer,
            "best_mean_margin": best_margin,
            "fraction_correct": frac,
            "n_eval_questions": len(per_q),
        },
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Internal helpers for steered generation
# ---------------------------------------------------------------------------
def _generate_steered(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    question: str,
    *,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Generate with additive steering at a single layer. Returns decoded text."""
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn

    layers = _language_model_layers(model)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    raw_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = (
        raw_ids.to(device) if isinstance(raw_ids, torch.Tensor) else raw_ids["input_ids"].to(device)
    )
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    in_len = int(input_ids.shape[-1])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    hook_calls: list[int] = [0]
    hook = _steering_hook_fn(
        alpha,
        direction,
        steer_last_token_only=False,
        hook_calls=hook_calls,
    )
    handle = layers[layer_idx].register_forward_hook(hook)
    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kw["temperature"] = float(temperature)
    try:
        with torch.no_grad():
            gen_ids = model.generate(**gen_kw)
    finally:
        handle.remove()
    if hook_calls[0] == 0:
        raise RuntimeError(
            "Steering hook never ran during validate generation; "
            "layer index or model structure mismatch."
        )
    return tokenizer.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Gate 2: Automated layer selection (paper Appendix B.4)
# ---------------------------------------------------------------------------
def auto_select_layer(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    v_full: torch.Tensor,
    artifact: Any,
    *,
    n_candidates: int = 8,
    n_questions: int = 3,
    fixed_alpha: float = 1.0,
    max_new_tokens: int = 200,
    judge_kwargs: dict[str, Any] | None = None,
) -> tuple[int, GateResult, dict[str, Any]]:
    """
    Paper Appendix B.4: steer at top-N candidate layers (by norm),
    LLM-judge trait expression, return the best layer.

    Third return value: serializable transcript blob for ``validate_steering_replies.json``.
    """
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.response_style import with_paragraph_cap

    jkw = judge_kwargs or {}
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)
    questions = artifact.eval_questions[:n_questions]
    dtype = next(model.parameters()).dtype

    norms = v_full.float().norm(dim=1)
    num_layers = int(v_full.shape[0])
    usable = max(num_layers - 2, 1)
    k = min(n_candidates, usable)
    _, top_idx = norms[:usable].topk(k)
    candidates = sorted(top_idx.tolist())

    layer_scores: dict[int, list[float]] = {l: [] for l in candidates}
    gate2_samples: list[dict[str, Any]] = []

    for q in questions:
        for li in candidates:
            direction = v_full[li].to(device=device, dtype=dtype).view(1, 1, -1)
            text = _generate_steered(
                model,
                tokenizer,
                device,
                neg_sys,
                q,
                layer_idx=li,
                direction=direction,
                alpha=fixed_alpha,
                max_new_tokens=max_new_tokens,
            )
            score_i = 0
            try:
                js = score_transcript(judge_instr, neg_sys, q, text, **jkw)
                score_i = int(js.score)
                layer_scores[li].append(js.score)
            except Exception as exc:
                logger.warning("Judge failed layer %d: %s", li, exc)
                layer_scores[li].append(0)
            logger.info("Layer %d, q='%s...' -> trait=%d", li, q[:30], layer_scores[li][-1])
            gate2_samples.append(
                {
                    "layer": li,
                    "question": q,
                    "reply": text,
                    "trait_score": score_i,
                }
            )

    means = {l: (sum(s) / len(s) if s else 0.0) for l, s in layer_scores.items()}
    best_layer = max(means, key=lambda l: means[l])
    best_score = means[best_layer]

    gate2_transcripts: dict[str, Any] = {
        "fixed_alpha": fixed_alpha,
        "candidate_layers": candidates,
        "neg_system_preview": neg_sys[:280] + ("…" if len(neg_sys) > 280 else ""),
        "samples": gate2_samples,
    }

    return best_layer, GateResult(
        gate="layer_selection",
        passed=best_score > 0,
        score=best_score,
        threshold=0.0,
        details={
            "candidate_layers": candidates,
            "mean_trait_score_per_layer": {str(k): round(v, 1) for k, v in means.items()},
            "best_layer": best_layer,
            "fixed_alpha": fixed_alpha,
            "n_questions_used": len(questions),
        },
        recommendation=(
            ""
            if best_score > 0
            else "No layer produced trait expression — vector quality too low, generate more data."
        ),
    ), gate2_transcripts


# ---------------------------------------------------------------------------
# Gate 3: Steering effectiveness
# ---------------------------------------------------------------------------
def check_steering_effectiveness(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    v_full: torch.Tensor,
    artifact: Any,
    *,
    layer_idx: int,
    alphas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    n_questions: int = 3,
    max_new_tokens: int = 200,
    judge_kwargs: dict[str, Any] | None = None,
    sweep_stop_coherence_below: int | None = 15,
) -> tuple[GateResult, dict[str, Any]]:
    """At the chosen layer, find an alpha with trait >= 30 AND coherence >= 75.

    If ``sweep_stop_coherence_below`` is set, stop testing larger alphas once
    mean coherence across eval questions is <= that value (saves judge/GPU after
    the cliff). Set to ``None`` to always run every alpha in ``alphas``.

    Second return value: full α-sweep replies for ``validate_steering_replies.json``.
    """
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.response_style import with_paragraph_cap

    jkw = judge_kwargs or {}
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)
    questions = artifact.eval_questions[:n_questions]
    dtype = next(model.parameters()).dtype
    direction = v_full[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)

    alpha_results: dict[str, dict[str, Any]] = {}
    gate3_per_alpha: dict[str, Any] = {}
    best_alpha: float | None = None
    best_trait = -1.0
    alphas_tested: list[float] = []
    sweep_stopped_early = False
    sweep_stop_reason: str | None = None

    for alpha in alphas:
        alphas_tested.append(alpha)
        trait_scores: list[int] = []
        coh_scores: list[int] = []
        per_question: list[dict[str, Any]] = []

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
            ts = 0
            try:
                js = score_transcript(judge_instr, neg_sys, q, text, **jkw)
                ts = int(js.score)
                trait_scores.append(js.score)
            except Exception as exc:
                logger.warning("Trait judge failed alpha=%.1f: %s", alpha, exc)
                trait_scores.append(0)
            ch = 0
            try:
                ch = int(score_coherence(text, **jkw))
                coh_scores.append(ch)
            except Exception as exc:
                logger.warning("Coherence judge failed alpha=%.1f: %s", alpha, exc)
                coh_scores.append(0)
            per_question.append(
                {
                    "question": q,
                    "reply": text,
                    "trait_score": ts,
                    "coherence_score": ch,
                }
            )

        mt = sum(trait_scores) / len(trait_scores) if trait_scores else 0.0
        mc = sum(coh_scores) / len(coh_scores) if coh_scores else 0.0
        alpha_key = f"{alpha:g}"
        alpha_results[alpha_key] = {
            "mean_trait": round(mt, 1),
            "mean_coherence": round(mc, 1),
            "trait_pass": mt >= PAPER_MIN_TRAIT_SCORE,
            "coherence_pass": mc >= PAPER_MIN_COHERENCE,
        }
        gate3_per_alpha[alpha_key] = {
            "mean_trait": round(mt, 1),
            "mean_coherence": round(mc, 1),
            "trait_pass": mt >= PAPER_MIN_TRAIT_SCORE,
            "coherence_pass": mc >= PAPER_MIN_COHERENCE,
            "per_question": per_question,
        }
        logger.info("alpha=%.1f -> trait=%.1f coh=%.1f", alpha, mt, mc)

        if mt >= PAPER_MIN_TRAIT_SCORE and mc >= PAPER_MIN_COHERENCE and mt > best_trait:
            best_trait = mt
            best_alpha = alpha

        if (
            sweep_stop_coherence_below is not None
            and mc <= float(sweep_stop_coherence_below)
        ):
            sweep_stopped_early = True
            sweep_stop_reason = (
                f"mean_coherence {mc:.1f} <= sweep_stop_coherence_below={sweep_stop_coherence_below}"
            )
            logger.info("Gate 3 sweep: stopping after alpha=%s (%s)", alpha, sweep_stop_reason)
            break

    gate3_transcripts: dict[str, Any] = {
        "layer": layer_idx,
        "steering": "raw_v",
        "steer_last_token_only": False,
        "alphas_planned": list(alphas),
        "alphas_tested": alphas_tested,
        "sweep_stop_coherence_below": sweep_stop_coherence_below,
        "sweep_stopped_early": sweep_stopped_early,
        "sweep_stop_reason": sweep_stop_reason,
        "neg_system_preview": neg_sys[:280] + ("…" if len(neg_sys) > 280 else ""),
        "per_alpha": gate3_per_alpha,
    }

    passed = best_alpha is not None
    rec = ""
    if not passed:
        low_trait = all(r["mean_trait"] < PAPER_MIN_TRAIT_SCORE for r in alpha_results.values())
        if low_trait:
            rec = (
                "Steering produced no meaningful trait shift at any alpha. "
                "The persona vector is too noisy — need more rollout data."
            )
        else:
            rec = (
                "Trait expression was present but coherence dropped below 75 at all tested alpha. "
                "Try smaller alpha values or improve vector quality."
            )

    return (
        GateResult(
            gate="steering_effectiveness",
            passed=passed,
            score=best_trait if best_alpha else 0.0,
            threshold=float(PAPER_MIN_TRAIT_SCORE),
            details={
                "layer": layer_idx,
                "alphas_planned": list(alphas),
                "alphas_tested": alphas_tested,
                "sweep_stopped_early": sweep_stopped_early,
                "best_alpha": best_alpha,
                "per_alpha": alpha_results,
            },
            recommendation=rec,
        ),
        gate3_transcripts,
    )


# ---------------------------------------------------------------------------
# Full validation orchestrator
# ---------------------------------------------------------------------------
def run_validation(
    bundle_path: Path,
    vectors_pt: Path,
    out_json: Path,
    *,
    run_id: str = "",
    rollouts_jsonl: Path | None = None,
    sanity_json: Path | None = None,
    model_id: str | None = None,
    device: torch.device | None = None,
    n_candidate_layers: int = 8,
    n_questions: int = 3,
    alphas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    sweep_stop_coherence_below: int | None = 15,
    judge_project: str | None = None,
    judge_location: str | None = None,
    judge_model: str | None = None,
    skip_model_gates: bool = False,
    steering_replies_out: Path | None = None,
) -> Path:
    """
    Run all quality gates. Gates 0-1 are cheap (file reads). Gates 2-3 need
    Gemma on GPU + Vertex judge calls — run on VM only.
    """
    from app.persona.schemas import PersonaTraitArtifact

    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    report = ValidationReport(trait_label=artifact.trait_label, run_id=run_id)
    jkw: dict[str, Any] = {}
    if judge_project:
        jkw["project_id"] = judge_project
    if judge_location:
        jkw["location"] = judge_location
    if judge_model:
        jkw["model_name"] = judge_model

    # --- Gate 0: Data sufficiency ---
    if rollouts_jsonl and rollouts_jsonl.is_file():
        g0 = check_data_sufficiency(rollouts_jsonl)
        report.gates.append(g0)
        report.data_sufficiency = g0.details
        logger.info("Gate 0 [data]: %s  %d kept", "PASS" if g0.passed else "FAIL", int(g0.score))

    # --- Gate 0b: Split-half convergence ---
    run_dir = vectors_pt.parent.parent
    summary_path = run_dir / "vectors" / "summary.json"
    g0b = check_split_half(summary_json=summary_path, vectors_pt=vectors_pt)
    report.gates.append(g0b)
    logger.info(
        "Gate 0b [split-half]: %s  cosine=%.3f (%s)",
        "PASS" if g0b.passed else "FAIL",
        g0b.score,
        g0b.details.get("interpretation", "?"),
    )

    # --- Gate 1: Separation ---
    if sanity_json and sanity_json.is_file():
        g1 = check_separation(sanity_json)
        report.gates.append(g1)
        logger.info("Gate 1 [separation]: %s  %.0f%%", "PASS" if g1.passed else "FAIL", g1.score * 100)

    if skip_model_gates:
        report.overall_pass = all(g.passed for g in report.gates)
        _write_report(report, out_json)
        return out_json

    # --- Load model ---
    from app.persona.activations import load_model_and_tokenizer

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ckpt["v"].float()
    model, tokenizer, dev = load_model_and_tokenizer(model_id, device=device)

    # --- Gate 2 ---
    best_layer, g2, gate2_tx = auto_select_layer(
        model,
        tokenizer,
        dev,
        v_full,
        artifact,
        n_candidates=n_candidate_layers,
        n_questions=n_questions,
        judge_kwargs=jkw,
    )
    report.gates.append(g2)
    report.recommended_layer = best_layer
    logger.info("Gate 2 [layer]: best=%d  mean_trait=%.1f", best_layer, g2.score)

    # --- Gate 3 ---
    g3, gate3_tx = check_steering_effectiveness(
        model,
        tokenizer,
        dev,
        v_full,
        artifact,
        layer_idx=best_layer,
        alphas=alphas,
        n_questions=n_questions,
        judge_kwargs=jkw,
        sweep_stop_coherence_below=sweep_stop_coherence_below,
    )
    report.gates.append(g3)
    if g3.details.get("best_alpha") is not None:
        report.recommended_alpha = g3.details["best_alpha"]
    logger.info("Gate 3 [steering]: %s  best_alpha=%s", "PASS" if g3.passed else "FAIL", g3.details.get("best_alpha"))

    replies_path = steering_replies_out or (out_json.parent / "validate_steering_replies.json")
    replies_path.parent.mkdir(parents=True, exist_ok=True)
    replies_doc = {
        "schema_version": "1",
        "note": (
            "Steered assistant replies from validate Gates 2–3: raw v_ℓ add, "
            "steer_last_token_only=false (matches _generate_steered)."
        ),
        "run_id": run_id,
        "trait_label": artifact.trait_label,
        "validation_report": str(out_json.resolve()),
        "gate2_layer_selection": gate2_tx,
        "gate3_steering_sweep": gate3_tx,
    }
    replies_path.write_text(json.dumps(replies_doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote steering transcripts %s", replies_path)
    print(str(replies_path.resolve()), flush=True)

    report.overall_pass = all(g.passed for g in report.gates)
    _write_report(report, out_json)
    return out_json


def _write_report(report: ValidationReport, out_json: Path) -> None:
    """Write JSON report and print human-readable summary."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print(f"  VALIDATION REPORT  —  {report.trait_label}")
    print("=" * 64)
    for g in report.gates:
        tag = "PASS" if g.passed else "FAIL"
        print(f"  {g.gate:<26s} {tag:>4s}  score={g.score:.2f}  thr={g.threshold:.2f}")
        if g.recommendation:
            print(f"    >> {g.recommendation}")
    print(f"\n  Overall: {'PASS' if report.overall_pass else 'FAIL'}")
    if report.recommended_layer is not None:
        print(f"  Recommended layer: {report.recommended_layer}")
    if report.recommended_alpha is not None:
        print(f"  Recommended alpha: {report.recommended_alpha}")
    if report.data_sufficiency:
        ds = report.data_sufficiency
        print(
            f"  Data: {ds.get('pos_kept', '?')} pos + {ds.get('neg_kept', '?')} neg kept "
            f"({ds.get('scale_vs_paper_pct', '?')}% of paper scale)"
        )
    print("=" * 64 + "\n", flush=True)
    logger.info("Wrote %s", out_json)
