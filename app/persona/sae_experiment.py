"""
SAE persona interpretation experiment CLI.

Subcommands:
  generate   — build sae/generations.json from rollouts + steered neg replies
  encode     — SAE-encode assistant spans -> sae/sae_latents.pt
  attribute  — signed feature attribution -> sae/feature_attribution.json
  autointerp — label top features via Neuronpedia + Vertex fallback
  validate   — sparse vs dense steering causal comparison
  recon      — SAE reconstruction fidelity + top-k sweep (Experiments A/B)
  ablate     — dense steer + SAE feature ablation necessity test (Experiment C)
  all        — run full pipeline (generate -> encode -> attribute -> autointerp -> validate)
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.persona.activations import (
    MODEL_ID,
    _hf_login,
    _pick_device,
    _pick_model_dtype,
    load_model_and_tokenizer,
)
from app.persona.config import PERSONA_RUNS_DIR
from app.persona.response_style import with_paragraph_cap
from app.persona.sae_common import load_rollout_question_pairs
from app.persona.sae_autointerp import run_autointerp
from app.persona.sae_encode import (
    build_sparse_direction,
    run_encode_latents,
    run_feature_attribution,
)
from app.persona.sae_causality import (
    magnitude_matched_alpha,
    run_ablation_validation,
    run_reconstruction_diagnostics,
)
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0]
DEFAULT_LAYER = 31
DEFAULT_STEERED_ALPHA = 2.0
DEFAULT_TOP_K = 10


def _sae_dir(run_dir: Path) -> Path:
    return run_dir / "sae"


def _load_dnd_layer(run_id: str, default: int = DEFAULT_LAYER) -> int:
    cfg_path = PERSONA_RUNS_DIR / "dnd_config.json"
    if not cfg_path.is_file():
        return default
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        key = run_id.replace("dnd_", "")
        block = raw.get("traits") if isinstance(raw.get("traits"), dict) else raw
        spec = block.get(key) if isinstance(block, dict) else None
        if isinstance(spec, dict) and spec.get("layer") is not None:
            return int(spec["layer"])
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return default



def _generate_steered_reply(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    neg_system: str,
    question: str,
    layer_idx: int,
    u: torch.Tensor,
    alpha: float,
    *,
    max_new_tokens: int = 256,
    steer_last_token_only: bool = False,
    rng_seed: int | None = None,
) -> str:
    dtype = next(model.parameters()).dtype
    messages = [
        {"role": "system", "content": neg_system},
        {"role": "user", "content": question},
    ]
    raw_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(raw_ids, torch.Tensor):
        input_ids = raw_ids.to(device)
    else:
        input_ids = raw_ids["input_ids"].to(device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    layers = _language_model_layers(model)
    direction = u.to(device=device, dtype=dtype).view(1, 1, -1)
    hook_calls = [0]
    hook = _steering_hook_fn(
        float(alpha),
        direction,
        steer_last_token_only=steer_last_token_only,
        hook_calls=hook_calls,
    )
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        if rng_seed is not None:
            torch.manual_seed(int(rng_seed))
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
    finally:
        handle.remove()

    if hook_calls[0] == 0:
        raise RuntimeError("Steering hook never ran during generation.")
    return tokenizer.decode(
        gen_ids[0, input_ids.shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def run_generate(
    run_dir: Path,
    out_json: Path,
    *,
    layer_idx: int,
    alphas: list[float],
    limit: int = 0,
    model_id: str | None = None,
    device: torch.device | None = None,
    max_new_tokens: int = 256,
) -> Path:
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    rollouts_path = run_dir / "rollouts" / "rollouts.jsonl"
    vectors_pt = run_dir / "vectors" / "persona_vectors.pt"

    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing bundle: {bundle_path}")
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"Missing rollouts: {rollouts_path}")
    if not vectors_pt.is_file():
        raise FileNotFoundError(f"Missing vectors: {vectors_pt}")

    pairs = load_rollout_question_pairs(rollouts_path, bundle_path)
    if limit and limit < len(pairs):
        pairs = pairs[:limit]

    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ckpt["v"].float()
    if not (0 <= layer_idx < v_full.shape[0]):
        raise ValueError(f"layer_idx {layer_idx} out of range for v shape {tuple(v_full.shape)}")
    direction = v_full[layer_idx]

    _hf_login()
    dev = device or _pick_device()
    mid = model_id or MODEL_ID
    logger.info("Loading %s on %s for steered generation…", mid, dev)
    tok = AutoTokenizer.from_pretrained(mid)
    dtype = _pick_model_dtype(dev)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(dev)
    model.eval()

    questions_out: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        logger.info("Generate steered %s/%s: %s", i + 1, len(pairs), pair["question"][:60])
        steered: list[dict[str, Any]] = []
        for j, alpha in enumerate(alphas):
            if alpha == 0.0:
                reply = pair["neg_reply"]
            else:
                reply = _generate_steered_reply(
                    model,
                    tok,
                    dev,
                    pair["neg_system"],
                    pair["question"],
                    layer_idx,
                    direction,
                    alpha,
                    max_new_tokens=max_new_tokens,
                    rng_seed=42 + i * 100 + j,
                )
            steered.append({"alpha": float(alpha), "reply": reply})
        questions_out.append({**pair, "steered": steered})

    doc = {
        "trait": artifact.trait_label,
        "run_id": run_dir.name,
        "layer": layer_idx,
        "alphas": [float(a) for a in alphas],
        "vectors_pt": str(vectors_pt.resolve()),
        "rollouts_jsonl": str(rollouts_path.resolve()),
        "model_id": mid,
        "n_questions": len(questions_out),
        "questions": questions_out,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def run_causal_validation(
    run_dir: Path,
    latents_pt: Path,
    attribution_json: Path,
    out_json: Path,
    *,
    layer_idx: int,
    steer_alpha: float = DEFAULT_STEERED_ALPHA,
    top_k: int = DEFAULT_TOP_K,
    limit: int = 0,
    model_id: str | None = None,
    sae_release: str | None = None,
    sae_id: str | None = None,
    device: torch.device | None = None,
    skip_judge: bool = False,
    project_id: str | None = None,
    normalize_sparse: bool = False,
    match_dense_magnitude: bool = True,
) -> Path:
    from app.phase2 import load_sae_for_layer
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.quality_gates import score_coherence

    gen_path = _sae_dir(run_dir) / "generations.json"
    if not gen_path.is_file():
        raise FileNotFoundError(f"Missing {gen_path}; run generate first.")
    gen = json.loads(gen_path.read_text(encoding="utf-8"))
    attr = json.loads(attribution_json.read_text(encoding="utf-8"))

    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    vectors_pt = run_dir / "vectors" / "persona_vectors.pt"
    artifact = PersonaTraitArtifact.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    neg_sys_default = with_paragraph_cap(artifact.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(artifact.judge_rubric)

    pos_feats = attr.get("top_positive_features") or []
    if not pos_feats:
        raise ValueError("No top_positive_features in attribution JSON.")
    selected = pos_feats[:top_k]
    feature_ids = [int(r["feature_id"]) for r in selected]
    coefficients = [float(r["mean_delta_steered"]) for r in selected]

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ckpt["v"].float()
    u_dense = v_full[layer_idx]

    model, tokenizer, dev = load_model_and_tokenizer(model_id, device=device)
    sae, sae_info = load_sae_for_layer(dev, release=sae_release, sae_id=sae_id)
    dtype = next(model.parameters()).dtype
    u_sparse_raw = build_sparse_direction(
        sae, feature_ids, coefficients, dev, dtype, normalize=normalize_sparse
    )
    sparse_alpha = float(steer_alpha)
    if match_dense_magnitude:
        sparse_alpha = magnitude_matched_alpha(
            u_dense,
            u_sparse_raw,
            steer_alpha,
            normalize_sparse=normalize_sparse,
        )
    u_sparse = u_sparse_raw

    cosine_dense_sparse = float(
        torch.dot(
            (u_dense / (u_dense.norm() + 1e-8)).cpu().float(),
            (u_sparse_raw / (u_sparse_raw.norm() + 1e-8)).cpu().float(),
        ).item()
    )

    questions = gen.get("questions") or []
    if limit and limit < len(questions):
        questions = questions[:limit]

    rows: list[dict[str, Any]] = []
    for i, qrow in enumerate(questions):
        q = qrow["question"]
        neg_sys = qrow.get("neg_system") or neg_sys_default
        logger.info("Causal validate %s/%s", i + 1, len(questions))

        baseline = _generate_steered_reply(
            model, tokenizer, dev, neg_sys, q, layer_idx, u_dense, 0.0
        )
        dense_reply = _generate_steered_reply(
            model, tokenizer, dev, neg_sys, q, layer_idx, u_dense, steer_alpha
        )
        sparse_reply = _generate_steered_reply(
            model, tokenizer, dev, neg_sys, q, layer_idx, u_sparse, sparse_alpha
        )

        row: dict[str, Any] = {
            "question_index": i,
            "question": q,
            "alpha": steer_alpha,
            "sparse_alpha": sparse_alpha,
            "feature_ids": feature_ids,
            "baseline_reply": baseline,
            "dense_steered_reply": dense_reply,
            "sparse_steered_reply": sparse_reply,
            "cosine_dense_sparse_direction": cosine_dense_sparse,
        }
        if not skip_judge:
            pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            for key, reply in (
                ("baseline", baseline),
                ("dense", dense_reply),
                ("sparse", sparse_reply),
            ):
                try:
                    js = score_transcript(
                        judge_instr, neg_sys, q, reply, project_id=pid
                    )
                    row[f"{key}_trait_score"] = int(js.score)
                    row[f"{key}_trait_reason"] = js.short_reason
                except (RuntimeError, json.JSONDecodeError) as e:
                    logger.warning("Judge failed for %s q%s: %s", key, i, e)
                    row[f"{key}_trait_score"] = -1
                    row[f"{key}_trait_reason"] = f"judge_error: {e}"
                try:
                    row[f"{key}_coherence"] = score_coherence(reply, project_id=pid)
                except (RuntimeError, json.JSONDecodeError, ValueError) as e:
                    logger.warning("Coherence failed for %s q%s: %s", key, i, e)
                    row[f"{key}_coherence"] = -1
        rows.append(row)

    doc = {
        "trait": gen.get("trait"),
        "run_id": run_dir.name,
        "layer": layer_idx,
        "steer_alpha": steer_alpha,
        "sparse_alpha": sparse_alpha,
        "normalize_sparse": normalize_sparse,
        "match_dense_magnitude": match_dense_magnitude,
        "cosine_dense_sparse_direction": cosine_dense_sparse,
        "top_k_features": top_k,
        "feature_ids": feature_ids,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "latents_pt": str(latents_pt.resolve()),
        "attribution_json": str(attribution_json.resolve()),
        "skip_judge": skip_judge,
        "comparisons": rows,
    }
    if rows and not skip_judge:
        dense_scores = [r["dense_trait_score"] for r in rows if r.get("dense_trait_score") is not None]
        sparse_scores = [r["sparse_trait_score"] for r in rows if r.get("sparse_trait_score") is not None]
        if dense_scores:
            doc["mean_dense_trait_score"] = sum(dense_scores) / len(dense_scores)
        if sparse_scores:
            doc["mean_sparse_trait_score"] = sum(sparse_scores) / len(sparse_scores)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def _parse_alphas(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAE persona vector interpretation experiment.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run-id", required=True, help="Run id under persona_runs/.")
        sp.add_argument("--force-cpu", action="store_true")
        sp.add_argument("--model-id", default="")

    p_gen = sub.add_parser("generate", help="Build sae/generations.json")
    add_common(p_gen)
    p_gen.add_argument("--layer", type=int, default=None)
    p_gen.add_argument(
        "--alphas",
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="Comma-separated steering alphas (default: 0,1,2,3,4).",
    )
    p_gen.add_argument("--limit", type=int, default=0)
    p_gen.add_argument("--out", default="")
    p_gen.add_argument("--max-new-tokens", type=int, default=256)

    p_enc = sub.add_parser("encode", help="SAE-encode assistant spans")
    add_common(p_enc)
    p_enc.add_argument("--layer", type=int, default=None)
    p_enc.add_argument("--generations", default="")
    p_enc.add_argument("--out-pt", default="")
    p_enc.add_argument("--sae-release", default="")
    p_enc.add_argument("--sae-id", default="")

    p_attr = sub.add_parser("attribute", help="Signed SAE feature attribution")
    p_attr.add_argument("--run-id", required=True)
    p_attr.add_argument("--latents-pt", default="")
    p_attr.add_argument("--out-json", default="")
    p_attr.add_argument("--steered-alpha", type=float, default=DEFAULT_STEERED_ALPHA)
    p_attr.add_argument("--top-k", type=int, default=20)

    p_ai = sub.add_parser("autointerp", help="Label top SAE features (Neuronpedia + Vertex)")
    p_ai.add_argument("--run-id", required=True)
    p_ai.add_argument("--attribution-json", default="")
    p_ai.add_argument("--out-json", default="")
    p_ai.add_argument("--top-k-positive", type=int, default=20)
    p_ai.add_argument("--top-k-negative", type=int, default=10)
    p_ai.add_argument("--neuronpedia-model", default="gemma-3-4b-it")
    p_ai.add_argument("--neuronpedia-source", default="", help="Override source set slug.")
    p_ai.add_argument("--explain-model", default="")
    p_ai.add_argument("--skip-vertex", action="store_true")
    p_ai.add_argument("--project", default="")

    p_val = sub.add_parser("validate", help="Sparse vs dense steering causal test")
    add_common(p_val)
    p_val.add_argument("--layer", type=int, default=None)
    p_val.add_argument("--latents-pt", default="")
    p_val.add_argument("--attribution-json", default="")
    p_val.add_argument("--out-json", default="")
    p_val.add_argument("--steer-alpha", type=float, default=DEFAULT_STEERED_ALPHA)
    p_val.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_val.add_argument("--limit", type=int, default=0)
    p_val.add_argument("--sae-release", default="")
    p_val.add_argument("--sae-id", default="")
    p_val.add_argument("--skip-judge", action="store_true")
    p_val.add_argument("--project", default="")
    p_val.add_argument(
        "--normalize-sparse",
        action="store_true",
        help="Unit-normalize sparse SAE direction before steering (legacy behavior).",
    )
    p_val.add_argument(
        "--no-match-dense-magnitude",
        action="store_true",
        help="Do not scale sparse alpha to match dense injection magnitude.",
    )

    p_recon = sub.add_parser(
        "recon",
        help="SAE reconstruction fidelity + top-k sweep (Experiments A/B)",
    )
    add_common(p_recon)
    p_recon.add_argument("--layer", type=int, default=None)
    p_recon.add_argument("--out-json", default="")
    p_recon.add_argument("--sae-release", default="")
    p_recon.add_argument("--sae-id", default="")
    p_recon.add_argument(
        "--topk-sweep",
        default="10,50,100,200,500,1000",
        help="Comma-separated k values for top-k cosine recovery.",
    )

    p_abl = sub.add_parser(
        "ablate",
        help="Dense steer + SAE feature ablation necessity test (Experiment C)",
    )
    add_common(p_abl)
    p_abl.add_argument("--layer", type=int, default=None)
    p_abl.add_argument("--attribution-json", default="")
    p_abl.add_argument("--out-json", default="")
    p_abl.add_argument("--steer-alpha", type=float, default=4.0)
    p_abl.add_argument("--top-k", type=int, default=50)
    p_abl.add_argument("--limit", type=int, default=0)
    p_abl.add_argument("--sae-release", default="")
    p_abl.add_argument("--sae-id", default="")
    p_abl.add_argument("--skip-judge", action="store_true")
    p_abl.add_argument("--project", default="")

    p_all = sub.add_parser("all", help="Run generate -> encode -> attribute -> validate")
    add_common(p_all)
    p_all.add_argument("--layer", type=int, default=None)
    p_all.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    p_all.add_argument("--limit", type=int, default=0)
    p_all.add_argument("--steered-alpha", type=float, default=DEFAULT_STEERED_ALPHA)
    p_all.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_all.add_argument("--skip-judge", action="store_true")
    p_all.add_argument("--skip-autointerp", action="store_true")
    p_all.add_argument("--project", default="")
    p_all.add_argument("--sae-release", default="")
    p_all.add_argument("--sae-id", default="")

    return p


def _run_dir(args: argparse.Namespace) -> Path:
    return (PERSONA_RUNS_DIR / args.run_id).resolve()


def _layer(args: argparse.Namespace) -> int:
    if getattr(args, "layer", None) is not None:
        return int(args.layer)
    return _load_dnd_layer(args.run_id)


def cmd_generate(args: argparse.Namespace) -> int:
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"
    run_dir = _run_dir(args)
    out = Path(args.out).resolve() if args.out else _sae_dir(run_dir) / "generations.json"
    run_generate(
        run_dir,
        out,
        layer_idx=_layer(args),
        alphas=_parse_alphas(args.alphas),
        limit=args.limit,
        model_id=args.model_id or None,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"
    run_dir = _run_dir(args)
    gen = (
        Path(args.generations).resolve()
        if args.generations
        else _sae_dir(run_dir) / "generations.json"
    )
    out_pt = Path(args.out_pt).resolve() if args.out_pt else _sae_dir(run_dir) / "sae_latents.pt"
    run_encode_latents(
        gen,
        out_pt,
        layer_idx=_layer(args),
        sae_release=args.sae_release or None,
        sae_id=args.sae_id or None,
        model_id=args.model_id or None,
    )
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args)
    latents = (
        Path(args.latents_pt).resolve()
        if args.latents_pt
        else _sae_dir(run_dir) / "sae_latents.pt"
    )
    out = (
        Path(args.out_json).resolve()
        if args.out_json
        else _sae_dir(run_dir) / "feature_attribution.json"
    )
    run_feature_attribution(
        latents,
        out,
        steered_alpha_key=f"{float(args.steered_alpha):g}",
        top_k=args.top_k,
    )
    return 0


def cmd_autointerp(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args)
    sae_dir = _sae_dir(run_dir)
    attr = (
        Path(args.attribution_json).resolve()
        if args.attribution_json
        else sae_dir / "feature_attribution.json"
    )
    out = (
        Path(args.out_json).resolve()
        if args.out_json
        else sae_dir / "feature_autointerp.json"
    )
    run_autointerp(
        attr,
        out,
        model_id=args.neuronpedia_model or "gemma-3-4b-it",
        top_k_positive=int(args.top_k_positive),
        top_k_negative=int(args.top_k_negative),
        use_vertex=not args.skip_vertex,
        project_id=args.project or None,
        explain_model=args.explain_model or None,
        neuronpedia_source=args.neuronpedia_source or None,
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"
    run_dir = _run_dir(args)
    sae_dir = _sae_dir(run_dir)
    latents = Path(args.latents_pt).resolve() if args.latents_pt else sae_dir / "sae_latents.pt"
    attr = (
        Path(args.attribution_json).resolve()
        if args.attribution_json
        else sae_dir / "feature_attribution.json"
    )
    out = Path(args.out_json).resolve() if args.out_json else sae_dir / "causal_validation.json"
    steer_alpha = getattr(args, "steer_alpha", getattr(args, "steered_alpha", DEFAULT_STEERED_ALPHA))
    run_causal_validation(
        run_dir,
        latents,
        attr,
        out,
        layer_idx=_layer(args),
        steer_alpha=steer_alpha,
        top_k=args.top_k,
        limit=args.limit,
        model_id=args.model_id or None,
        sae_release=args.sae_release or None,
        sae_id=args.sae_id or None,
        skip_judge=args.skip_judge,
        project_id=args.project or None,
        normalize_sparse=bool(args.normalize_sparse),
        match_dense_magnitude=not args.no_match_dense_magnitude,
    )
    return 0


def cmd_recon(args: argparse.Namespace) -> int:
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"
    run_dir = _run_dir(args)
    out = (
        Path(args.out_json).resolve()
        if args.out_json
        else _sae_dir(run_dir) / "reconstruction_diagnostics.json"
    )
    ks = [int(x.strip()) for x in args.topk_sweep.split(",") if x.strip()]
    run_reconstruction_diagnostics(
        run_dir,
        out,
        layer_idx=_layer(args),
        sae_release=args.sae_release or None,
        sae_id=args.sae_id or None,
        topk_sweep=ks,
    )
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"
    run_dir = _run_dir(args)
    sae_dir = _sae_dir(run_dir)
    attr = (
        Path(args.attribution_json).resolve()
        if args.attribution_json
        else sae_dir / "feature_attribution.json"
    )
    out = (
        Path(args.out_json).resolve()
        if args.out_json
        else sae_dir / "ablation_validation.json"
    )
    run_ablation_validation(
        run_dir,
        attr,
        out,
        layer_idx=_layer(args),
        steer_alpha=float(args.steer_alpha),
        top_k=int(args.top_k),
        limit=args.limit,
        model_id=args.model_id or None,
        sae_release=args.sae_release or None,
        sae_id=args.sae_id or None,
        skip_judge=args.skip_judge,
        project_id=args.project or None,
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    gen_args = argparse.Namespace(**vars(args))
    gen_args.cmd = "generate"
    gen_args.out = ""
    gen_args.max_new_tokens = 256
    if cmd_generate(gen_args) != 0:
        return 1

    enc_args = argparse.Namespace(**vars(args))
    enc_args.cmd = "encode"
    enc_args.generations = ""
    enc_args.out_pt = ""
    if cmd_encode(enc_args) != 0:
        return 1

    attr_args = argparse.Namespace(**vars(args))
    attr_args.cmd = "attribute"
    attr_args.latents_pt = ""
    attr_args.out_json = ""
    if cmd_attribute(attr_args) != 0:
        return 1

    if not getattr(args, "skip_autointerp", False):
        ai_args = argparse.Namespace(**vars(args))
        ai_args.cmd = "autointerp"
        ai_args.attribution_json = ""
        ai_args.out_json = ""
        ai_args.top_k_positive = int(getattr(args, "top_k", 20))
        ai_args.top_k_negative = 10
        ai_args.neuronpedia_model = "gemma-3-4b-it"
        ai_args.neuronpedia_source = ""
        ai_args.explain_model = ""
        ai_args.skip_vertex = False
        if cmd_autointerp(ai_args) != 0:
            return 1

    val_args = argparse.Namespace(**vars(args))
    val_args.cmd = "validate"
    val_args.latents_pt = ""
    val_args.attribution_json = ""
    val_args.out_json = ""
    return cmd_validate(val_args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "generate": cmd_generate,
        "encode": cmd_encode,
        "attribute": cmd_attribute,
        "autointerp": cmd_autointerp,
        "validate": cmd_validate,
        "recon": cmd_recon,
        "ablate": cmd_ablate,
        "all": cmd_all,
    }
    return int(handlers[args.cmd](args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
