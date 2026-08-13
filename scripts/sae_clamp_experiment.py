#!/usr/bin/env python3
"""
SAE encode-modify-decode clamping experiments (Options C, A, B).

Option C: Verify infrastructure with a French-language feature (positive control).
Option A: Single-feature clamping for Good persona.
Option B: Multi-feature clamping sweep for Good persona.

Usage (on VM):
  cd ~/gemma-chat && PYTHONPATH=$HOME/gemma-chat GOOGLE_CLOUD_PROJECT=applied-ai-practice00 \\
    .venv/bin/python3 -u scripts/sae_clamp_experiment.py --phase all --run-id dnd_good_scale
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from app.persona.activations import _pick_device, load_model_and_tokenizer
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.quality_gates import score_coherence
from app.persona.response_style import with_paragraph_cap
from app.persona.sae_autointerp import (
    DEFAULT_NEURONPEDIA_MODEL,
    explanation_from_neuronpedia,
    fetch_neuronpedia_feature,
    neuronpedia_source_set,
)
from app.persona.sae_causality import sae_feature_clamp_hook_fn
from app.persona.sae_common import compute_sta_attribution
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers
from app.phase2 import load_sae_for_layer

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

PERSONA_RUNS = Path(os.environ.get("PERSONA_RUNS", Path.home() / "gemma-chat/persona_runs"))
DEFAULT_LAYER = 16
DEFAULT_SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
DEFAULT_SAE_ID = "layer_16_width_16k_l0_small"
DEFAULT_STEER_ALPHA = 1.5

FRENCH_SAMPLES = [
    "Bonjour, comment allez-vous aujourd'hui?",
    "La capitale de la France est Paris.",
    "J'aime lire des livres en français.",
    "Le chat dort sur le canapé.",
    "Nous allons au marché demain matin.",
    "C'est une belle journée ensoleillée.",
    "Je parle français avec mes amis.",
    "La cuisine française est délicieuse.",
]

ENGLISH_SAMPLES = [
    "Hello, how are you today?",
    "The capital of France is Paris.",
    "I enjoy reading books in English.",
    "The cat sleeps on the couch.",
    "We are going to the market tomorrow morning.",
    "It is a beautiful sunny day.",
    "I speak English with my friends.",
    "French cuisine is delicious.",
]

NEUTRAL_PROMPTS = [
    "Tell me about your day.",
    "What is the capital of Germany?",
    "Describe your favorite hobby.",
    "Give me advice on staying healthy.",
    "What do you think about learning new languages?",
]

FRENCH_MARKERS = {
    "le", "la", "les", "de", "du", "des", "est", "je", "vous", "bonjour",
    "merci", "une", "un", "à", "pour", "nous", "c'est", "qui", "dans",
    "avec", "pas", "que", "sur", "en", "au", "ce", "il", "elle", "sont",
    "très", "bien", "mais", "aussi", "comme", "mon", "ton", "notre",
}


def is_predominantly_french(text: str, *, min_words: int = 8, threshold: float = 0.08) -> bool:
    words = re.findall(r"\b[\w']+\b", text.lower())
    if len(words) < min_words:
        return False
    hits = sum(1 for w in words if w in FRENCH_MARKERS)
    return (hits / len(words)) >= threshold


def _sae_dir(run_dir: Path) -> Path:
    return run_dir / "sae"


def _apply_chat(tokenizer, system: str, user: str, device: torch.device) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(raw_ids, torch.Tensor):
        return raw_ids.to(device)
    return raw_ids["input_ids"].to(device)


def _encode_last_token_activations(
    model,
    tokenizer,
    device: torch.device,
    sae,
    layer_idx: int,
    texts: list[str],
    *,
    system: str = "You are a helpful assistant.",
) -> torch.Tensor:
    """Return mean SAE latent vector (d_sae,) from last-token activations."""
    layers = _language_model_layers(model)
    sae_dev = next(sae.parameters()).device
    latents: list[torch.Tensor] = []

    for text in texts:
        input_ids = _apply_chat(tokenizer, system, text, device)
        captured: list[torch.Tensor] = []

        def capture_hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                captured.append(h[:, -1:, :].detach().clone())
            return output

        handle = layers[layer_idx].register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"Failed to capture activations for: {text[:40]}")
        h = captured[0].to(sae_dev)
        with torch.no_grad():
            z = sae.encode(h)[0, 0].float().cpu()
        latents.append(z)

    return torch.stack(latents, dim=0).mean(dim=0)


def _neuronpedia_lookup(
    feature_id: int,
    *,
    sae_release: str,
    sae_id: str,
    layer_idx: int,
) -> dict[str, Any]:
    source_set = neuronpedia_source_set(sae_release, sae_id)
    alt_source = f"{layer_idx + 1}-gemmascope-2-res-16k"
    result: dict[str, Any] = {
        "feature_id": feature_id,
        "source_set": source_set,
        "explanation": None,
        "neuronpedia_url": None,
    }
    for src in [source_set, alt_source]:
        if not src:
            continue
        doc = fetch_neuronpedia_feature(DEFAULT_NEURONPEDIA_MODEL, src, feature_id)
        if doc:
            result["source_set"] = src
            result["explanation"] = explanation_from_neuronpedia(doc)
            result["neuronpedia_url"] = (
                f"https://www.neuronpedia.org/{DEFAULT_NEURONPEDIA_MODEL}/{src}/{feature_id}"
            )
            pos = [str(t).strip() for t in (doc.get("pos_str") or [])[:8] if str(t).strip()]
            result["pos_tokens"] = pos
            break
    return result


FRENCH_LABEL_KEYWORDS = (
    "french", "français", "france", "foreign words", "foreign word",
    "multilingual", "language", "translation",
)
FRENCH_POS_KEYWORDS = (
    "amour", "bonjour", "français", "francais", "merci", "ét", "vous", "nous",
)


def _label_matches_french_concept(expl: str, pos: str) -> tuple[bool, float]:
    text = (expl + " " + pos).lower()
    score = 0.0
    if "french" in text or "français" in text or "france" in text:
        score += 10.0
    for kw in FRENCH_LABEL_KEYWORDS:
        if kw in text:
            score += 3.0
    for kw in FRENCH_POS_KEYWORDS:
        if kw in text:
            score += 2.0
    return score > 0.0, score


def find_french_feature(
    model,
    tokenizer,
    device: torch.device,
    sae,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    top_n: int = 300,
    french_feature_id: int | None = None,
) -> dict[str, Any]:
    logger.info("Finding French-differential features at layer %s…", layer_idx)
    z_fr = _encode_last_token_activations(
        model, tokenizer, device, sae, layer_idx, FRENCH_SAMPLES
    )
    z_en = _encode_last_token_activations(
        model, tokenizer, device, sae, layer_idx, ENGLISH_SAMPLES
    )
    delta = z_fr - z_en

    if french_feature_id is not None:
        fid = int(french_feature_id)
        selected = {
            "feature_id": fid,
            "mean_fr_activation": float(z_fr[fid].item()),
            "mean_en_activation": float(z_en[fid].item()),
            "delta_fr_minus_en": float(delta[fid].item()),
            "label_score": 0.0,
            "label_match": False,
            "neuronpedia": _neuronpedia_lookup(
                fid, sae_release=sae_release, sae_id=sae_id, layer_idx=layer_idx
            ),
        }
        clamp_p95 = max(float(z_fr[fid].item()) * 2.0, 50.0)
        logger.info(
            "Using override French feature %s (fr_act=%.3f, expl=%s)",
            fid,
            selected["mean_fr_activation"],
            (selected["neuronpedia"].get("explanation") or "none")[:80],
        )
        return {
            "layer": layer_idx,
            "sae_release": sae_release,
            "sae_id": sae_id,
            "selected_feature": selected,
            "labeled_candidates": [],
            "top_fr_activation_candidates": [],
            "suggested_clamp_p95": clamp_p95,
            "override_feature_id": fid,
        }

    fr_vals, fr_idx = torch.topk(z_fr, k=min(top_n, z_fr.numel()))
    delta_vals, delta_idx = torch.topk(delta, k=min(top_n, delta.numel()))

    # Limit Neuronpedia lookups: prioritize top French activations, then top deltas.
    lookup_ids: list[int] = []
    for fid in fr_idx.tolist()[:80]:
        lookup_ids.append(int(fid))
    for fid in delta_idx.tolist()[:30]:
        if int(fid) not in lookup_ids:
            lookup_ids.append(int(fid))

    candidates: list[dict[str, Any]] = []
    for fid in lookup_ids:
        np_info = _neuronpedia_lookup(
            int(fid), sae_release=sae_release, sae_id=sae_id, layer_idx=layer_idx
        )
        expl = np_info.get("explanation") or ""
        pos = " ".join(np_info.get("pos_tokens") or [])
        matches, label_score = _label_matches_french_concept(expl, pos)
        candidates.append(
            {
                "feature_id": int(fid),
                "mean_fr_activation": float(z_fr[fid].item()),
                "mean_en_activation": float(z_en[fid].item()),
                "delta_fr_minus_en": float(delta[fid].item()),
                "label_score": label_score,
                "label_match": matches,
                "neuronpedia": np_info,
            }
        )

    labeled = [c for c in candidates if c["label_match"]]
    labeled.sort(
        key=lambda c: (c["label_score"], c["mean_fr_activation"]),
        reverse=True,
    )

    if french_feature_id is not None:
        selected = next(
            (c for c in candidates if c["feature_id"] == french_feature_id),
            None,
        )
        if selected is None:
            selected = {
                "feature_id": french_feature_id,
                "mean_fr_activation": float(z_fr[french_feature_id].item()),
                "mean_en_activation": float(z_en[french_feature_id].item()),
                "delta_fr_minus_en": float(delta[french_feature_id].item()),
                "label_score": 0.0,
                "label_match": False,
                "neuronpedia": _neuronpedia_lookup(
                    french_feature_id,
                    sae_release=sae_release,
                    sae_id=sae_id,
                    layer_idx=layer_idx,
                ),
            }
    elif labeled:
        selected = labeled[0]
    else:
        # Fallback: highest French activation among scanned candidates
        candidates.sort(key=lambda c: c["mean_fr_activation"], reverse=True)
        selected = candidates[0]

    clamp_p95 = _percentile(
        [float(z_fr[int(fid)].item()) for fid in fr_idx.tolist()[:50]],
        95.0,
    )
    logger.info(
        "Selected French feature %s (fr_act=%.3f, label=%s, expl=%s)",
        selected["feature_id"],
        selected["mean_fr_activation"],
        selected.get("label_match"),
        (selected["neuronpedia"].get("explanation") or "none")[:80],
    )
    return {
        "layer": layer_idx,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "selected_feature": selected,
        "labeled_candidates": labeled[:10],
        "top_fr_activation_candidates": sorted(
            candidates, key=lambda c: c["mean_fr_activation"], reverse=True
        )[:10],
        "suggested_clamp_p95": clamp_p95,
    }


def _generate_with_clamp(
    model,
    tokenizer,
    device: torch.device,
    sae,
    layer_idx: int,
    system: str,
    question: str,
    feature_ids: list[int],
    clamp_values: list[float],
    *,
    mode: str = "additive_delta",
    steer_last_token_only: bool = False,
    max_new_tokens: int = 256,
) -> tuple[str, int]:
    input_ids = _apply_chat(tokenizer, system, question, device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    layers = _language_model_layers(model)
    hook_calls = [0]
    hook = sae_feature_clamp_hook_fn(
        sae,
        feature_ids,
        clamp_values,
        hook_calls,
        mode=mode,
        steer_last_token_only=steer_last_token_only,
    )
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
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
        raise RuntimeError("Clamp hook never ran during generation.")
    reply = tokenizer.decode(
        gen_ids[0, input_ids.shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return reply, hook_calls[0]


def _generate_baseline(
    model,
    tokenizer,
    device: torch.device,
    system: str,
    question: str,
    *,
    max_new_tokens: int = 256,
) -> str:
    input_ids = _apply_chat(tokenizer, system, question, device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
        )
    return tokenizer.decode(
        gen_ids[0, input_ids.shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    k = max(0, min(len(values) - 1, int(round(pct / 100.0 * (len(values) - 1)))))
    return float(torch.sort(t)[0][k].item())


def _resolve_latents_pt(run_dir: Path, latents_pt: Path | None) -> Path:
    if latents_pt is not None:
        p = latents_pt.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing latents: {p}")
        return p
    sae_dir = _sae_dir(run_dir)
    candidates = [
        sae_dir / "sae_latents_l16_v2.pt",
        sae_dir / "sae_latents_l16.pt",
        sae_dir / "latents.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No latents file found under {sae_dir}; pass --latents-pt explicitly."
    )


def _load_good_feature_clamp_values(
    run_dir: Path,
    feature_ids: list[int],
    *,
    steered_alpha_key: str = "1.5",
    latents_pt: Path | None = None,
) -> dict[int, float]:
    latents_path = _resolve_latents_pt(run_dir, latents_pt)
    ckpt = torch.load(latents_path, map_location="cpu", weights_only=False)
    questions = ckpt.get("questions") or []
    if not questions:
        raise ValueError("No questions in latents.pt")

    clamp_values: dict[int, float] = {}
    for fid in feature_ids:
        acts: list[float] = []
        for qd in questions:
            z_st = qd.get("z_steered", {}).get(steered_alpha_key)
            if z_st is None:
                continue
            acts.append(float(z_st[fid].item()))
        if not acts:
            raise ValueError(
                f"No steered activations for feature {fid} at alpha={steered_alpha_key}"
            )
        clamp_values[fid] = _percentile(acts, 95.0)
    return clamp_values


def _select_good_features(
    run_dir: Path,
    *,
    top_k: int,
    steered_alpha_key: str,
    latents_pt: Path | None = None,
) -> list[dict[str, Any]]:
    latents_path = _resolve_latents_pt(run_dir, latents_pt)
    ckpt = torch.load(latents_path, map_location="cpu", weights_only=False)
    questions = ckpt.get("questions") or []
    sta = compute_sta_attribution(
        questions,
        steered_alpha_key=steered_alpha_key,
        amplitude_threshold=0.0,
        frequency_threshold=0.0,
        top_k=top_k,
    )
    atoms = sta.get("sta_positive_atoms") or []
    if not atoms:
        raise ValueError("No positive STA atoms found in latents.")
    return atoms[:top_k]


def _score_rows(
    rows: list[dict[str, Any]],
    judge_instr: str,
    neg_sys: str,
    *,
    skip_judge: bool,
    project_id: str | None,
) -> None:
    if skip_judge:
        return
    pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    for row in rows:
        q = row["question"]
        for key in ("baseline", "clamped", "dense"):
            reply = row.get(f"{key}_reply")
            if not reply:
                continue
            try:
                js = score_transcript(judge_instr, neg_sys, q, reply, project_id=pid)
                row[f"{key}_trait_score"] = int(js.score)
                row[f"{key}_trait_reason"] = js.short_reason
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.warning("Judge failed %s: %s", key, e)
                row[f"{key}_trait_score"] = -1
            try:
                row[f"{key}_coherence"] = score_coherence(reply, project_id=pid)
            except (RuntimeError, json.JSONDecodeError, ValueError) as e:
                logger.warning("Coherence failed %s: %s", key, e)
                row[f"{key}_coherence"] = -1


def _mean_score(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key, -1) >= 0]
    return sum(vals) / len(vals) if vals else None


def run_phase_c(
    model,
    tokenizer,
    device: torch.device,
    sae,
    out_dir: Path,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    clamp_mode: str,
    french_feature_id: int | None = None,
    steer_last_token_only: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    discovery = find_french_feature(
        model,
        tokenizer,
        device,
        sae,
        layer_idx=layer_idx,
        sae_release=sae_release,
        sae_id=sae_id,
        french_feature_id=french_feature_id,
    )
    fid = int(discovery["selected_feature"]["feature_id"])
    p95 = float(discovery.get("suggested_clamp_p95") or 10.0)
    strengths = sorted(
        {
            max(1.0, p95 * 0.25),
            max(2.0, p95 * 0.5),
            max(5.0, p95 * 0.75),
            p95,
            p95 * 1.5,
            p95 * 2.0,
        }
    )

    results: list[dict[str, Any]] = []
    pass_at_strength: float | None = None
    for prompt in NEUTRAL_PROMPTS:
        baseline = _generate_baseline(model, tokenizer, device, "You are a helpful assistant.", prompt)
        row: dict[str, Any] = {
            "prompt": prompt,
            "baseline_reply": baseline,
            "baseline_french": is_predominantly_french(baseline),
            "strength_sweep": [],
        }
        for strength in strengths:
            reply, n_hooks = _generate_with_clamp(
                model,
                tokenizer,
                device,
                sae,
                layer_idx,
                "You are a helpful assistant.",
                prompt,
                [fid],
                [strength],
                mode=clamp_mode,
                steer_last_token_only=steer_last_token_only,
            )
            french = is_predominantly_french(reply)
            row["strength_sweep"].append(
                {
                    "strength": strength,
                    "reply": reply,
                    "french_detected": french,
                    "hook_calls": n_hooks,
                }
            )
            if french and pass_at_strength is None and strength >= 10.0:
                pass_at_strength = strength
        results.append(row)
        logger.info(
            "Phase C prompt done: baseline_fr=%s best_fr=%s",
            row["baseline_french"],
            any(s["french_detected"] for s in row["strength_sweep"]),
        )

    any_french_at_10 = any(
        any(s["french_detected"] for s in r["strength_sweep"])
        for r in results
    )
    doc = {
        "phase": "C",
        "layer": layer_idx,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "clamp_mode": clamp_mode,
        "steer_last_token_only": steer_last_token_only,
        "feature_discovery": discovery,
        "feature_id": fid,
        "suggested_clamp_p95": p95,
        "strengths_tested": strengths,
        "prompts": results,
        "pass_criterion_met": any_french_at_10,
        "pass_at_strength": pass_at_strength,
    }
    out_path = out_dir / "phase_c_french_clamp.json"
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (pass=%s)", out_path, any_french_at_10)
    return doc


def run_phase_a(
    model,
    tokenizer,
    device: torch.device,
    sae,
    run_dir: Path,
    out_dir: Path,
    *,
    layer_idx: int,
    sae_release: str,
    sae_id: str,
    steer_alpha: float,
    steered_alpha_key: str,
    clamp_mode: str,
    limit: int,
    skip_judge: bool,
    project_id: str | None,
    latents_pt: Path | None,
    steer_last_token_only: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts" / "trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(bundle.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )

    atoms = _select_good_features(
        run_dir, top_k=1, steered_alpha_key=steered_alpha_key, latents_pt=latents_pt
    )
    fid = int(atoms[0]["feature_id"])
    clamp_val = _load_good_feature_clamp_values(
        run_dir, [fid], steered_alpha_key=steered_alpha_key, latents_pt=latents_pt
    )[fid]

    vectors_pt = run_dir / "vectors" / "persona_vectors.pt"
    u_dense = torch.load(vectors_pt, map_location="cpu", weights_only=False)["v"].float()[layer_idx]

    from app.persona.sae_experiment import _generate_steered_reply

    questions = bundle.eval_questions or []
    if limit and limit < len(questions):
        questions = questions[:limit]

    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        logger.info("Phase A %s/%s", i + 1, len(questions))
        baseline = _generate_baseline(model, tokenizer, device, neg_sys, q)
        clamped, _ = _generate_with_clamp(
            model,
            tokenizer,
            device,
            sae,
            layer_idx,
            neg_sys,
            q,
            [fid],
            [clamp_val],
            mode=clamp_mode,
            steer_last_token_only=steer_last_token_only,
        )
        dense = _generate_steered_reply(
            model, tokenizer, device, neg_sys, q, layer_idx, u_dense, steer_alpha
        )
        rows.append(
            {
                "question_index": i,
                "question": q,
                "feature_id": fid,
                "clamp_value": clamp_val,
                "baseline_reply": baseline,
                "clamped_reply": clamped,
                "dense_reply": dense,
            }
        )

    _score_rows(rows, judge_instr, neg_sys, skip_judge=skip_judge, project_id=project_id)

    mean_baseline = _mean_score(rows, "baseline_trait_score")
    mean_clamped = _mean_score(rows, "clamped_trait_score")
    mean_dense = _mean_score(rows, "dense_trait_score")
    recovery = None
    if mean_baseline is not None and mean_dense is not None and mean_clamped is not None:
        denom = mean_dense - mean_baseline
        recovery = ((mean_clamped - mean_baseline) / denom * 100.0) if abs(denom) > 1e-6 else None

    doc = {
        "phase": "A",
        "run_id": run_dir.name,
        "layer": layer_idx,
        "feature_id": fid,
        "clamp_value": clamp_val,
        "clamp_mode": clamp_mode,
        "steer_alpha": steer_alpha,
        "n_questions": len(rows),
        "comparisons": rows,
        "mean_baseline_trait_score": mean_baseline,
        "mean_clamped_trait_score": mean_clamped,
        "mean_dense_trait_score": mean_dense,
        "trait_recovery_pct": recovery,
    }
    out_path = out_dir / "phase_a_single_clamp.json"
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (recovery=%s%%)", out_path, recovery)
    return doc


def run_phase_b(
    model,
    tokenizer,
    device: torch.device,
    sae,
    run_dir: Path,
    out_dir: Path,
    *,
    layer_idx: int,
    steer_alpha: float,
    steered_alpha_key: str,
    clamp_mode: str,
    k_values: list[int],
    limit: int,
    skip_judge: bool,
    project_id: str | None,
    latents_pt: Path | None,
    steer_last_token_only: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts" / "trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(bundle.neg_system_prompt)
    judge_instr = judge_rubric_to_instructions(
        bundle.judge_rubric, trait_label=bundle.trait_label
    )

    max_k = max(k_values)
    atoms = _select_good_features(
        run_dir, top_k=max_k, steered_alpha_key=steered_alpha_key, latents_pt=latents_pt
    )

    vectors_pt = run_dir / "vectors" / "persona_vectors.pt"
    u_dense = torch.load(vectors_pt, map_location="cpu", weights_only=False)["v"].float()[layer_idx]
    from app.persona.sae_experiment import _generate_steered_reply

    questions = bundle.eval_questions or []
    if limit and limit < len(questions):
        questions = questions[:limit]

    # Dense baseline (once)
    dense_scores: list[float] = []
    baseline_scores: list[float] = []

    sweep_results: list[dict[str, Any]] = []
    for k in k_values:
        selected = atoms[:k]
        fids = [int(a["feature_id"]) for a in selected]
        clamp_map = _load_good_feature_clamp_values(
            run_dir, fids, steered_alpha_key=steered_alpha_key, latents_pt=latents_pt
        )
        clamp_vals = [clamp_map[fid] for fid in fids]

        rows: list[dict[str, Any]] = []
        for i, q in enumerate(questions):
            logger.info("Phase B k=%s q=%s/%s", k, i + 1, len(questions))
            baseline = _generate_baseline(model, tokenizer, device, neg_sys, q)
            clamped, _ = _generate_with_clamp(
                model,
                tokenizer,
                device,
                sae,
                layer_idx,
                neg_sys,
                q,
                fids,
                clamp_vals,
                mode=clamp_mode,
                steer_last_token_only=steer_last_token_only,
            )
            dense = _generate_steered_reply(
                model, tokenizer, device, neg_sys, q, layer_idx, u_dense, steer_alpha
            )
            rows.append(
                {
                    "question_index": i,
                    "question": q,
                    "baseline_reply": baseline,
                    "clamped_reply": clamped,
                    "dense_reply": dense,
                }
            )

        _score_rows(rows, judge_instr, neg_sys, skip_judge=skip_judge, project_id=project_id)
        mean_baseline = _mean_score(rows, "baseline_trait_score")
        mean_clamped = _mean_score(rows, "clamped_trait_score")
        mean_dense = _mean_score(rows, "dense_trait_score")
        recovery = None
        if mean_baseline is not None and mean_dense is not None and mean_clamped is not None:
            denom = mean_dense - mean_baseline
            recovery = (
                (mean_clamped - mean_baseline) / denom * 100.0 if abs(denom) > 1e-6 else None
            )

        sweep_results.append(
            {
                "k": k,
                "feature_ids": fids,
                "clamp_values": {str(fid): clamp_map[fid] for fid in fids},
                "mean_baseline_trait_score": mean_baseline,
                "mean_clamped_trait_score": mean_clamped,
                "mean_dense_trait_score": mean_dense,
                "trait_recovery_pct": recovery,
                "comparisons": rows,
            }
        )
        if mean_dense is not None:
            dense_scores.append(mean_dense)
        if mean_baseline is not None:
            baseline_scores.append(mean_baseline)

    doc = {
        "phase": "B",
        "run_id": run_dir.name,
        "layer": layer_idx,
        "k_values": k_values,
        "clamp_mode": clamp_mode,
        "steer_alpha": steer_alpha,
        "sweep": sweep_results,
    }
    out_path = out_dir / "phase_b_multi_clamp.json"
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return doc


def main() -> int:
    p = argparse.ArgumentParser(description="SAE clamp experiments (Options C, A, B)")
    p.add_argument(
        "--phase",
        choices=["c", "a", "b", "all"],
        default="all",
        help="Which experiment phase to run",
    )
    p.add_argument("--run-id", default="dnd_good_scale", help="Persona run id (phases A/B)")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE)
    p.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    p.add_argument("--steer-alpha", type=float, default=DEFAULT_STEER_ALPHA)
    p.add_argument("--steered-alpha-key", default="1.5")
    p.add_argument(
        "--clamp-mode",
        choices=["additive_delta", "full_replacement"],
        default="additive_delta",
    )
    p.add_argument("--k-values", default="5,10,20,50", help="Comma-separated k for phase B")
    p.add_argument("--limit", type=int, default=0, help="Limit eval questions (0=all)")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--project", default="")
    p.add_argument("--out-dir", default="", help="Output directory")
    p.add_argument("--french-feature-id", type=int, default=0, help="Override French feature id")
    p.add_argument(
        "--steer-last-token-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp only last token position during generation",
    )
    p.add_argument("--latents-pt", default="", help="Path to sae latents checkpoint")
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    if args.force_cpu:
        os.environ["PERSONA_FORCE_CPU"] = "1"

    run_dir = PERSONA_RUNS / args.run_id
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else run_dir / "sae" / "clamp_experiments"
    )
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    latents_pt = Path(args.latents_pt).resolve() if args.latents_pt else None

    t0 = time.time()
    model, tokenizer, device = load_model_and_tokenizer()
    sae, sae_info = load_sae_for_layer(
        device,
        release=args.sae_release,
        sae_id=args.sae_id,
        hidden_state_index=args.layer + 1,
    )
    logger.info(
        "Loaded SAE %s / %s on %s (%.1fs)",
        args.sae_release,
        args.sae_id,
        device,
        time.time() - t0,
    )

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "layer": args.layer,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "clamp_mode": args.clamp_mode,
        "phases": {},
    }

    french_feature_id = args.french_feature_id if args.french_feature_id > 0 else None

    if args.phase in ("c", "all"):
        summary["phases"]["C"] = run_phase_c(
            model,
            tokenizer,
            device,
            sae,
            out_dir,
            layer_idx=args.layer,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            clamp_mode=args.clamp_mode,
            french_feature_id=french_feature_id,
            steer_last_token_only=args.steer_last_token_only,
        )

    if args.phase in ("a", "all"):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing run dir: {run_dir}")
        summary["phases"]["A"] = run_phase_a(
            model,
            tokenizer,
            device,
            sae,
            run_dir,
            out_dir,
            layer_idx=args.layer,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            steer_alpha=args.steer_alpha,
            steered_alpha_key=args.steered_alpha_key,
            clamp_mode=args.clamp_mode,
            limit=args.limit,
            skip_judge=args.skip_judge,
            project_id=args.project or None,
            latents_pt=latents_pt,
            steer_last_token_only=args.steer_last_token_only,
        )

    if args.phase in ("b", "all"):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing run dir: {run_dir}")
        summary["phases"]["B"] = run_phase_b(
            model,
            tokenizer,
            device,
            sae,
            run_dir,
            out_dir,
            layer_idx=args.layer,
            steer_alpha=args.steer_alpha,
            steered_alpha_key=args.steered_alpha_key,
            clamp_mode=args.clamp_mode,
            k_values=k_values,
            limit=args.limit,
            skip_judge=args.skip_judge,
            project_id=args.project or None,
            latents_pt=latents_pt,
            steer_last_token_only=args.steer_last_token_only,
        )

    summary_path = out_dir / "clamp_experiment_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Done in %.1fs; summary at %s", time.time() - t0, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
