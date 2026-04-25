"""Additive residual steering along persona direction v_ℓ (neg system prompt + α·v or α·û)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from app.persona.activations import (
    MODEL_ID,
    _hf_login,
    _pick_device,
    _pick_model_dtype,
    mean_residuals_over_assistant,
)
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact

logger = logging.getLogger(__name__)


def _steering_hook_fn(
    alpha: float,
    direction: torch.Tensor,
    *,
    steer_last_token_only: bool,
    hook_calls: list[int],
) -> Any:
    """Build a forward_hook that adds alpha * direction to decoder hidden states."""

    def hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, tuple) and len(output) > 0:
            h = output[0]
        elif isinstance(output, torch.Tensor):
            h = output
        else:
            return output
        if h.dim() == 3:
            hook_calls[0] += 1
            if steer_last_token_only:
                h[:, -1:, :].add_(alpha * direction)
            else:
                h.add_(alpha * direction)
        return output

    return hook


def _language_model_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Resolve decoder layers for Gemma-3 multimodal and common causal LMs."""
    m = model
    if hasattr(m, "model") and m.model is not None:
        inner = m.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    raise RuntimeError(
        "Could not find decoder layers on this model class; extend _language_model_layers."
    )


def run_steering_ramp(
    bundle_path: Path,
    vectors_pt: Path,
    out_json: Path,
    *,
    question: str,
    layer_idx: int,
    model_id: str | None = None,
    device: torch.device | None = None,
    n_steps: int = 5,
    alpha_max: float = 6.0,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    use_cache: bool = True,
    steer_last_token_only: bool = True,
    rng_seed: int | None = None,
) -> Path:
    """
    Fix **neg** (boring) system prompt; at decoder block `layer_idx`, add
    α * û to the residual stream, where û = v_ℓ / ||v_ℓ||.

    By default only the **last sequence position** is steered each forward (prefill + each
    decode step). Broadcasting the same δ to *all* tokens often barely moves greedy
    next-token argmax; last-token steering matches common activation-steering practice.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    raw = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw)
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ckpt["v"].float()
    if layer_idx < 0:
        layer_idx = int(v_full.shape[0]) + layer_idx
    if not (0 <= layer_idx < v_full.shape[0]):
        raise ValueError(f"layer_idx {layer_idx} out of range for v shape {tuple(v_full.shape)}")
    v_ell = v_full[layer_idx]
    u = v_ell / (v_ell.norm() + 1e-8)

    _hf_login()
    dev = device or _pick_device()
    mid = model_id or MODEL_ID
    logger.info("Loading %s on %s for steering ramp…", mid, dev)
    tok = AutoTokenizer.from_pretrained(mid)
    dtype = _pick_model_dtype(dev)
    logger.info("Model dtype: %s", dtype)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(dev)
    model.eval()

    layers = _language_model_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx {layer_idx} >= num_layers {len(layers)} on loaded model."
        )

    messages = [
        {"role": "system", "content": neg_sys},
        {"role": "user", "content": question},
    ]
    raw_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(raw_ids, torch.Tensor):
        input_ids = raw_ids.to(dev)
    else:
        input_ids = raw_ids["input_ids"].to(dev)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=dev)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    results: list[dict[str, Any]] = []
    fractions = [i / (n_steps - 1) for i in range(n_steps)]

    for i, frac in enumerate(fractions):
        alpha = float(frac * alpha_max)
        direction = u.to(device=dev, dtype=dtype).view(1, 1, -1)

        layer_mod = layers[layer_idx]
        hook_calls = [0]
        hook = _steering_hook_fn(
            alpha,
            direction,
            steer_last_token_only=steer_last_token_only,
            hook_calls=hook_calls,
        )
        handle = layer_mod.register_forward_hook(hook)
        try:
            if do_sample and rng_seed is not None:
                s = int(rng_seed) + i * 10_007
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
            gen_kw: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attn,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": pad_id,
                "use_cache": use_cache,
            }
            if do_sample:
                gen_kw["temperature"] = temperature
            with torch.no_grad():
                gen_ids = model.generate(**gen_kw)
        finally:
            handle.remove()

        if hook_calls[0] == 0:
            raise RuntimeError(
                "Steering hook never ran; layer index or model structure mismatch."
            )
        logger.info("Hook ran %s times (alpha=%.4f)", hook_calls[0], alpha)

        text = tok.decode(
            gen_ids[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        results.append(
            {
                "step_index": i,
                "fraction_of_max": frac,
                "alpha": alpha,
                "alpha_max": alpha_max,
                "layer": layer_idx,
                "hook_calls": hook_calls[0],
                "reply": text,
            }
        )
        logger.info("Steering step %s/%s alpha=%.4f", i + 1, n_steps, alpha)

    doc: dict[str, Any] = {
        "note": "Neg (boring) system prompt fixed; residual at layer L gets +α·û with û=v_L/||v_L|| from persona_vectors.pt (pos−neg extraction).",
        "model_id": mid,
        "trait_label": artifact.trait_label,
        "question": question,
        "neg_system_preview": neg_sys[:280] + ("…" if len(neg_sys) > 280 else ""),
        "vectors_pt": str(vectors_pt.resolve()),
        "layer": layer_idx,
        "alpha_max": alpha_max,
        "n_steps": n_steps,
        "do_sample": do_sample,
        "steer_last_token_only": steer_last_token_only,
        "rng_seed": rng_seed,
        "unique_reply_count": len({x["reply"] for x in results}),
        "iterations": results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def run_steering_ab_compare(
    bundle_path: Path,
    vectors_pt: Path,
    out_json: Path,
    *,
    question: str,
    layer_idx: int,
    model_id: str | None = None,
    device: torch.device | None = None,
    alpha: float = 1.0,
    steering_alphas: list[float] | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    use_cache: bool = True,
    steer_last_token_only: bool = False,
    rng_seed: int | None = None,
    include_pos_baseline: bool = False,
) -> Path:
    """
    Minimal A/B: same boring (neg) system prompt, same question.

    - **A:** no steering.
    - **B:** at layer ``layer_idx``, **h ← h + α·v_ℓ** with **raw** ``v_ℓ`` from ``persona_vectors.pt``
      (paper §3.2 style; not unit-normalized). Default **α=1** is exactly **one full extracted**
      difference vector at that layer (not û, not a tiny scaled direction).

    Optional **C:** ``include_pos_baseline`` — same question with **pos** (jester) system prompt only,
    no vector add, so you can compare steered-neg to true in-prompt persona.

    If B differs from A, additive steering is having an effect at this α; if prior ramps with û
    looked dead, scaling/normalization was likely the issue.
    """
    raw = bundle_path.read_text(encoding="utf-8")
    artifact = PersonaTraitArtifact.model_validate_json(raw)
    neg_sys = with_paragraph_cap(artifact.neg_system_prompt)
    pos_sys = with_paragraph_cap(artifact.pos_system_prompt)

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ckpt["v"].float()
    if layer_idx < 0:
        layer_idx = int(v_full.shape[0]) + layer_idx
    if not (0 <= layer_idx < v_full.shape[0]):
        raise ValueError(f"layer_idx {layer_idx} out of range for v shape {tuple(v_full.shape)}")
    v_ell = v_full[layer_idx]

    _hf_login()
    dev = device or _pick_device()
    mid = model_id or MODEL_ID
    logger.info("Loading %s on %s for steering A/B…", mid, dev)
    tok = AutoTokenizer.from_pretrained(mid)
    dtype = _pick_model_dtype(dev)
    logger.info("Model dtype: %s", dtype)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(dev)
    model.eval()

    layers = _language_model_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx {layer_idx} >= num_layers {len(layers)} on loaded model."
        )

    def _encode_chat(system: str, user_q: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_q},
        ]
        raw_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(raw_ids, torch.Tensor):
            ids = raw_ids.to(dev)
        else:
            ids = raw_ids["input_ids"].to(dev)
        mask = torch.ones_like(ids, dtype=torch.long, device=dev)
        return ids, mask, int(ids.shape[-1])

    neg_ids, neg_attn, neg_in_len = _encode_chat(neg_sys, question)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    direction = v_ell.to(device=dev, dtype=dtype).view(1, 1, -1)
    alphas_list = (
        [float(x) for x in steering_alphas]
        if steering_alphas is not None
        else [float(alpha)]
    )
    if not alphas_list:
        raise ValueError("steering_alphas / alpha must yield at least one value.")

    def generate_once(
        input_ids: torch.Tensor,
        attn: torch.Tensor,
        in_len: int,
        *,
        with_steering: bool,
        steer_alpha: float = 1.0,
        seed_offset: int = 0,
    ) -> tuple[str, int]:
        if do_sample and rng_seed is not None:
            s = int(rng_seed) + seed_offset
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
        gen_kw: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "use_cache": use_cache,
        }
        if do_sample:
            gen_kw["temperature"] = temperature
        if not with_steering:
            with torch.no_grad():
                gen_ids = model.generate(**gen_kw)
            text = tok.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()
            return text, 0
        hook_calls = [0]
        hook = _steering_hook_fn(
            float(steer_alpha),
            direction,
            steer_last_token_only=steer_last_token_only,
            hook_calls=hook_calls,
        )
        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                gen_ids = model.generate(**gen_kw)
        finally:
            handle.remove()
        if hook_calls[0] == 0:
            raise RuntimeError(
                "Steering hook never ran; layer index or model structure mismatch."
            )
        text = tok.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()
        return text, hook_calls[0]

    baseline_reply, _ = generate_once(neg_ids, neg_attn, neg_in_len, with_steering=False, seed_offset=0)

    steered_by_alpha: dict[str, dict[str, Any]] = {}
    steered_reply = ""
    hook_calls = 0
    for i, a in enumerate(alphas_list):
        rep, hc = generate_once(
            neg_ids,
            neg_attn,
            neg_in_len,
            with_steering=True,
            steer_alpha=a,
            seed_offset=100 + i,
        )
        key = f"{float(a):g}"
        steered_by_alpha[key] = {"reply": rep, "hook_calls": hc}
        if i == 0:
            steered_reply = rep
            hook_calls = hc

    pos_persona_reply: str | None = None
    if include_pos_baseline:
        p_ids, p_attn, p_in_len = _encode_chat(pos_sys, question)
        pos_persona_reply, _ = generate_once(
            p_ids, p_attn, p_in_len, with_steering=False, seed_offset=50
        )

    v_cpu = v_full  # (L, d), float32 on CPU
    u_cpu = v_ell / (v_ell.norm() + 1e-8)

    def along_persona_vector(reply: str, *, system_for_forward: str) -> dict[str, Any]:
        """Mean-pooled assistant hidden states per layer; dot with v_ℓ and with û at steering layer."""
        empty: dict[str, Any] = {
            "h_dot_v_at_steering_layer": None,
            "h_dot_u_at_steering_layer": None,
            "h_dot_v_per_layer": None,
            "error": None,
        }
        if not reply.strip():
            empty["error"] = "empty_reply"
            return empty
        try:
            h = mean_residuals_over_assistant(
                model, tok, dev, system_for_forward, question, reply
            ).float()
        except (RuntimeError, ValueError) as e:
            empty["error"] = str(e)
            return empty
        if h.shape != v_cpu.shape:
            empty["error"] = (
                f"hidden shape {tuple(h.shape)} != v {tuple(v_cpu.shape)} "
                "(model vs persona_vectors.pt mismatch?)"
            )
            return empty
        vf = v_cpu.to(device=h.device, dtype=h.dtype)
        per_layer = (h * vf).sum(dim=1).detach().cpu()
        u_dev = u_cpu.to(device=h.device, dtype=h.dtype)
        h_ell = h[layer_idx]
        return {
            "h_dot_v_at_steering_layer": float(per_layer[layer_idx].item()),
            "h_dot_u_at_steering_layer": float((h_ell * u_dev).sum().item()),
            "h_dot_v_per_layer": [float(x) for x in per_layer.tolist()],
            "error": None,
        }

    along_b = along_persona_vector(baseline_reply, system_for_forward=neg_sys)
    for a_key, row in steered_by_alpha.items():
        row["along_persona_vector"] = along_persona_vector(
            row["reply"], system_for_forward=neg_sys
        )
    along_s = steered_by_alpha[f"{float(alphas_list[0]):g}"]["along_persona_vector"]
    along_p: dict[str, Any] | None = None
    if pos_persona_reply is not None:
        along_p = along_persona_vector(pos_persona_reply, system_for_forward=pos_sys)
    db = along_b.get("h_dot_v_at_steering_layer")
    ds = along_s.get("h_dot_v_at_steering_layer")
    delta_v = None if db is None or ds is None else float(ds - db)
    ub = along_b.get("h_dot_u_at_steering_layer")
    sb_u = along_s.get("h_dot_u_at_steering_layer")
    delta_u = None if ub is None or sb_u is None else float(sb_u - ub)

    logger.info(
        "Along v @ layer %s: baseline h·v=%s first-α steered h·v=%s (Δ=%s); "
        "baseline h·û=%s first-α steered h·û=%s (Δ=%s); alphas=%s",
        layer_idx,
        db,
        ds,
        delta_v,
        ub,
        sb_u,
        delta_u,
        alphas_list,
    )

    print(
        f"\n=== steering-ab: layer {layer_idx}, raw v_ℓ, α sweep {alphas_list} ===\n"
        f"baseline  h·v={db!r}  h·û={ub!r}\n"
        f"first α   h·v={ds!r}  h·û={sb_u!r}  (Δ h·v={delta_v!r}  Δ h·û={delta_u!r})\n"
        f"\n--- baseline (neg only) ---\n{baseline_reply}\n",
        flush=True,
    )
    for a in alphas_list:
        ak = f"{float(a):g}"
        row = steered_by_alpha[ak]
        av = row.get("along_persona_vector") or {}
        print(
            f"\n--- steered neg + α·v_ℓ, α={ak} ---\n"
            f"    h·v={av.get('h_dot_v_at_steering_layer')!r}  "
            f"h·û={av.get('h_dot_u_at_steering_layer')!r}\n"
            f"{row['reply']}\n",
            flush=True,
        )
    if pos_persona_reply is not None:
        print(
            f"\n--- pos / jester (system prompt only, no vector) ---\n{pos_persona_reply}\n",
            flush=True,
        )

    doc: dict[str, Any] = {
        "note": "A/B: neg-only vs neg + α·v_ℓ (multi-α in steered_by_alpha). Optional pos baseline.",
        "model_id": mid,
        "trait_label": artifact.trait_label,
        "question": question,
        "neg_system_preview": neg_sys[:280] + ("…" if len(neg_sys) > 280 else ""),
        "pos_system_preview": pos_sys[:280] + ("…" if len(pos_sys) > 280 else ""),
        "vectors_pt": str(vectors_pt.resolve()),
        "layer": layer_idx,
        "steering_alphas": alphas_list,
        "alpha": float(alphas_list[0]),
        "steering": "raw_v",
        "steer_last_token_only": steer_last_token_only,
        "v_ell_l2": float(v_ell.float().norm().item()),
        "do_sample": do_sample,
        "rng_seed": rng_seed,
        "baseline_reply": baseline_reply,
        "steered_reply": steered_reply,
        "steered_by_alpha": steered_by_alpha,
        "replies_identical": baseline_reply == steered_reply,
        "steered_hook_calls": hook_calls,
        "pos_persona_reply": pos_persona_reply,
        "steered_matches_pos_persona": (
            None
            if pos_persona_reply is None
            else (steered_reply.strip() == pos_persona_reply.strip())
        ),
        "along_persona_vector": {
            "steering_layer": layer_idx,
            "baseline": along_b,
            "steered_first_alpha": along_s,
            "pos_persona": along_p,
            "delta_h_dot_v_at_steering_layer": delta_v,
            "delta_h_dot_u_at_steering_layer": delta_u,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (identical=%s)", out_json, doc["replies_identical"])
    return out_json
