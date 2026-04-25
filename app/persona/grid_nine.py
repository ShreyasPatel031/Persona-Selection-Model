"""
3×3 dual-axis steering: sycophancy vector × chaos vector (from two persona_vectors.pt).

Checkpoint stores v = h_pos − h_neg. For the **sycophancy** axis we steer along **−v** so that
positive steering coefficients line up with the “detractor / neg-pole” side and negative
coefficients with the sycophant (pos) side — see COL_SPECS multipliers below.

Columns (left→right): Sycophant · neutral · Hater  (multipliers −1, 0, +1 on −v, times --scale-syc)
Rows: Lawful · neutral · Chaotic  (multipliers −1, 0, +1 on v_chaos, times --scale-chaos)

**2×2 coexistence mode** (``--grid-four``): corners only — neutral, sycophant-only, chaos-only,
both traits on — same α semantics as the 3×3 center row/column and Sycophant+Chaotic corner.
Each reply is scored with **both** trait judges (Vertex) plus coherence.

Default corner |α| scales: legacy 0.75 / 2.0, or — if present —
``persona_runs/grid_coherence80_calibration.json`` from
``python -m app.persona.coherence_alpha_sweep`` (mean coherence ≥ 80 on eval questions).

System prompt defaults to ``with_paragraph_cap("You are a helpful assistant.")`` (one short paragraph).

Example:
  python -m app.persona.grid_nine \\
    --question "Was I right to ship this on Friday?" \\
    --syc-vectors persona_runs/sycophancy_iter1/vectors/persona_vectors.pt \\
    --chaos-vectors persona_runs/chaos_iter1/vectors/persona_vectors.pt \\
    --syc-layer 29 --chaos-layer 27

  python -m app.persona.grid_nine --grid-four --question "..." \\
    --syc-bundle persona_runs/sycophancy_iter2/artifacts/trait_bundle.json \\
    --chaos-bundle persona_runs/chaos_iter2/artifacts/trait_bundle.json \\
    --syc-vectors ... --chaos-vectors ...

  # Composition experiments (see also ``python -m app.persona.vector_compose``):
  python -m app.persona.grid_nine --grid-four ... --orthogonalize-chaos
  python -m app.persona.grid_nine --grid-four ... --combined-layer 27 --norm-budget
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

# Row: chaos axis on +v_chaos (v = chaotic_pos − orderly_neg). −scale → lawful, +scale → chaotic.
ROW_SPECS = [
    ("Lawful (order)", -1),
    ("Neutral (chaos axis)", 0),
    ("Chaotic", 1),
]
# Col: syc axis on **−v_syc** (negate checkpoint v so Sycophant column steers toward pos). Mults × scale_syc.
# Sycophant: −1·scale on −v → +v  (toward flattery pole). Hater: +1·scale on −v → −v (toward neg pole).
COL_SPECS = [
    ("Sycophant", -1),
    ("Neutral (syc axis)", 0),
    ("Hater", 1),
]

# 2×2: (label, col_mult for syc, row_mult for chaos) — same multipliers as 3×3 COL_SPECS / ROW_SPECS.
FOUR_CELL_SPECS = [
    ("Neutral (no steer)", 0, 0),
    ("Sycophant only", -1, 0),
    ("Chaos only", 0, 1),
    ("Sycophant + chaos", -1, 1),
]


def build_steering_directions(
    v_s: torch.Tensor,
    v_c: torch.Tensor,
    layer_syc: int,
    layer_chaos: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    orthogonalize_chaos: bool = False,
    combined_layer: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Build d_s = −v_s[Ls], d_c = v_c[Lc] (or orthogonalized chaos per layer first).
    If combined_layer is set, Ls = Lc = combined_layer (Experiment E: single-layer injection).
    """
    from app.persona.vector_compose import orthogonalize_chaos_vs_syc

    v_c_use = orthogonalize_chaos_vs_syc(v_s, v_c) if orthogonalize_chaos else v_c
    if combined_layer is not None:
        ls = lc = int(combined_layer)
    else:
        ls, lc = layer_syc, layer_chaos
    d_s = (-v_s[ls]).to(device=device, dtype=dtype).view(1, 1, -1)
    d_c = v_c_use[lc].to(device=device, dtype=dtype).view(1, 1, -1)
    return d_s, d_c, ls, lc


def generate_steered_two_axes(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    question: str,
    *,
    layers: Any,
    layer_syc: int,
    direction_syc: torch.Tensor,
    alpha_syc: float,
    layer_chaos: int,
    direction_chaos: torch.Tensor,
    alpha_chaos: float,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Steer with v_syc at layer_syc and v_chaos at layer_chaos; α can be negative, zero, or positive."""
    from app.persona.steering_demo import _steering_hook_fn

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

    handles: list[Any] = []
    hook_calls_combined = [0]

    def _register(alpha: float, direction: torch.Tensor, layer_idx: int, hook_calls: list[int]) -> None:
        if alpha == 0.0:
            return
        hook = _steering_hook_fn(
            float(alpha),
            direction,
            steer_last_token_only=False,
            hook_calls=hook_calls,
        )
        handles.append(layers[layer_idx].register_forward_hook(hook))

    try:
        if layer_syc == layer_chaos:
            combined = alpha_syc * direction_syc + alpha_chaos * direction_chaos
            if combined.abs().max().item() > 0:
                hook = _steering_hook_fn(
                    1.0,
                    combined,
                    steer_last_token_only=False,
                    hook_calls=hook_calls_combined,
                )
                handles.append(layers[layer_syc].register_forward_hook(hook))
        else:
            _register(alpha_syc, direction_syc, layer_syc, hook_calls_combined)
            _register(alpha_chaos, direction_chaos, layer_chaos, hook_calls_combined)

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
        with torch.no_grad():
            gen_ids = model.generate(**gen_kw)
    finally:
        for h in handles:
            h.remove()

    if (alpha_syc != 0.0 or alpha_chaos != 0.0) and len(handles) > 0 and hook_calls_combined[0] == 0:
        raise RuntimeError("Steering hooks never ran; check layer indices vs model depth.")
    return tokenizer.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()


def generate_steered_two_axes_stream(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    question: str,
    *,
    layers: Any,
    layer_syc: int,
    direction_syc: torch.Tensor,
    alpha_syc: float,
    layer_chaos: int,
    direction_chaos: torch.Tensor,
    alpha_chaos: float,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
):
    """Yield assistant text chunks (same steering as ``generate_steered_two_axes``, streaming)."""
    from threading import Thread

    from app.persona.steering_demo import _steering_hook_fn
    from transformers import TextIteratorStreamer

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

    handles: list[Any] = []
    hook_calls_combined = [0]
    had_hooks = False

    def _register(alpha: float, direction: torch.Tensor, layer_idx: int, hook_calls: list[int]) -> None:
        if alpha == 0.0:
            return
        hook = _steering_hook_fn(
            float(alpha),
            direction,
            steer_last_token_only=False,
            hook_calls=hook_calls,
        )
        handles.append(layers[layer_idx].register_forward_hook(hook))

    try:
        if layer_syc == layer_chaos:
            combined = alpha_syc * direction_syc + alpha_chaos * direction_chaos
            if combined.abs().max().item() > 0:
                hook = _steering_hook_fn(
                    1.0,
                    combined,
                    steer_last_token_only=False,
                    hook_calls=hook_calls_combined,
                )
                handles.append(layers[layer_syc].register_forward_hook(hook))
        else:
            _register(alpha_syc, direction_syc, layer_syc, hook_calls_combined)
            _register(alpha_chaos, direction_chaos, layer_chaos, hook_calls_combined)

        had_hooks = len(handles) > 0
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kw: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "use_cache": True,
            "streamer": streamer,
        }
        if do_sample:
            gen_kw["temperature"] = float(temperature)

        def run_generate():
            with torch.no_grad():
                model.generate(**gen_kw)

        thread = Thread(target=run_generate)
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            thread.join(timeout=7200)
    finally:
        for h in handles:
            h.remove()

    if (alpha_syc != 0.0 or alpha_chaos != 0.0) and had_hooks and hook_calls_combined[0] == 0:
        raise RuntimeError("Steering hooks never ran; check layer indices vs model depth.")


def generate_plain_from_messages(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Continue the chat with no steering (assistant completion). ``messages`` must end with ``user``."""
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
    with torch.no_grad():
        gen_ids = model.generate(**gen_kw)
    return tokenizer.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()


def generate_steered_multi_gates_from_messages(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    messages: list[dict[str, str]],
    *,
    layers: Any,
    contributions: list[tuple[int, torch.Tensor, float]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """
    Assistant completion with additive steering. Each contribution is ``(layer_idx, direction_1x1d, alpha)``.
    Same-layer contributions are summed before a single hook is registered.
    """
    from app.persona.steering_demo import _steering_hook_fn

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

    combined_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx, direction, alpha in contributions:
        if abs(float(alpha)) < 1e-12:
            continue
        acc = float(alpha) * direction
        if layer_idx in combined_by_layer:
            combined_by_layer[layer_idx] = combined_by_layer[layer_idx] + acc
        else:
            combined_by_layer[layer_idx] = acc

    handles: list[Any] = []
    hook_calls_combined = [0]
    try:
        for layer_idx, vec in combined_by_layer.items():
            if vec.abs().max().item() <= 0:
                continue
            hook = _steering_hook_fn(
                1.0,
                vec,
                steer_last_token_only=False,
                hook_calls=hook_calls_combined,
            )
            handles.append(layers[layer_idx].register_forward_hook(hook))
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
        with torch.no_grad():
            gen_ids = model.generate(**gen_kw)
    finally:
        for h in handles:
            h.remove()

    if combined_by_layer and handles and hook_calls_combined[0] == 0:
        raise RuntimeError("Steering hooks never ran; check layer indices vs model depth.")
    return tokenizer.decode(gen_ids[0, in_len:], skip_special_tokens=True).strip()


def generate_plain_from_messages_stream(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
):
    """Stream assistant tokens with no steering."""
    from threading import Thread

    from transformers import TextIteratorStreamer

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
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "use_cache": True,
        "streamer": streamer,
    }
    if do_sample:
        gen_kw["temperature"] = float(temperature)

    def run_generate():
        with torch.no_grad():
            model.generate(**gen_kw)

    thread = Thread(target=run_generate)
    thread.start()
    try:
        for text in streamer:
            if text:
                yield text
    finally:
        thread.join(timeout=7200)


def generate_steered_multi_gates_from_messages_stream(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    messages: list[dict[str, str]],
    *,
    layers: Any,
    contributions: list[tuple[int, torch.Tensor, float]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
):
    """Stream assistant tokens with same additive multi-gate steering as the non-streaming helper."""
    from threading import Thread

    from app.persona.steering_demo import _steering_hook_fn
    from transformers import TextIteratorStreamer

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
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    combined_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx, direction, alpha in contributions:
        if abs(float(alpha)) < 1e-12:
            continue
        acc = float(alpha) * direction
        if layer_idx in combined_by_layer:
            combined_by_layer[layer_idx] = combined_by_layer[layer_idx] + acc
        else:
            combined_by_layer[layer_idx] = acc

    handles: list[Any] = []
    hook_calls_combined = [0]
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "use_cache": True,
        "streamer": streamer,
    }
    if do_sample:
        gen_kw["temperature"] = float(temperature)

    def run_generate():
        try:
            for layer_idx, vec in combined_by_layer.items():
                if vec.abs().max().item() <= 0:
                    continue
                hook = _steering_hook_fn(
                    1.0,
                    vec,
                    steer_last_token_only=False,
                    hook_calls=hook_calls_combined,
                )
                handles.append(layers[layer_idx].register_forward_hook(hook))
            with torch.no_grad():
                model.generate(**gen_kw)
        finally:
            for h in handles:
                h.remove()

    thread = Thread(target=run_generate)
    thread.start()
    try:
        for text in streamer:
            if text:
                yield text
    finally:
        thread.join(timeout=7200)


def run_grid(
    *,
    question: str,
    syc_vectors_pt: Path,
    chaos_vectors_pt: Path,
    layer_syc: int,
    layer_chaos: int,
    scale_syc: float,
    scale_chaos: float,
    system: str,
    model_id: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    orthogonalize_chaos: bool = False,
    norm_budget: bool = False,
    combined_layer: int | None = None,
) -> list[dict[str, Any]]:
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers

    if not syc_vectors_pt.is_file():
        raise FileNotFoundError(f"sycophancy vectors not found: {syc_vectors_pt}")
    if not chaos_vectors_pt.is_file():
        raise FileNotFoundError(f"chaos vectors not found: {chaos_vectors_pt}")

    ck_s = torch.load(syc_vectors_pt, map_location="cpu", weights_only=False)
    ck_c = torch.load(chaos_vectors_pt, map_location="cpu", weights_only=False)
    v_s = ck_s["v"].float()
    v_c = ck_c["v"].float()
    if v_s.shape != v_c.shape:
        logger.warning(
            "Sycophancy and chaos vectors differ in shape %s vs %s — same model assumed.",
            v_s.shape,
            v_c.shape,
        )

    model, tokenizer, device = load_model_and_tokenizer(model_id, device=None)
    dtype = next(model.parameters()).dtype
    layers = _language_model_layers(model)

    ls_eff = lc_eff = int(combined_layer) if combined_layer is not None else None
    eff_syc = ls_eff if ls_eff is not None else layer_syc
    eff_chaos = lc_eff if lc_eff is not None else layer_chaos
    n_layers = len(layers)
    for name, li in ("syc", eff_syc), ("chaos", eff_chaos):
        if not (0 <= li < n_layers):
            raise ValueError(f"layer_{name}={li} out of range [0, {n_layers - 1}]")

    d_s, d_c, ls, lc = build_steering_directions(
        v_s,
        v_c,
        layer_syc,
        layer_chaos,
        device,
        dtype,
        orthogonalize_chaos=orthogonalize_chaos,
        combined_layer=combined_layer,
    )
    if norm_budget and ls != lc:
        logger.warning(
            "--norm-budget ignored: steering layers differ (%s vs %s). Use --combined-layer.",
            ls,
            lc,
        )

    from app.persona.vector_compose import norm_budget_scale_same_layer

    out: list[dict[str, Any]] = []

    for row_label, mr in ROW_SPECS:
        alpha_chaos = float(mr) * scale_chaos
        for col_label, mc in COL_SPECS:
            alpha_syc = float(mc) * scale_syc
            a_s, a_c = alpha_syc, alpha_chaos
            lam = 1.0
            if norm_budget and ls == lc and a_s != 0.0 and a_c != 0.0:
                a_s, a_c, lam = norm_budget_scale_same_layer(
                    a_s, a_c, d_s, d_c, scale_syc, scale_chaos
                )
            reply = generate_steered_two_axes(
                model,
                tokenizer,
                device,
                system,
                question,
                layers=layers,
                layer_syc=ls,
                direction_syc=d_s,
                alpha_syc=a_s,
                layer_chaos=lc,
                direction_chaos=d_c,
                alpha_chaos=a_c,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            cell: dict[str, Any] = {
                "row": row_label,
                "col": col_label,
                "alpha_syc": a_s,
                "alpha_chaos": a_c,
                "reply": reply,
            }
            if lam != 1.0:
                cell["norm_budget_lambda"] = lam
                cell["alpha_syc_before_budget"] = alpha_syc
                cell["alpha_chaos_before_budget"] = alpha_chaos
            out.append(cell)

    return out


def run_grid_four(
    *,
    question: str,
    syc_vectors_pt: Path,
    chaos_vectors_pt: Path,
    layer_syc: int,
    layer_chaos: int,
    scale_syc: float,
    scale_chaos: float,
    system: str,
    model_id: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    orthogonalize_chaos: bool = False,
    norm_budget: bool = False,
    combined_layer: int | None = None,
) -> list[dict[str, Any]]:
    """Four corners: neutral, syc-only, chaos-only, both (same steering math as run_grid)."""
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.steering_demo import _language_model_layers

    if not syc_vectors_pt.is_file():
        raise FileNotFoundError(f"sycophancy vectors not found: {syc_vectors_pt}")
    if not chaos_vectors_pt.is_file():
        raise FileNotFoundError(f"chaos vectors not found: {chaos_vectors_pt}")

    ck_s = torch.load(syc_vectors_pt, map_location="cpu", weights_only=False)
    ck_c = torch.load(chaos_vectors_pt, map_location="cpu", weights_only=False)
    v_s = ck_s["v"].float()
    v_c = ck_c["v"].float()
    if v_s.shape != v_c.shape:
        logger.warning(
            "Sycophancy and chaos vectors differ in shape %s vs %s — same model assumed.",
            v_s.shape,
            v_c.shape,
        )

    model, tokenizer, device = load_model_and_tokenizer(model_id, device=None)
    dtype = next(model.parameters()).dtype
    layers = _language_model_layers(model)

    eff_syc = int(combined_layer) if combined_layer is not None else layer_syc
    eff_chaos = int(combined_layer) if combined_layer is not None else layer_chaos
    n_layers = len(layers)
    for name, li in ("syc", eff_syc), ("chaos", eff_chaos):
        if not (0 <= li < n_layers):
            raise ValueError(f"layer_{name}={li} out of range [0, {n_layers - 1}]")

    d_s, d_c, ls, lc = build_steering_directions(
        v_s,
        v_c,
        layer_syc,
        layer_chaos,
        device,
        dtype,
        orthogonalize_chaos=orthogonalize_chaos,
        combined_layer=combined_layer,
    )
    if norm_budget and ls != lc:
        logger.warning(
            "--norm-budget ignored: steering layers differ (%s vs %s). Use --combined-layer.",
            ls,
            lc,
        )

    from app.persona.vector_compose import norm_budget_scale_same_layer

    out: list[dict[str, Any]] = []
    for label, mc, mr in FOUR_CELL_SPECS:
        alpha_syc = float(mc) * scale_syc
        alpha_chaos = float(mr) * scale_chaos
        a_s, a_c = alpha_syc, alpha_chaos
        lam = 1.0
        if norm_budget and ls == lc and a_s != 0.0 and a_c != 0.0:
            a_s, a_c, lam = norm_budget_scale_same_layer(
                a_s, a_c, d_s, d_c, scale_syc, scale_chaos
            )
        reply = generate_steered_two_axes(
            model,
            tokenizer,
            device,
            system,
            question,
            layers=layers,
            layer_syc=ls,
            direction_syc=d_s,
            alpha_syc=a_s,
            layer_chaos=lc,
            direction_chaos=d_c,
            alpha_chaos=a_c,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        row: dict[str, Any] = {
            "cell": label,
            "alpha_syc": a_s,
            "alpha_chaos": a_c,
            "reply": reply,
        }
        if lam != 1.0:
            row["norm_budget_lambda"] = lam
            row["alpha_syc_before_budget"] = alpha_syc
            row["alpha_chaos_before_budget"] = alpha_chaos
        out.append(row)
    return out


def score_grid_four_dual_traits(
    *,
    cells: list[dict[str, Any]],
    question: str,
    syc_bundle: Path,
    chaos_bundle: Path,
    judge_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach sycophancy score, chaos score, and coherence to each cell (mutates copies)."""
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.quality_gates import score_coherence
    from app.persona.response_style import with_paragraph_cap
    from app.persona.schemas import PersonaTraitArtifact

    syc_art = PersonaTraitArtifact.model_validate_json(
        syc_bundle.read_text(encoding="utf-8")
    )
    chaos_art = PersonaTraitArtifact.model_validate_json(
        chaos_bundle.read_text(encoding="utf-8")
    )
    syc_instr = judge_rubric_to_instructions(syc_art.judge_rubric)
    chaos_instr = judge_rubric_to_instructions(chaos_art.judge_rubric)
    syc_sys = with_paragraph_cap(syc_art.neg_system_prompt)
    chaos_sys = with_paragraph_cap(chaos_art.neg_system_prompt)
    jkw = dict(judge_kwargs)

    enriched: list[dict[str, Any]] = []
    for c in cells:
        row = dict(c)
        text = str(row.get("reply", ""))
        # Sycophancy trait judge (syc bundle system + rubric)
        try:
            js = score_transcript(syc_instr, syc_sys, question, text, **jkw)
            row["sycophancy_trait_score"] = int(js.score)
            row["sycophancy_trait_reason"] = js.short_reason
        except Exception as exc:
            logger.warning("Sycophancy judge failed for cell %s: %s", row.get("cell"), exc)
            row["sycophancy_trait_score"] = None
            row["sycophancy_trait_reason"] = str(exc)
        # Chaos trait judge
        try:
            jc = score_transcript(chaos_instr, chaos_sys, question, text, **jkw)
            row["chaos_trait_score"] = int(jc.score)
            row["chaos_trait_reason"] = jc.short_reason
        except Exception as exc:
            logger.warning("Chaos judge failed for cell %s: %s", row.get("cell"), exc)
            row["chaos_trait_score"] = None
            row["chaos_trait_reason"] = str(exc)
        try:
            row["coherence_score"] = int(score_coherence(text, **jkw))
        except Exception as exc:
            logger.warning("Coherence judge failed for cell %s: %s", row.get("cell"), exc)
            row["coherence_score"] = None
        enriched.append(row)
    return enriched


def load_calibration_scales(
    path: Path,
) -> tuple[float | None, float | None]:
    """Read scale_recommended for sycophancy and chaos from coherence sweep JSON."""
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    s = data.get("sycophancy") or {}
    c = data.get("chaos") or {}
    rs = s.get("scale_recommended")
    rc = c.get("scale_recommended")
    return (float(rs) if rs is not None else None, float(rc) if rc is not None else None)


def main(argv: list[str] | None = None) -> int:
    from app.persona.response_style import with_paragraph_cap

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Generate 3×3 or 2×2 replies: sycophancy × chaos steering; optional dual trait + coherence judges.",
    )
    p.add_argument(
        "--grid-four",
        action="store_true",
        help="2×2 coexistence grid: neutral, syc-only, chaos-only, both; score both traits per cell.",
    )
    p.add_argument("--question", required=True, help="Single eval user message.")
    p.add_argument(
        "--syc-bundle",
        type=Path,
        default=Path("persona_runs/sycophancy_iter1/artifacts/trait_bundle.json"),
        help="Trait bundle for sycophancy Vertex judge (should match --syc-vectors extraction).",
    )
    p.add_argument(
        "--chaos-bundle",
        type=Path,
        default=Path("persona_runs/chaos_iter1/artifacts/trait_bundle.json"),
        help="Trait bundle for chaos Vertex judge (should match --chaos-vectors extraction).",
    )
    p.add_argument(
        "--syc-vectors",
        type=Path,
        default=Path("persona_runs/sycophancy_iter1/vectors/persona_vectors.pt"),
        help="persona_vectors.pt from sycophancy extraction.",
    )
    p.add_argument(
        "--chaos-vectors",
        type=Path,
        default=Path("persona_runs/chaos_iter1/vectors/persona_vectors.pt"),
        help="persona_vectors.pt from chaos extraction.",
    )
    p.add_argument("--syc-layer", type=int, default=29, help="Decoder layer for v_syc (default 29).")
    p.add_argument(
        "--chaos-layer",
        type=int,
        default=27,
        help="Decoder layer for v_chaos (default 27; tune per chaos_iter2 validation).",
    )
    p.add_argument(
        "--scale-syc",
        type=float,
        default=None,
        help="Corner |α| on syc axis. Default: from --calibration-json if present, else 0.75.",
    )
    p.add_argument(
        "--scale-chaos",
        type=float,
        default=None,
        help="Corner |α| on chaos axis. Default: from --calibration-json if present, else 2.0.",
    )
    p.add_argument(
        "--calibration-json",
        type=Path,
        default=Path("persona_runs/grid_coherence80_calibration.json"),
        help="Optional JSON from coherence_alpha_sweep (scale_recommended per trait).",
    )
    p.add_argument(
        "--system",
        default="",
        help="Base system prompt. Default: helpful assistant + one-paragraph cap.",
    )
    p.add_argument(
        "--no-paragraph-cap",
        action="store_true",
        help="Do not append the one-paragraph reply constraint to --system.",
    )
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--model-id",
        default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full grid JSON (optional).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write grid as CSV (row, col, alpha_syc, alpha_chaos, question, reply).",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="With --grid-four: generate only; do not call Vertex (no trait/coherence scores).",
    )
    p.add_argument(
        "--orthogonalize-chaos",
        action="store_true",
        help="Project chaos v per layer orthogonal to syc v before steering (Experiment C).",
    )
    p.add_argument(
        "--norm-budget",
        action="store_true",
        help="When both α non-zero and injection is same-layer, scale (α_syc, α_chaos) so "
        "||α_s d_s + α_c d_c|| ≤ max(||scale_syc·d_s||, ||scale_chaos·d_c||) (Experiment D).",
    )
    p.add_argument(
        "--combined-layer",
        type=int,
        default=None,
        help="Inject both directions at this decoder layer (Experiment E; enables norm-budget with split cal layers).",
    )
    args = p.parse_args(argv)

    cal_syc, cal_chaos = load_calibration_scales(args.calibration_json.resolve())
    scale_syc = float(args.scale_syc) if args.scale_syc is not None else (cal_syc if cal_syc is not None else 0.75)
    scale_chaos = (
        float(args.scale_chaos) if args.scale_chaos is not None else (cal_chaos if cal_chaos is not None else 2.0)
    )

    system = args.system.strip() if args.system else "You are a helpful assistant."
    if not args.no_paragraph_cap:
        system = with_paragraph_cap(system)

    jkw: dict[str, Any] = {}
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if pid:
        jkw["project_id"] = pid

    try:
        if args.grid_four:
            cells = run_grid_four(
                question=args.question,
                syc_vectors_pt=args.syc_vectors.resolve(),
                chaos_vectors_pt=args.chaos_vectors.resolve(),
                layer_syc=args.syc_layer,
                layer_chaos=args.chaos_layer,
                scale_syc=scale_syc,
                scale_chaos=scale_chaos,
                system=system,
                model_id=args.model_id or None,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                orthogonalize_chaos=args.orthogonalize_chaos,
                norm_budget=args.norm_budget,
                combined_layer=args.combined_layer,
            )
            if not args.skip_judge:
                cells = score_grid_four_dual_traits(
                    cells=cells,
                    question=args.question,
                    syc_bundle=args.syc_bundle.resolve(),
                    chaos_bundle=args.chaos_bundle.resolve(),
                    judge_kwargs=jkw,
                )
        else:
            cells = run_grid(
                question=args.question,
                syc_vectors_pt=args.syc_vectors.resolve(),
                chaos_vectors_pt=args.chaos_vectors.resolve(),
                layer_syc=args.syc_layer,
                layer_chaos=args.chaos_layer,
                scale_syc=scale_syc,
                scale_chaos=scale_chaos,
                system=system,
                model_id=args.model_id or None,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                orthogonalize_chaos=args.orthogonalize_chaos,
                norm_budget=args.norm_budget,
                combined_layer=args.combined_layer,
            )
    except Exception as e:
        logger.exception("%s", e)
        return 1

    doc: dict[str, Any] = {
        "question": args.question,
        "system": system,
        "syc_vectors": str(args.syc_vectors.resolve()),
        "chaos_vectors": str(args.chaos_vectors.resolve()),
        "syc_bundle": str(args.syc_bundle.resolve()),
        "chaos_bundle": str(args.chaos_bundle.resolve()),
        "layer_syc": args.syc_layer,
        "layer_chaos": args.chaos_layer,
        "layer_syc_effective": args.combined_layer if args.combined_layer is not None else args.syc_layer,
        "layer_chaos_effective": args.combined_layer if args.combined_layer is not None else args.chaos_layer,
        "scale_syc": scale_syc,
        "scale_chaos": scale_chaos,
        "calibration_json": str(args.calibration_json.resolve()) if args.calibration_json else None,
        "syc_steering_direction": "negated_v (checkpoint v = pos−neg; steer along −v for column semantics)",
        "grid_four": bool(args.grid_four),
        "orthogonalize_chaos": bool(args.orthogonalize_chaos),
        "norm_budget": bool(args.norm_budget),
        "combined_layer": args.combined_layer,
        "grid": cells,
    }

    print()
    print("=" * 72)
    if args.grid_four:
        print("  2×2 coexistence  (chaos off | chaos on)")
        print("                   (syc off | syc on  = Sycophant corner)")
        print(
            f"  scales: |α_syc|={scale_syc:g} (Sycophant → α_syc=-{scale_syc:g}), "
            f"|α_chaos|={scale_chaos:g} (Chaotic → α_chaos=+{scale_chaos:g})",
            flush=True,
        )
        if not args.skip_judge:
            print("  Vertex: sycophancy rubric + chaos rubric + coherence per cell", flush=True)
        if args.orthogonalize_chaos:
            print("  Composition: chaos v orthogonalized vs syc v (per layer)", flush=True)
        if args.norm_budget:
            print("  Composition: norm-budget scaling when both axes active (same layer only)", flush=True)
        if args.combined_layer is not None:
            print(f"  Composition: both hooks at layer {args.combined_layer}", flush=True)
    else:
        print("  3×3 grid  (columns: Sycophant · Neutral · Hater)")
        print("            (rows: Lawful · Neutral · Chaotic)")
        print(
            f"  scales: syc ±{scale_syc:g}, chaos ±{scale_chaos:g}  "
            f"(CLI override or {args.calibration_json.name} if present)",
            flush=True,
        )
    print("=" * 72)

    if args.grid_four:
        def _score_suffix(c: dict[str, Any]) -> str:
            if args.skip_judge or c.get("sycophancy_trait_score") is None:
                return ""
            return (
                f"  | scores: sycophancy={c.get('sycophancy_trait_score')}, "
                f"chaos={c.get('chaos_trait_score')}, coherence={c.get('coherence_score')}"
            )

        print()
        print(
            "  Layout (same α as 3×3 neutral row × neutral col + Sycophant/Chaotic corner):",
            flush=True,
        )
        print(
            f"    chaos off, syc off  → Neutral   |  chaos on, syc off  → Chaos only",
            flush=True,
        )
        print(
            f"    chaos off, syc on   → Syc only  |  chaos on, syc on  → Both",
            flush=True,
        )
        for c in cells:
            print(
                f"\n--- {c['cell']} ---  α_syc={c['alpha_syc']:.4g}, α_chaos={c['alpha_chaos']:.4g}"
                f"{_score_suffix(c)}"
            )
            print(c["reply"])
            if not args.skip_judge and c.get("sycophancy_trait_reason"):
                print(f"  (syc judge) {c.get('sycophancy_trait_reason')}")
            if not args.skip_judge and c.get("chaos_trait_reason"):
                print(f"  (chaos judge) {c.get('chaos_trait_reason')}")
    else:
        for i, (row_label, _) in enumerate(ROW_SPECS):
            print(f"\n--- {row_label} ---\n")
            for j, (col_label, _) in enumerate(COL_SPECS):
                cell = cells[i * 3 + j]
                print(f"** {col_label} **  (α_syc={cell['alpha_syc']:.4g}, α_chaos={cell['alpha_chaos']:.4g})")
                print(cell["reply"])
                print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out_json.resolve()}", file=sys.stderr)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if args.grid_four:
                w.writerow(
                    [
                        "cell",
                        "alpha_syc",
                        "alpha_chaos",
                        "question",
                        "sycophancy_trait_score",
                        "chaos_trait_score",
                        "coherence_score",
                        "reply",
                    ]
                )
                for cell in cells:
                    w.writerow(
                        [
                            cell.get("cell", ""),
                            cell["alpha_syc"],
                            cell["alpha_chaos"],
                            args.question,
                            cell.get("sycophancy_trait_score", ""),
                            cell.get("chaos_trait_score", ""),
                            cell.get("coherence_score", ""),
                            cell["reply"],
                        ]
                    )
            else:
                w.writerow(["row", "col", "alpha_syc", "alpha_chaos", "question", "reply"])
                for cell in cells:
                    w.writerow(
                        [
                            cell["row"],
                            cell["col"],
                            cell["alpha_syc"],
                            cell["alpha_chaos"],
                            args.question,
                            cell["reply"],
                        ]
                    )
        print(f"Wrote {args.out_csv.resolve()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
