"""
Phase 2: pretrained Gemma Scope 2 SAE (sae-lens) + activation snapshot at hook layer.

Default release/id match google/gemma-3-4b-it resid_post SAEs from HF hub
(see https://huggingface.co/google/gemma-scope-2-4b-it).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)

_sae = None
_sae_info: dict[str, Any] = {}


def sae_loaded() -> bool:
    return _sae is not None


def sae_status() -> dict[str, Any]:
    return {"loaded": _sae is not None, **_sae_info}


def try_load_sae(device: torch.device) -> None:
    """Load SAE once. Skips if DISABLE_SAE=1 or import/load fails."""
    global _sae, _sae_info
    _sae = None
    _sae_info = {}

    if os.environ.get("DISABLE_SAE", "").lower() in ("1", "true", "yes"):
        logger.info("DISABLE_SAE set; skipping SAE load.")
        return

    release = os.environ.get("SAE_RELEASE", "gemma-scope-2-4b-it-res")
    sae_id = os.environ.get("SAE_ID", "layer_22_width_16k_l0_medium")

    try:
        from sae_lens import SAE
    except ImportError as e:
        logger.warning("sae-lens not installed; Phase 2 disabled: %s", e)
        return

    try:
        sae = SAE.from_pretrained(release, sae_id)
        sae = sae.to(device)
        sae.eval()
    except Exception as e:
        logger.warning("Failed to load SAE %s / %s: %s", release, sae_id, e)
        return

    md = sae.cfg.metadata
    hook = None
    if md is not None:
        if isinstance(md, dict):
            hook = md.get("hook_name")
        else:
            hook = getattr(md, "hook_name", None)

    layer = _resid_layer_from_hook(hook) or _layer_from_sae_id(sae_id)
    if layer is None:
        hs_idx = int(os.environ.get("SAE_HIDDEN_STATE_INDEX", "23"))
        logger.warning(
            "Could not parse SAE layer from hook/id; using SAE_HIDDEN_STATE_INDEX=%s",
            hs_idx,
        )
    else:
        default_hs = layer + 1
        hs_idx = int(os.environ.get("SAE_HIDDEN_STATE_INDEX", str(default_hs)))

    _sae = sae
    _sae_info = {
        "release": release,
        "sae_id": sae_id,
        "hook_name": hook,
        "resid_layer": layer,
        "hidden_state_index": hs_idx,
        "d_in": int(sae.cfg.d_in),
        "d_sae": int(sae.cfg.d_sae),
    }
    logger.info("Loaded SAE %s / %s hook=%s hs_index=%s", release, sae_id, hook, hs_idx)


def _layer_from_sae_id(sae_id: str) -> int | None:
    m = re.match(r"layer_(\d+)_", sae_id)
    return int(m.group(1)) if m else None


def _resid_layer_from_hook(hook_name: str | None) -> int | None:
    if not hook_name:
        return None
    m = re.search(r"blocks\.(\d+)\.hook_resid_post", hook_name)
    if m:
        return int(m.group(1))
    m = re.search(r"layers\.(\d+)\.output", hook_name)
    if m:
        return int(m.group(1))
    return None


def build_conversation(system: str, user_message: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]


def _snapshot_tensor(
    model,
    tokenizer,
    sae,
    system: str,
    user_message: str,
    topk: int,
    hs_index: int,
) -> dict[str, Any]:
    conv = build_conversation(system, user_message)
    inputs = tokenizer.apply_chat_template(
        conv,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if not out.hidden_states or hs_index >= len(out.hidden_states):
        raise ValueError(
            f"hidden_states index {hs_index} out of range (len={len(out.hidden_states)})"
        )

    h = out.hidden_states[hs_index][0, -1, :].float()
    if h.shape[-1] != sae.cfg.d_in:
        raise ValueError(
            f"Hidden dim {h.shape[-1]} != SAE d_in {sae.cfg.d_in}; wrong layer/index?"
        )

    x = h.unsqueeze(0).unsqueeze(0)
    z = sae.encode(x)
    z1 = z[0, 0]
    k = min(int(topk), z1.numel())
    mag = z1.abs()
    vals, idx = torch.topk(mag, k=k)
    # feature id + signed activation
    signed = torch.gather(z1, 0, idx)

    eps = 1e-5
    l0 = int((mag > eps).sum().item())

    topk_list = [
        {"i": int(idx[j].item()), "v": float(signed[j].item())}
        for j in range(k)
    ]

    return {
        "hidden_l2": float(torch.norm(h).item()),
        "l0_gt_eps": l0,
        "eps": eps,
        "topk": topk_list,
        "z_abs_mean": float(mag.mean().item()),
    }


def compute_snapshot(pipe, system: str, user_message: str, topk: int = 24) -> dict[str, Any]:
    if _sae is None:
        raise RuntimeError("SAE not loaded")
    model = pipe.model
    tokenizer = pipe.tokenizer
    hs_index = int(_sae_info.get("hidden_state_index", -1))
    if hs_index < 0:
        raise RuntimeError("Invalid SAE_HIDDEN_STATE_INDEX / resid_layer")

    inner = _snapshot_tensor(model, tokenizer, _sae, system, user_message, topk, hs_index)
    return {
        "position": "last_prefill_token",
        **_sae_info,
        **inner,
    }


def compute_compare(
    pipe,
    message: str,
    system_a: str,
    system_b: str,
    topk: int = 24,
) -> dict[str, Any]:
    a = compute_snapshot(pipe, system_a, message, topk)
    b = compute_snapshot(pipe, system_b, message, topk)
    ids_a = {x["i"] for x in a["topk"]}
    ids_b = {x["i"] for x in b["topk"]}
    inter = len(ids_a & ids_b)
    union = len(ids_a | ids_b) or 1
    jaccard = inter / union

    # cosine on full z would need second forward with same seq - approximate with topk only:
    # build sparse dict from topk for rough overlap score only; full z cosine = re-encode (expensive)
    return {
        "a": a,
        "b": b,
        "topk_jaccard": float(jaccard),
        "topk_overlap_count": inter,
    }
