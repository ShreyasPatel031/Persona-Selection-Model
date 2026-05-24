"""SAE encoding over assistant-span residuals + signed feature attribution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from app.persona.activations import (
    _as_input_ids_tensor,
    _chat_turns,
    _prompt_token_len,
    load_model_and_tokenizer,
)

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


def _hidden_state_index_for_layer(layer_idx: int) -> int:
    """Gemma-style: hs[0]=embeddings; hs[i]=after block i-1."""
    return layer_idx + 1


def assistant_hidden_span_at_layer(
    model: "PreTrainedModel",
    tokenizer: "AutoTokenizer",
    device: torch.device,
    system: str,
    user_q: str,
    assistant_a: str,
    layer_idx: int,
) -> tuple[torch.Tensor, list[str], int]:
    """
    Teacher forward; return assistant-token hidden states at one layer.

    Returns:
        h: (n_assistant_tokens, hidden_dim)
        token_strs: decoded token strings for assistant span
        prompt_len: index where assistant starts
    """
    if not assistant_a.strip():
        raise ValueError("Empty assistant text.")

    messages = _chat_turns(system, user_q, assistant_a)
    raw_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    input_ids = _as_input_ids_tensor(raw_ids, device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

    prompt_len = _prompt_token_len(tokenizer, system, user_q)
    seq_len = input_ids.shape[-1]
    if prompt_len >= seq_len:
        raise RuntimeError("No assistant tokens in sequence.")

    hs_index = _hidden_state_index_for_layer(layer_idx)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )

    hs = out.hidden_states
    if hs is None or hs_index >= len(hs):
        raise RuntimeError(
            f"hidden_states index {hs_index} out of range (len={len(hs) if hs else 0})"
        )

    h = hs[hs_index][0, prompt_len:, :].float()
    if h.shape[0] == 0:
        raise RuntimeError("Empty assistant span for hidden states.")

    ids = input_ids[0, prompt_len:].tolist()
    token_strs = [tokenizer.decode([tid]) for tid in ids]
    return h, token_strs, prompt_len


def encode_hidden_span(
    sae: Any,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SAE-encode each assistant token hidden state.

    Returns:
        z: (n_tokens, d_sae) signed feature activations
        z_mean: (d_sae,) mean over assistant tokens
    """
    if h.dim() != 2:
        raise ValueError(f"Expected h shape (n_tokens, d); got {tuple(h.shape)}")
    d_in = int(sae.cfg.d_in)
    if h.shape[-1] != d_in:
        raise ValueError(f"Hidden dim {h.shape[-1]} != SAE d_in {d_in}")

    sae_dev = next(sae.parameters()).device
    x = h.unsqueeze(0).to(sae_dev)  # (1, n_tokens, d_in) — match SAE device
    with torch.no_grad():
        z = sae.encode(x)[0].float()  # (n_tokens, d_sae)
    z_mean = z.mean(dim=0)
    return z, z_mean


def collect_feature_token_examples(
    z: torch.Tensor,
    token_strs: list[str],
    feature_id: int,
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Top assistant tokens where feature_id had highest |activation|."""
    if z.shape[0] != len(token_strs):
        raise ValueError("z rows must match token_strs length.")
    col = z[:, feature_id]
    k = min(top_n, col.numel())
    if k == 0:
        return []
    vals, idx = torch.topk(col.abs(), k=k)
    out: list[dict[str, Any]] = []
    for j in range(k):
        ti = int(idx[j].item())
        out.append(
            {
                "token_index": ti,
                "token": token_strs[ti],
                "activation": float(col[ti].item()),
            }
        )
    return out


def build_sparse_direction(
    sae: Any,
    feature_ids: list[int],
    coefficients: list[float],
    device: torch.device,
    dtype: torch.dtype,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Reconstruct dense direction from sparse SAE feature subset via decode."""
    d_sae = int(sae.cfg.d_sae)
    z = torch.zeros(1, 1, d_sae, device=device, dtype=torch.float32)
    for fid, coef in zip(feature_ids, coefficients):
        if not (0 <= fid < d_sae):
            raise ValueError(f"feature id {fid} out of range for d_sae={d_sae}")
        z[0, 0, fid] = float(coef)
    with torch.no_grad():
        h_delta = sae.decode(z).float()
    direction = h_delta[0, 0]
    if normalize:
        norm = direction.norm()
        if norm < 1e-8:
            raise ValueError("Sparse reconstruction norm is near zero.")
        direction = direction / norm
    return direction.to(dtype=dtype)


from app.persona.sae_common import compute_feature_attribution


def run_encode_latents(
    generations_path: Path,
    out_pt: Path,
    *,
    layer_idx: int,
    sae_release: str | None = None,
    sae_id: str | None = None,
    model_id: str | None = None,
    device: torch.device | None = None,
) -> Path:
    """Load generations.json, SAE-encode pos/neg/steered replies; save sae_latents.pt."""
    from app.phase2 import load_sae_for_layer

    gen = json.loads(generations_path.read_text(encoding="utf-8"))
    questions_in = gen.get("questions") or []
    if not questions_in:
        raise ValueError(f"No questions in {generations_path}")

    model, tokenizer, dev = load_model_and_tokenizer(model_id, device=device)
    # Load large SAEs (262k+) on CPU to avoid GPU OOM; 16k fits on GPU fine
    sae_device = dev
    if sae_id and "262k" in sae_id:
        sae_device = torch.device("cpu")
        logger.info("Loading 262k SAE on CPU to avoid GPU OOM")
    sae, sae_info = load_sae_for_layer(sae_device, release=sae_release, sae_id=sae_id)

    resid_layer = sae_info.get("resid_layer")
    if resid_layer is not None and int(resid_layer) != layer_idx:
        logger.warning(
            "SAE resid_layer=%s != requested layer_idx=%s; using layer_idx for extraction.",
            resid_layer,
            layer_idx,
        )

    encoded_questions: list[dict[str, Any]] = []
    feature_examples: dict[int, list[dict[str, Any]]] = {}

    for qi, qrow in enumerate(questions_in):
        logger.info("SAE encode question %s/%s", qi + 1, len(questions_in))
        q = qrow["question"]
        pos_sys = qrow["pos_system"]
        neg_sys = qrow["neg_system"]
        pos_reply = qrow["pos_reply"]
        neg_reply = qrow["neg_reply"]

        h_pos, tok_pos, _ = assistant_hidden_span_at_layer(
            model, tokenizer, dev, pos_sys, q, pos_reply, layer_idx
        )
        h_neg, tok_neg, _ = assistant_hidden_span_at_layer(
            model, tokenizer, dev, neg_sys, q, neg_reply, layer_idx
        )
        z_pos, z_pos_mean = encode_hidden_span(sae, h_pos)
        z_neg, z_neg_mean = encode_hidden_span(sae, h_neg)

        z_steered: dict[str, torch.Tensor] = {}
        z_steered_tokens: dict[str, torch.Tensor] = {}
        tok_steered: dict[str, list[str]] = {}
        for s in qrow.get("steered") or []:
            alpha_key = f"{float(s['alpha']):g}"
            reply = s["reply"]
            h_st, tok_st, _ = assistant_hidden_span_at_layer(
                model, tokenizer, dev, neg_sys, q, reply, layer_idx
            )
            z_st, z_st_mean = encode_hidden_span(sae, h_st)
            z_steered[alpha_key] = z_st_mean.cpu()
            z_steered_tokens[alpha_key] = z_st.cpu()
            tok_steered[alpha_key] = tok_st

        # accumulate token examples from pos/neg for later attribution labels
        for z_tok, toks in (
            (z_pos, tok_pos),
            (z_neg, tok_neg),
            *((z_steered_tokens[k], tok_steered[k]) for k in z_steered_tokens),
        ):
            if z_tok.numel() == 0:
                continue
            max_per_feat = z_tok.abs().max(dim=0).values
            top_feat_ids = torch.topk(max_per_feat, k=min(32, max_per_feat.numel())).indices
            for fid in top_feat_ids.tolist():
                ex = collect_feature_token_examples(z_tok, toks, fid, top_n=1)
                if not ex:
                    continue
                prev = feature_examples.get(fid, [])
                if len(prev) < 5:
                    prev.append({**ex[0], "question_index": qi})
                    feature_examples[fid] = prev

        encoded_questions.append(
            {
                "question": q,
                "question_index": qi,
                "z_pos_mean": z_pos_mean.cpu(),
                "z_neg_mean": z_neg_mean.cpu(),
                "z_steered": z_steered,
            }
        )

    payload = {
        "trait": gen.get("trait"),
        "layer": layer_idx,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "d_sae": int(sae.cfg.d_sae),
        "d_in": int(sae.cfg.d_in),
        "questions": encoded_questions,
        "feature_token_examples": feature_examples,
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    logger.info("Wrote %s", out_pt)
    return out_pt


def run_feature_attribution(
    latents_pt: Path,
    out_json: Path,
    *,
    steered_alpha_key: str = "2.0",
    top_k: int = 20,
) -> Path:
    ckpt = torch.load(latents_pt, map_location="cpu", weights_only=False)
    questions = ckpt.get("questions") or []
    attr = compute_feature_attribution(
        questions,
        steered_alpha_key=steered_alpha_key,
        top_k=top_k,
    )

    examples = ckpt.get("feature_token_examples") or {}
    for block_name in ("top_positive_features", "top_negative_features"):
        for row in attr[block_name]:
            fid = row["feature_id"]
            row["token_examples"] = examples.get(fid, [])

    doc = {
        "trait": ckpt.get("trait"),
        "layer": ckpt.get("layer"),
        "sae_release": ckpt.get("sae_release"),
        "sae_id": ckpt.get("sae_id"),
        "latents_pt": str(latents_pt.resolve()),
        **attr,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json
