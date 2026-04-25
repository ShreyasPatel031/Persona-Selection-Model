"""Step D: mean residual hidden states over assistant tokens → persona vectors v_ℓ (paper §2.2)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from app.persona.layer_heuristics import v1_layer_recommendation

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it")


def _load_dotenv_repo_root() -> None:
    """Load repo-root .env into os.environ (gitignored; optional dependency)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")


def _hf_login() -> None:
    _load_dotenv_repo_root()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)


def _pick_device() -> torch.device:
    if os.environ.get("PERSONA_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_model_dtype(dev: torch.device) -> torch.dtype:
    """
    Load dtype for Gemma teacher-forwards (step-d, steering).

    - CPU: fp32.
    - CUDA: **fp16 is not used by default** — it produced NaN `v` / norms for Gemma-3-4B-it
      teacher-forwards on T4-class GPUs. Default is **bf16** when supported, else **fp32**.
      Opt-in to fast fp16 only with ``PERSONA_CUDA_ALLOW_FP16=1`` (may reproduce NaNs).
    """
    if dev.type != "cuda":
        return torch.float32
    if os.environ.get("PERSONA_CUDA_ALLOW_FP16", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.warning(
            "PERSONA_CUDA_ALLOW_FP16: loading Gemma in float16 — may NaN in persona vectors."
        )
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    model_id: str | None = None,
    *,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, AutoTokenizer, torch.device]:
    mid = model_id or MODEL_ID
    dev = device or _pick_device()
    _hf_login()
    logger.info("Loading %s on %s…", mid, dev)
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
    return model, tok, dev


def _chat_turns(system: str, user_q: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_q},
        {"role": "assistant", "content": assistant},
    ]


def _as_input_ids_tensor(raw: Any, device: torch.device) -> torch.Tensor:
    """apply_chat_template may return Tensor or BatchEncoding (Gemma 3)."""
    if isinstance(raw, torch.Tensor):
        return raw.to(device)
    return raw["input_ids"].to(device)


def _prompt_token_len(
    tokenizer: AutoTokenizer, system: str, user_q: str
) -> int:
    """Tokens through start of assistant generation (exclusive of assistant text)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_q},
    ]
    t = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(t, torch.Tensor):
        return int(t.shape[-1])
    return int(t["input_ids"].shape[-1])


def mean_residuals_over_assistant(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    system: str,
    user_q: str,
    assistant_a: str,
) -> torch.Tensor:
    """
    Teacher forward on full [system,user,assistant] sequence.
    Returns tensor (num_layers, hidden_dim) — mean over assistant token positions per layer.
    Uses hidden_states[1..num_layers] (post-layer residuals, aligned with layer index).
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
    if prompt_len > seq_len:
        raise RuntimeError(
            f"prompt_len {prompt_len} > seq_len {seq_len}; chat template mismatch."
        )
    if prompt_len == seq_len:
        raise RuntimeError("No assistant tokens in sequence (prompt_len == seq_len).")

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )

    hs = out.hidden_states
    if hs is None or len(hs) < 2:
        raise RuntimeError("Model returned no hidden_states.")

    # hs[0] = embeddings; hs[i] = output after block i-1 for Gemma-style stacks
    slices = []
    for li in range(1, len(hs)):
        h = hs[li][0, prompt_len:, :]  # (n_resp, d)
        if h.shape[0] == 0:
            raise RuntimeError("Empty assistant span for hidden state extraction.")
        slices.append(h.mean(dim=0))
    return torch.stack(slices, dim=0)  # (n_blocks, d)


def iter_kept_rollouts(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if not o.get("kept"):
            continue
        if o.get("error"):
            continue
        if o.get("score") is None:
            continue
        yield o


def _split_half_cosine(
    pos_stack: torch.Tensor,
    neg_stack: torch.Tensor,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Split-half reliability of the persona vector direction.

    Randomly partition pos and neg rollout activations into two halves, compute
    an independent vector from each, and measure cosine similarity per layer.
    Repeats `n_splits` times and reports mean ± std.

    Interpretation (per layer, at the recommended layer):
      cosine ≥ 0.8  →  direction is stable, vector quality is likely sufficient
      0.5–0.8       →  partially converged, more data may help
      < 0.5         →  direction is noise-dominated, need significantly more data
    """
    n_pos, n_neg = pos_stack.shape[0], neg_stack.shape[0]
    if n_pos < 4 or n_neg < 4:
        return {
            "mean_cosine_per_layer": [],
            "mean_cosine_at_argmax_norm": None,
            "interpretation": "too_few_samples",
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_splits": 0,
        }

    rng = torch.Generator().manual_seed(seed)
    num_layers = int(pos_stack.shape[1])
    all_cosines = []  # (n_splits, num_layers)

    for _ in range(n_splits):
        pi = torch.randperm(n_pos, generator=rng)
        ni = torch.randperm(n_neg, generator=rng)
        hp1 = pos_stack[pi[: n_pos // 2]].mean(0)
        hp2 = pos_stack[pi[n_pos // 2 :]].mean(0)
        hn1 = neg_stack[ni[: n_neg // 2]].mean(0)
        hn2 = neg_stack[ni[n_neg // 2 :]].mean(0)
        v1 = hp1 - hn1
        v2 = hp2 - hn2
        cos = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)  # (num_layers,)
        all_cosines.append(cos)

    stacked = torch.stack(all_cosines, dim=0)  # (n_splits, num_layers)
    mean_per_layer = stacked.mean(dim=0)
    std_per_layer = stacked.std(dim=0) if n_splits > 1 else torch.zeros(num_layers)

    norms = (pos_stack.mean(0) - neg_stack.mean(0)).norm(dim=-1)
    argmax_norm = int(norms[:max(num_layers - 2, 1)].argmax().item())
    cos_at_best = float(mean_per_layer[argmax_norm].item())

    if cos_at_best >= 0.8:
        interp = "stable"
    elif cos_at_best >= 0.5:
        interp = "partially_converged"
    else:
        interp = "noise_dominated"

    return {
        "mean_cosine_per_layer": [round(float(x), 4) for x in mean_per_layer.tolist()],
        "std_cosine_per_layer": [round(float(x), 4) for x in std_per_layer.tolist()],
        "mean_cosine_at_argmax_norm": round(cos_at_best, 4),
        "argmax_norm_layer": argmax_norm,
        "interpretation": interp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_splits": n_splits,
    }


def run_step_d(
    rollouts_jsonl: Path,
    out_pt: Path,
    summary_json: Path,
    *,
    model_id: str | None = None,
    device: torch.device | None = None,
) -> Path:
    pos_rows = []
    neg_rows = []
    for o in iter_kept_rollouts(rollouts_jsonl):
        arm = o.get("arm")
        if arm == "pos":
            pos_rows.append(o)
        elif arm == "neg":
            neg_rows.append(o)

    if not pos_rows:
        raise ValueError("No kept pos rollouts in jsonl (need kept=true, score set).")
    if not neg_rows:
        raise ValueError("No kept neg rollouts in jsonl.")

    model, tokenizer, dev = load_model_and_tokenizer(model_id, device=device)

    def run_arm(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean over rollouts, stacked per-rollout activations)."""
        mats = []
        for i, o in enumerate(rows):
            logger.info("Forward %s/%s (%s)", i + 1, len(rows), o.get("arm"))
            m = mean_residuals_over_assistant(
                model,
                tokenizer,
                dev,
                o["system"],
                o["question"],
                o["assistant_a"],
            )
            mats.append(m.cpu())
        stack = torch.stack(mats, dim=0)  # (n, layers, d)
        return stack.mean(dim=0), stack  # (layers, d), (n, layers, d)

    h_pos, pos_mats = run_arm(pos_rows)
    h_neg, neg_mats = run_arm(neg_rows)
    v = h_pos - h_neg

    split_half = _split_half_cosine(pos_mats, neg_mats)
    sh_cos = split_half.get("mean_cosine_at_argmax_norm")
    sh_interp = split_half.get("interpretation", "?")
    logger.info(
        "Split-half cosine = %s (%s) at layer %s — %s",
        sh_cos,
        sh_interp,
        split_half.get("argmax_norm_layer"),
        {
            "stable": "direction converged, vector quality likely sufficient",
            "partially_converged": "partially converged, more data may improve",
            "noise_dominated": "NOISY — need significantly more rollout data",
            "too_few_samples": "too few samples to compute (<4 per arm)",
        }.get(sh_interp, ""),
    )

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    layer_v1 = v1_layer_recommendation(v)
    meta = {
        "model_id": model_id or MODEL_ID,
        "num_layers": int(v.shape[0]),
        "hidden_dim": int(v.shape[1]),
        "kept_pos": len(pos_rows),
        "kept_neg": len(neg_rows),
        "rollouts_jsonl": str(rollouts_jsonl.resolve()),
        "layer_recommendation_v1": layer_v1,
        "split_half_cosine": split_half,
    }
    torch.save(
        {
            "meta": meta,
            "h_pos_mean": h_pos,
            "h_neg_mean": h_neg,
            "v": v,
        },
        out_pt,
    )
    summary_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_pt)
    return out_pt
