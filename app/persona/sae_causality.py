"""SAE reconstruction fidelity, top-k recovery, and feature ablation diagnostics."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TOPK_SWEEP = [10, 50, 100, 200, 500, 1000]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _sae_device(sae: Any) -> torch.device:
    return next(sae.parameters()).device


def encode_vector(sae: Any, v: torch.Tensor) -> torch.Tensor:
    """Encode a single dense vector (d_in,) -> latent (d_sae,)."""
    d_in = int(sae.cfg.d_in)
    if v.shape[-1] != d_in:
        raise ValueError(f"Vector dim {v.shape[-1]} != SAE d_in {d_in}")
    dev = _sae_device(sae)
    x = v.float().unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        z = sae.encode(x)[0, 0].float()
    return z


def decode_latent(sae: Any, z: torch.Tensor) -> torch.Tensor:
    """Decode latent (d_sae,) -> dense (d_in,)."""
    dev = _sae_device(sae)
    d_sae = int(sae.cfg.d_sae)
    if z.shape[-1] != d_sae:
        raise ValueError(f"Latent dim {z.shape[-1]} != SAE d_sae {d_sae}")
    z3 = z.float().unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        h = sae.decode(z3)[0, 0].float()
    return h


def latent_topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k features by |activation|."""
    z_k = torch.zeros_like(z)
    if k <= 0 or z.numel() == 0:
        return z_k
    kk = min(int(k), int(z.numel()))
    idx = torch.topk(z.abs(), k=kk).indices
    z_k[idx] = z[idx]
    return z_k


def full_reconstruction_metrics(sae: Any, v: torch.Tensor) -> dict[str, Any]:
    """Experiment A: full SAE encode->decode fidelity for persona vector."""
    z = encode_vector(sae, v)
    v_recon = decode_latent(sae, z)
    n_active = int((z.abs() > 1e-8).sum().item())
    return {
        "cosine_full": cosine_similarity(v, v_recon),
        "dense_norm": float(v.float().norm().item()),
        "recon_norm": float(v_recon.norm().item()),
        "n_active_features": n_active,
        "d_sae": int(sae.cfg.d_sae),
    }


def topk_reconstruction_sweep(
    sae: Any,
    v: torch.Tensor,
    ks: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Experiment B: cosine recovery vs number of retained SAE features."""
    ks = ks or DEFAULT_TOPK_SWEEP
    z = encode_vector(sae, v)
    rows: list[dict[str, Any]] = []
    for k in ks:
        z_k = latent_topk_mask(z, k)
        v_k = decode_latent(sae, z_k)
        rows.append(
            {
                "k": int(k),
                "cosine": cosine_similarity(v, v_k),
                "recon_norm": float(v_k.norm().item()),
                "n_nonzero": int((z_k.abs() > 1e-8).sum().item()),
            }
        )
    return rows


def sparse_direction_from_latent(
    sae: Any,
    z: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Decode latent to dense direction; optionally unit-normalize."""
    direction = decode_latent(sae, z)
    if normalize:
        norm = direction.norm()
        if norm < 1e-8:
            raise ValueError("Sparse reconstruction norm is near zero.")
        direction = direction / norm
    return direction


def magnitude_matched_alpha(
    dense_vector: torch.Tensor,
    sparse_direction: torch.Tensor,
    base_alpha: float,
    *,
    normalize_sparse: bool,
) -> float:
    """
    Scale steering alpha so ||alpha * sparse|| matches ||base_alpha * dense||.

    When sparse is unit-normalized, alpha_sparse = base_alpha * ||dense||.
  When sparse is raw decode, alpha_sparse = base_alpha * (||dense|| / ||sparse||).
    """
    dense_norm = float(dense_vector.float().norm().item())
    sparse_norm = float(sparse_direction.float().norm().item())
    if dense_norm < 1e-8:
        return float(base_alpha)
    if normalize_sparse:
        return float(base_alpha) * dense_norm
    if sparse_norm < 1e-8:
        return float(base_alpha)
    return float(base_alpha) * (dense_norm / sparse_norm)


def check_hf_sae_checkpoint(
    release: str,
    sae_id: str,
    *,
    repo_id: str = "google/gemma-scope-2-4b-it",
) -> dict[str, Any]:
    """Check whether a Gemma Scope SAE checkpoint exists on HuggingFace."""
    # sae-lens release maps to HF subfolder under resid_post_all or resid_post
    subfolders = ["resid_post_all", "resid_post", "transcoder_all"]
    results: dict[str, Any] = {
        "release": release,
        "sae_id": sae_id,
        "repo_id": repo_id,
        "exists": False,
        "subfolder": None,
        "config_url": None,
        "params_url": None,
    }
    for sub in subfolders:
        cfg_path = f"{sub}/{sae_id}/config.json"
        url = f"https://huggingface.co/{repo_id}/resolve/main/{cfg_path}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    results["exists"] = True
                    results["subfolder"] = sub
                    results["config_url"] = url
                    results["params_url"] = (
                        f"https://huggingface.co/{repo_id}/resolve/main/"
                        f"{sub}/{sae_id}/params.safetensors"
                    )
                    return results
        except urllib.error.HTTPError as e:
            if e.code not in (404, 307, 302):
                logger.debug("HF HEAD %s -> %s", url, e.code)
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning("HF check failed for %s: %s", url, e)
            break
    return results


def load_persona_vector(
    vectors_pt: Path,
    layer_idx: int,
) -> torch.Tensor:
    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    v_full = ckpt["v"].float()
    if not (0 <= layer_idx < v_full.shape[0]):
        raise ValueError(
            f"layer_idx {layer_idx} out of range for v shape {tuple(v_full.shape)}"
        )
    return v_full[layer_idx]


def run_reconstruction_diagnostics(
    run_dir: Path,
    out_json: Path,
    *,
    layer_idx: int,
    sae_release: str | None = None,
    sae_id: str | None = None,
    topk_sweep: list[int] | None = None,
    device: torch.device | None = None,
) -> Path:
    """Run Experiments A + B; write sae/reconstruction_diagnostics.json."""
    from app.persona.activations import _pick_device
    from app.phase2 import load_sae_for_layer

    vectors_pt = run_dir / "vectors" / "persona_vectors.pt"
    if not vectors_pt.is_file():
        raise FileNotFoundError(f"Missing {vectors_pt}")

    dev = device or _pick_device()
    sae_dev = dev
    sid = sae_id or ""
    if sid and "262k" in sid:
        sae_dev = torch.device("cpu")
        logger.info("Loading 262k SAE on CPU for reconstruction diagnostics")
    sae, sae_info = load_sae_for_layer(sae_dev, release=sae_release, sae_id=sae_id)

    v = load_persona_vector(vectors_pt, layer_idx)
    full = full_reconstruction_metrics(sae, v.to(_sae_device(sae)))
    sweep = topk_reconstruction_sweep(sae, v.to(_sae_device(sae)), topk_sweep)

    hf_check = None
    if sae_info.get("sae_id"):
        hf_check = check_hf_sae_checkpoint(
            str(sae_info.get("release") or sae_release or ""),
            str(sae_info.get("sae_id")),
        )

    doc: dict[str, Any] = {
        "run_id": run_dir.name,
        "layer": layer_idx,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "vectors_pt": str(vectors_pt.resolve()),
        "full_reconstruction": full,
        "topk_sweep": sweep,
        "hf_checkpoint_check": hf_check,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json


def sae_feature_ablation_hook_fn(
    sae: Any,
    feature_ids: list[int],
    hook_calls: list[int],
) -> Any:
    """
    Forward hook: subtract SAE decoder contribution of ``feature_ids`` from hidden states.

    Used after dense steering to test whether ablating features removes trait behavior.
    """

    def hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, tuple) and len(output) > 0:
            h = output[0]
        elif isinstance(output, torch.Tensor):
            h = output
        else:
            return output
        if h.dim() != 3:
            return output

        hook_calls[0] += 1
        sae_dev = _sae_device(sae)
        x = h.to(sae_dev)
        with torch.no_grad():
            z = sae.encode(x)
            z_mask = torch.zeros_like(z)
            for fid in feature_ids:
                if 0 <= fid < z.shape[-1]:
                    z_mask[..., fid] = z[..., fid]
            contrib = sae.decode(z_mask).to(device=h.device, dtype=h.dtype)
        h.sub_(contrib)
        return output

    return hook


def run_ablation_validation(
    run_dir: Path,
    attribution_json: Path,
    out_json: Path,
    *,
    layer_idx: int,
    steer_alpha: float = 4.0,
    top_k: int = 50,
    limit: int = 0,
    model_id: str | None = None,
    sae_release: str | None = None,
    sae_id: str | None = None,
    device: torch.device | None = None,
    skip_judge: bool = False,
    project_id: str | None = None,
) -> Path:
    """Experiment C: dense steer + SAE feature ablation vs baseline/dense-only."""
    import os

    from app.phase2 import load_sae_for_layer
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.quality_gates import score_coherence
    from app.persona.response_style import with_paragraph_cap
    from app.persona.sae_experiment import _generate_steered_reply, _sae_dir
    from app.persona.schemas import PersonaTraitArtifact
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn

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

    ckpt = torch.load(vectors_pt, map_location="cpu", weights_only=False)
    u_dense = ckpt["v"].float()[layer_idx]

    model, tokenizer, dev = load_model_and_tokenizer(model_id, device=device)
    sae, sae_info = load_sae_for_layer(dev, release=sae_release, sae_id=sae_id)
    dtype = next(model.parameters()).dtype
    direction = u_dense.to(device=dev, dtype=dtype).view(1, 1, -1)

    questions = gen.get("questions") or []
    if limit and limit < len(questions):
        questions = questions[:limit]

    layers = _language_model_layers(model)
    rows: list[dict[str, Any]] = []
    for i, qrow in enumerate(questions):
        q = qrow["question"]
        neg_sys = qrow.get("neg_system") or neg_sys_default
        logger.info("Ablation validate %s/%s", i + 1, len(questions))

        baseline = _generate_steered_reply(
            model, tokenizer, dev, neg_sys, q, layer_idx, u_dense, 0.0
        )
        dense_reply = _generate_steered_reply(
            model, tokenizer, dev, neg_sys, q, layer_idx, u_dense, steer_alpha
        )

        # Dense steer + SAE ablation of top-k features on same layer
        messages = [
            {"role": "system", "content": neg_sys},
            {"role": "user", "content": q},
        ]
        raw_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = raw_ids.to(dev) if isinstance(raw_ids, torch.Tensor) else raw_ids["input_ids"].to(dev)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=dev)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        steer_calls = [0]
        ablate_calls = [0]
        steer_hook = _steering_hook_fn(
            float(steer_alpha),
            direction,
            steer_last_token_only=False,
            hook_calls=steer_calls,
        )
        ablate_hook = sae_feature_ablation_hook_fn(sae, feature_ids, ablate_calls)
        h1 = layers[layer_idx].register_forward_hook(steer_hook)
        h2 = layers[layer_idx].register_forward_hook(ablate_hook)
        try:
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
        finally:
            h2.remove()
            h1.remove()
        if steer_calls[0] == 0 or ablate_calls[0] == 0:
            raise RuntimeError("Steer or ablation hook did not run during generation.")
        ablated_reply = tokenizer.decode(
            gen_ids[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        row: dict[str, Any] = {
            "question_index": i,
            "question": q,
            "alpha": steer_alpha,
            "feature_ids": feature_ids,
            "baseline_reply": baseline,
            "dense_steered_reply": dense_reply,
            "ablated_steered_reply": ablated_reply,
        }
        if not skip_judge:
            pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            for key, reply in (
                ("baseline", baseline),
                ("dense", dense_reply),
                ("ablated", ablated_reply),
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

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key, -1) >= 0]
        return sum(vals) / len(vals) if vals else None

    doc = {
        "trait": gen.get("trait"),
        "run_id": run_dir.name,
        "layer": layer_idx,
        "steer_alpha": steer_alpha,
        "top_k_features": top_k,
        "feature_ids": feature_ids,
        "sae_release": sae_info.get("release"),
        "sae_id": sae_info.get("sae_id"),
        "attribution_json": str(attribution_json.resolve()),
        "skip_judge": skip_judge,
        "comparisons": rows,
        "mean_baseline_trait_score": _mean("baseline_trait_score"),
        "mean_dense_trait_score": _mean("dense_trait_score"),
        "mean_ablated_trait_score": _mean("ablated_trait_score"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return out_json
