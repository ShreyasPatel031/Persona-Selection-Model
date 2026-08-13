#!/usr/bin/env python3
"""
SAE-SSV: Supervised Steering Vector optimization in SAE latent space.

Implements the L_steer objective from He et al. (EMNLP 2025):
  L_steer = ||z' - mu+||^2 - ||z' - mu-||^2 + lambda_lm * L_LM + beta * ||v_I||_1

where z' = z + v, with v constrained to be nonzero only on d_steer
F-stat-selected dimensions.

After optimization, the steering vector is decoded back to residual stream
via W_dec and evaluated by generating + judging at multiple scaling factors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import (
    _as_input_ids_tensor,
    _chat_turns,
    _prompt_token_len,
    load_model_and_tokenizer,
)
from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
from app.persona.sae_common import load_rollout_question_pairs
from app.persona.sae_encode import assistant_hidden_span_at_layer
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers, _steering_hook_fn
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait


def _extract_hidden_3d(output) -> torch.Tensor | None:
    if isinstance(output, tuple) and len(output) > 0:
        h = output[0]
    elif isinstance(output, torch.Tensor):
        h = output
    else:
        return None
    if h.dim() != 3:
        return None
    return h


def f_statistic_per_feature(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask_pos = y > 0.5
    mask_neg = ~mask_pos
    n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())
    n = n_pos + n_neg
    grand = z.mean(axis=0)
    mean_pos = z[mask_pos].mean(axis=0)
    mean_neg = z[mask_neg].mean(axis=0)
    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((z[mask_pos] - mean_pos) ** 2).sum(axis=0) + (
        (z[mask_neg] - mean_neg) ** 2
    ).sum(axis=0)
    ms_within = ss_within / max(n - 2, 1)
    f = np.divide(ss_between, ms_within, out=np.zeros_like(ss_between), where=ms_within > 1e-12)
    return f.astype(np.float64)


def collect_latents(
    model, tok, dev, sae, layer: int, pairs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect mean SAE latents from rollout pairs.

    Returns z_all (n, d_sae), y_all (n,).
    """
    z_rows, y_rows = [], []

    for pi, pair in enumerate(pairs):
        for label, reply_key, sys_key in (
            (1, "pos_reply", "pos_system"),
            (0, "neg_reply", "neg_system"),
        ):
            system = str(pair[sys_key])
            question = str(pair["question"])
            reply = str(pair[reply_key])
            if len(reply.strip()) < 10:
                continue
            try:
                h, _, _ = assistant_hidden_span_at_layer(
                    model, tok, dev, system, question, reply, layer,
                )
                sae_dev = next(sae.parameters()).device
                x = h.unsqueeze(0).to(sae_dev)
                with torch.no_grad():
                    z = sae.encode(x)[0].float()
                z_mean = z.mean(dim=0)
                z_rows.append(z_mean.cpu().numpy().astype(np.float64))
                y_rows.append(label)
            except Exception as exc:
                print(f"  skip pair {pi} label={label}: {exc}", flush=True)

    if len(z_rows) < 8:
        raise RuntimeError(f"Too few samples ({len(z_rows)}); need >=8")
    return np.stack(z_rows, axis=0), np.array(y_rows, dtype=np.float64)


def collect_latents_samples(
    model, tok, dev, sae, layer: int, samples: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect mean SAE latents from individual labeled rollout rows.

    Each sample: {system, question, reply, label} with label 1=pos, 0=neg.
    Returns z_all (n, d_sae), y_all (n,).
    """
    z_rows, y_rows = [], []

    for si, sample in enumerate(samples):
        system = str(sample["system"])
        question = str(sample["question"])
        reply = str(sample["reply"])
        label = int(sample["label"])
        if len(reply.strip()) < 10:
            continue
        try:
            h, _, _ = assistant_hidden_span_at_layer(
                model, tok, dev, system, question, reply, layer,
            )
            sae_dev = next(sae.parameters()).device
            x = h.unsqueeze(0).to(sae_dev)
            with torch.no_grad():
                z = sae.encode(x)[0].float()
            z_mean = z.mean(dim=0)
            z_rows.append(z_mean.cpu().numpy().astype(np.float64))
            y_rows.append(label)
        except Exception as exc:
            print(f"  skip sample {si} label={label}: {exc}", flush=True)

    if len(z_rows) < 8:
        raise RuntimeError(f"Too few samples ({len(z_rows)}); need >=8")
    return np.stack(z_rows, axis=0), np.array(y_rows, dtype=np.float64)


def collect_lm_cache(
    tok,
    pairs: list[dict],
) -> dict[str, list[torch.Tensor] | list[int]]:
    """Cache neg full sequences + pos assistant token targets for real L_LM.

    For each rollout pair, stores:
      - neg_input_ids: full tokenized neg (system+user+assistant) sequence
      - neg_attention_mask: ones mask matching neg_input_ids
      - prompt_lens: index where assistant span starts in neg sequence
      - pos_assistant_ids: token ids of pos assistant span (CE targets)
    """
    neg_input_ids: list[torch.Tensor] = []
    neg_attention_mask: list[torch.Tensor] = []
    prompt_lens: list[int] = []
    pos_assistant_ids: list[torch.Tensor] = []

    for pi, pair in enumerate(pairs):
        question = str(pair["question"])
        neg_sys = str(pair["neg_system"])
        neg_reply = str(pair["neg_reply"])
        pos_sys = str(pair["pos_system"])
        pos_reply = str(pair["pos_reply"])
        if len(neg_reply.strip()) < 10 or len(pos_reply.strip()) < 10:
            continue
        try:
            msgs_neg = _chat_turns(neg_sys, question, neg_reply)
            raw_neg = tok.apply_chat_template(
                msgs_neg, tokenize=True, add_generation_prompt=False, return_tensors="pt",
            )
            ids_neg = _as_input_ids_tensor(raw_neg, torch.device("cpu")).squeeze(0)
            pl_neg = _prompt_token_len(tok, neg_sys, question)
            if pl_neg >= ids_neg.shape[0]:
                continue

            msgs_pos = _chat_turns(pos_sys, question, pos_reply)
            raw_pos = tok.apply_chat_template(
                msgs_pos, tokenize=True, add_generation_prompt=False, return_tensors="pt",
            )
            ids_pos = _as_input_ids_tensor(raw_pos, torch.device("cpu")).squeeze(0)
            pl_pos = _prompt_token_len(tok, pos_sys, question)
            if pl_pos >= ids_pos.shape[0]:
                continue
            pos_asst = ids_pos[pl_pos:].clone()

            neg_input_ids.append(ids_neg)
            neg_attention_mask.append(torch.ones_like(ids_neg))
            prompt_lens.append(int(pl_neg))
            pos_assistant_ids.append(pos_asst)
        except Exception as exc:
            print(f"  skip lm pair {pi}: {exc}", flush=True)

    if len(neg_input_ids) < 4:
        raise RuntimeError(f"Too few LM cache samples ({len(neg_input_ids)}); need >=4")
    print(f"  LM cache: {len(neg_input_ids)} neg/pos pairs", flush=True)
    return {
        "neg_input_ids": neg_input_ids,
        "neg_attention_mask": neg_attention_mask,
        "prompt_lens": prompt_lens,
        "pos_assistant_ids": pos_assistant_ids,
    }


def _pad_1d_batch(tensors: list[torch.Tensor], pad_val: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad list of 1D tensors to (batch, max_len). Returns (padded, lengths)."""
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    max_len = int(lengths.max())
    batch = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        batch[i, : t.shape[0]] = t
    return batch, lengths


def sae_steer_hook_fn(
    sae,
    v_masked: torch.Tensor,
    prompt_len: int,
):
    """Forward hook: SAE encode -> add v -> decode, replace assistant span."""

    def hook(_module, _inp, output):
        h = _extract_hidden_3d(output)
        if h is None:
            return output
        sae_dev = next(sae.parameters()).device
        x = h.detach().to(sae_dev).float()
        z = sae.encode(x)
        v_b = v_masked.to(z.device).view(1, 1, -1)
        z_prime = z + v_b
        decoded = sae.decode(z_prime).to(device=h.device, dtype=h.dtype)
        h_new = h.clone()
        h_new[:, prompt_len:, :] = decoded[:, prompt_len:, :]
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def compute_real_lm_loss(
    model,
    sae,
    layers,
    layer: int,
    v_masked: torch.Tensor,
    lm_cache: dict,
    batch_indices: list[int],
    device: torch.device,
    max_lm_tokens: int = 64,
) -> torch.Tensor:
    """Cross-entropy of steered neg forward against pos assistant targets."""
    neg_ids_list = [lm_cache["neg_input_ids"][i] for i in batch_indices]
    attn_list = [lm_cache["neg_attention_mask"][i] for i in batch_indices]
    prompt_lens = [lm_cache["prompt_lens"][i] for i in batch_indices]
    pos_targets = [lm_cache["pos_assistant_ids"][i] for i in batch_indices]

    losses = []
    for idx_in_batch, bi in enumerate(batch_indices):
        ids = neg_ids_list[idx_in_batch].unsqueeze(0).to(device)
        attn = attn_list[idx_in_batch].unsqueeze(0).to(device)
        pl = prompt_lens[idx_in_batch]
        pos_tgt = pos_targets[idx_in_batch].to(device)

        neg_asst_len = ids.shape[1] - pl
        tgt_len = pos_tgt.shape[0]
        T = min(neg_asst_len, tgt_len, max_lm_tokens)
        if T <= 0:
            continue

        hook_fn = sae_steer_hook_fn(sae, v_masked, pl)
        handle = layers[layer].register_forward_hook(hook_fn)
        try:
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits.float()
        finally:
            handle.remove()

        logit_slice = logits[0, pl - 1 : pl - 1 + T, :]
        target = pos_tgt[:T]
        losses.append(F.cross_entropy(logit_slice, target))
        del out, logits

    if not losses:
        return torch.tensor(0.0, device=v_masked.device, requires_grad=True)
    return torch.stack(losses).mean()


def optimize_v_steer(
    z_neg_means: torch.Tensor,
    W_dec: torch.Tensor,
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    n_iter: int = 100,
    lr: float = 0.05,
    lambda_lm: float = 0.5,
    beta: float = 0.01,
    beta_auto: float = 0.0,
    opt_device: torch.device,
    log_trajectory: bool = False,
    model=None,
    sae=None,
    layers=None,
    lm_cache: dict | None = None,
    lm_batch_size: int = 1,
    steering_layer: int | None = None,
    use_real_lm: bool = False,
    lm_max_tokens: int = 64,
    normalize_dist: bool = False,
) -> tuple[torch.Tensor, list[dict] | None]:
    """Optimize steering vector v in SAE space using L_steer.

    Pre-encodes neg samples to z_neg_means (n_neg, d_sae) so each iteration
    is just vector math for the distance term.

    L_steer = ||z' - mu+||^2 - ||z' - mu-||^2 + lambda_lm * L_LM + beta * ||v_I||_1

    When use_real_lm=True and model/sae/lm_cache are provided, L_LM is cross-entropy
    through the model after SAE encode-steer-decode at the steering layer (paper).
    Otherwise L_LM = ||W_dec @ v||^2 (legacy proxy).
    """
    d_sae = z_neg_means.shape[1]
    n_neg = z_neg_means.shape[0]
    mask_cpu = feature_mask.float()
    mask = mask_cpu.to(opt_device)

    z_neg = z_neg_means.to(opt_device).float()
    mu_p = mu_pos.to(opt_device).float()
    mu_n = mu_neg.to(opt_device).float()

    W_dec_I = W_dec[mask_cpu.bool()].to(opt_device).float()  # (k, d_model)

    real_lm = (
        use_real_lm
        and model is not None
        and sae is not None
        and layers is not None
        and lm_cache is not None
        and steering_layer is not None
    )
    n_lm_samples = len(lm_cache["neg_input_ids"]) if lm_cache else 0
    lm_batch_size = min(lm_batch_size, max(n_lm_samples, 1))

    if real_lm:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    delta = (mu_p - mu_n) * mask
    delta_norm = delta.norm()
    if delta_norm > 1e-8:
        delta = delta / delta_norm
    v = delta.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([v], lr=lr)

    lm_mode = "real CE" if real_lm else "proxy ||W_dec v||^2"
    dist_reduce = "mean" if normalize_dist else "sum"
    print(
        f"  Optimizing v: {int(mask.sum())} active dims, "
        f"{n_neg} neg samples, {n_iter} iters (device={opt_device}), "
        f"L_LM={lm_mode}, dist_reduce={dist_reduce}",
        flush=True,
    )

    trajectory: list[dict] | None = [] if log_trajectory else None
    rng = np.random.RandomState(42)
    effective_lambda_lm = lambda_lm
    effective_beta = beta

    for it in range(n_iter):
        optimizer.zero_grad()

        v_masked = v * mask  # (d_sae,)

        z_prime = z_neg + v_masked.unsqueeze(0)  # (n_neg, d_sae)

        sq_pos = (z_prime - mu_p.unsqueeze(0)).pow(2)
        sq_neg = (z_prime - mu_n.unsqueeze(0)).pow(2)
        if normalize_dist:
            dist_pos = sq_pos.mean(dim=1).mean()
            dist_neg = sq_neg.mean(dim=1).mean()
        else:
            dist_pos = sq_pos.sum(dim=1).mean()
            dist_neg = sq_neg.sum(dim=1).mean()
        dist_loss = dist_pos - dist_neg

        if real_lm:
            batch_idx = rng.choice(n_lm_samples, size=lm_batch_size, replace=False).tolist()
            lm_loss = compute_real_lm_loss(
                model, sae, layers, steering_layer, v_masked,
                lm_cache, batch_idx, opt_device, max_lm_tokens=lm_max_tokens,
            )
            if it == 0:
                with torch.no_grad():
                    dist_mag = abs(dist_loss.item())
                    lm_mag = abs(lm_loss.item())
                    if normalize_dist:
                        print(
                            f"    normalized dist: dist={dist_mag:.4f}, lm={lm_mag:.4f}, "
                            f"lambda_lm={lambda_lm}, lm_eff={lambda_lm * lm_mag:.4f}",
                            flush=True,
                        )
                    elif lm_mag > 1e-8:
                        effective_lambda_lm = lambda_lm * (dist_mag / lm_mag)
                        print(
                            f"    auto-scaled lambda_lm: {lambda_lm} -> {effective_lambda_lm:.2f} "
                            f"(dist={dist_mag:.1f}, lm={lm_mag:.1f}, ratio={dist_mag/lm_mag:.0f})",
                            flush=True,
                        )
            if (it + 1) % 20 == 0:
                torch.cuda.empty_cache()
        else:
            v_active = v_masked[mask.bool()]  # (k,)
            lm_loss = (W_dec_I.T @ v_active).pow(2).sum()

        sparsity = v_masked.abs().sum()

        if beta_auto > 0 and it == 0:
            with torch.no_grad():
                dist_mag = abs(dist_loss.item())
                sparsity_mag = sparsity.item()
                if sparsity_mag > 1e-8:
                    effective_beta = beta_auto * dist_mag / sparsity_mag
                    print(
                        f"    auto-scaled beta: {beta} -> {effective_beta:.4f} "
                        f"(target_ratio={beta_auto}, dist={dist_mag:.1f}, "
                        f"sparsity={sparsity_mag:.1f})",
                        flush=True,
                    )

        sparsity_loss = effective_beta * sparsity

        loss = dist_loss + effective_lambda_lm * lm_loss + sparsity_loss

        loss.backward()

        with torch.no_grad():
            if v.grad is not None:
                v.grad *= mask

        optimizer.step()

        with torch.no_grad():
            v.data *= mask

        if log_trajectory:
            with torch.no_grad():
                w = v_masked[mask.bool()].cpu().tolist()
                trajectory.append({
                    "iter": it,
                    "loss": round(float(loss), 5),
                    "dist_loss": round(float(dist_loss), 5),
                    "lm_loss": round(float(effective_lambda_lm * lm_loss), 5),
                    "sparsity_loss": round(float(sparsity_loss), 5),
                    "weights": [round(x, 5) for x in w],
                    "n_active": int((v_masked.abs() > 1e-8).sum()),
                })

        if (it + 1) % 20 == 0 or it == 0:
            print(
                f"    iter {it+1}/{n_iter}  loss={loss.item():.4f}  "
                f"dist={dist_loss.item():.4f}  "
                f"lm={lm_loss.item():.4f}  "
                f"lm_eff={effective_lambda_lm * lm_loss.item():.4f}  "
                f"sparsity={effective_beta * sparsity.item():.4f}  "
                f"v_nnz={int((v_masked.abs() > 1e-8).sum())}",
                flush=True,
            )

    return v.detach() * mask, trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--sae-id", default=None)
    ap.add_argument("--alpha", type=float, default=None, help="Steering alpha (default: from validation_report.json)")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--ks", default="5,10,20,50,100,128,200,256,512,750,1000")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lambda-lm", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--z-cache", default=None)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--optimize-only", action="store_true", help="Only run SSV optimization; skip generation/judging")
    ap.add_argument("--log-trajectory", action="store_true", help="Log per-iteration weights and loss for visualization; saves ssv_trajectory_<trait>_K<k>.json alongside --out")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--lm-loss-real", action="store_true", help="Use paper L_LM (CE through model) instead of ||W_dec @ v||^2 proxy")
    ap.add_argument("--lm-cache", default=None, help="Path to lm_loss_cache .pt (neg hiddens + pos targets)")
    ap.add_argument("--lm-batch-size", type=int, default=1, help="Mini-batch size for real L_LM per optim step (default 1 for T4 memory)")
    ap.add_argument("--lm-max-tokens", type=int, default=64, help="Max assistant tokens for real L_LM CE (memory cap)")
    ap.add_argument("--normalize-dist", action="store_true", help="Divide distance loss by d_sae (per-dim MSE). Matches paper's scale when using large SAEs.")
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.layer is not None:
        cfg["layer"] = args.layer
        from scripts.trait_sae_config import hidden_state_index, run_paths, sae_id_for_layer
        cfg["sae_id"] = args.sae_id or sae_id_for_layer(cfg["layer"])
        cfg["hs_index"] = hidden_state_index(cfg["layer"])
        cfg.update(run_paths(cfg["run_id"], cfg["layer"]))

    if args.alpha is None:
        args.alpha = float(cfg["alpha"])
    print(
        f"Steering alpha={args.alpha:.1f} layer={cfg['layer']} "
        f"(from {cfg.get('alpha_source', 'validate')})",
        flush=True,
    )

    layer = int(cfg["layer"])
    bundle_path = Path(args.bundle or cfg["bundle"])
    vectors_path = Path(args.vectors or cfg["vectors"])
    rollouts_path = Path(args.rollouts or (cfg["base"] / "rollouts" / "rollouts.jsonl"))
    out_path = Path(args.out or (cfg["sae_dir"] / f"sae_ssv_results_262k_l{layer}.json"))
    z_cache = Path(args.z_cache) if args.z_cache else (cfg["sae_dir"] / f"probe_z_cache_l{layer}.npz")
    lm_cache_path = Path(args.lm_cache) if args.lm_cache else (cfg["sae_dir"] / f"lm_loss_cache_l{layer}.pt")
    sae_id = args.sae_id or cfg["sae_id"]
    hs_index = cfg["hs_index"]
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})

    bundle = PersonaTraitArtifact.model_validate_json(bundle_path.read_text())
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    neg_sys = bundle.neg_system_prompt
    eval_qs = bundle.eval_questions[: args.n_questions]

    v_layer = None
    if vectors_path.exists():
        v_full = torch.load(vectors_path, map_location="cpu", weights_only=False)["v"]
        v_layer = v_full[layer].float()

    print(
        f"=== SAE-SSV optimize  trait={cfg['trait']} run={cfg['run_id']} "
        f"layer={layer} n_iter={args.n_iter} lr={args.lr} "
        f"lm_loss={'real' if args.lm_loss_real else 'proxy'} "
        f"normalize_dist={args.normalize_dist} ===",
        flush=True,
    )

    need_model = args.lm_loss_real or not (
        args.optimize_only and args.skip_collect and z_cache.exists()
    )
    model = tok = None
    layers = pad_id = dtype = dev = None

    if need_model:
        model, tok, dev = load_model_and_tokenizer()
        layers = _language_model_layers(model)
        pad_id = tok.pad_token_id or tok.eos_token_id
        dtype = next(model.parameters()).dtype
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Skipping model load (optimize-only + z cache present)", flush=True)

    opt_dev = dev if dev is not None and dev.type == "cuda" else torch.device("cpu")
    # SAE stays on CPU (262k SAE ~= 2.5GB); hook moves activations CPU<->GPU per forward.
    sae_dev = torch.device("cpu")
    sae, _ = load_sae_for_layer(
        sae_dev, release=SAE_RELEASE, sae_id=sae_id, hidden_state_index=hs_index,
    )
    W_dec = sae.W_dec.detach().float().cpu()
    d_sae = int(sae.cfg.d_sae)

    pairs = load_rollout_question_pairs(rollouts_path, bundle_path) if need_model else []

    if args.skip_collect and z_cache.exists():
        cached = np.load(z_cache)
        z_all, y_all = cached["z"], cached["y"]
        print(f"Loaded z cache: {z_all.shape[0]} samples", flush=True)
    else:
        if model is None:
            raise RuntimeError("z cache missing and model not loaded; run without --skip-collect first")
        print(f"Collecting latents from {len(pairs)} pairs...", flush=True)
        z_all, y_all = collect_latents(model, tok, dev, sae, layer, pairs)
        z_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(z_cache, z=z_all, y=y_all)
        print(f"Saved z cache ({z_all.shape[0]} samples)", flush=True)

    lm_cache = None
    if args.lm_loss_real:
        if model is None or tok is None:
            raise RuntimeError("--lm-loss-real requires model loaded")
        if args.skip_collect and lm_cache_path.exists():
            lm_cache = torch.load(lm_cache_path, map_location="cpu", weights_only=False)
            print(f"Loaded LM cache: {len(lm_cache['neg_input_ids'])} samples", flush=True)
        else:
            print("Collecting LM cache (neg sequences + pos targets)...", flush=True)
            lm_cache = collect_lm_cache(tok, pairs)
            lm_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lm_cache, lm_cache_path)
            print(f"Saved LM cache: {lm_cache_path}", flush=True)

    print("Computing F-statistics...", flush=True)
    f_stats = f_statistic_per_feature(z_all, y_all)

    mask_pos = y_all > 0.5
    mu_pos = torch.from_numpy(z_all[mask_pos].mean(axis=0)).float()
    mu_neg = torch.from_numpy(z_all[~mask_pos].mean(axis=0)).float()
    z_neg_means = torch.from_numpy(z_all[~mask_pos]).float()

    print(f"Optimization device: {opt_dev}, neg samples: {z_neg_means.shape[0]}", flush=True)

    steer_norm = float(v_layer.norm()) if v_layer is not None else 1.0

    def _optim_kwargs():
        return {
            "n_iter": args.n_iter,
            "lr": args.lr,
            "lambda_lm": args.lambda_lm,
            "beta": args.beta,
            "opt_device": opt_dev,
            "log_trajectory": args.log_trajectory,
            "model": model,
            "sae": sae,
            "layers": layers,
            "lm_cache": lm_cache,
            "lm_batch_size": args.lm_batch_size,
            "steering_layer": layer,
            "use_real_lm": args.lm_loss_real,
            "lm_max_tokens": args.lm_max_tokens,
            "normalize_dist": args.normalize_dist,
        }

    if args.optimize_only:
        # --- SAE-SSV optimization only (no model generation) ---
        results = []
        for k in ks:
            k = min(k, d_sae)
            print(f"\n=== SAE-SSV K={k} ===", flush=True)
            top_k_idx = np.argsort(f_stats)[::-1][:k].copy()
            feature_mask = torch.zeros(d_sae, dtype=torch.float32)
            feature_mask[top_k_idx] = 1.0
            v_opt, traj = optimize_v_steer(
                z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
                **_optim_kwargs(),
            )
            v_residual = (W_dec.T @ v_opt.cpu().float()).float()
            raw_norm = float(v_residual.norm())
            if raw_norm > 1e-8:
                v_residual = v_residual * (steer_norm / raw_norm)
            active_mask = v_opt.abs() > 1e-8
            active_fids = int(active_mask.sum())
            top_active = torch.argsort(v_opt.abs(), descending=True)
            top_fids = [int(f) for f in top_active if active_mask[f]]
            top_weights = [round(float(v_opt[f]), 6) for f in top_active if active_mask[f]]
            meta = {
                "k": k,
                "method": "sae_ssv",
                "n_active_features": active_fids,
                "feature_ids": top_fids,
                "feature_weights": top_weights,
                "norm_raw": round(raw_norm, 4),
                "norm_after_scale": round(float(v_residual.norm()), 4),
                "steer_norm": round(steer_norm, 4),
            }
            if v_layer is not None:
                cos = F.cosine_similarity(v_layer.unsqueeze(0), v_residual.unsqueeze(0)).item()
                meta["cosine_vs_dense"] = round(cos, 4)
            if traj is not None:
                traj_path = out_path.parent / f"ssv_trajectory_{cfg['trait']}_K{k}.json"
                traj_out = {
                    "trait": cfg["trait"],
                    "k": k,
                    "feature_ids": top_k_idx.tolist(),
                    "n_iter": args.n_iter,
                    "iterations": traj,
                    "final": {
                        "feature_ids": top_fids,
                        "feature_weights": top_weights,
                        "cosine_vs_dense": meta.get("cosine_vs_dense"),
                    },
                }
                traj_path.write_text(json.dumps(traj_out))
                print(f"  Saved trajectory: {traj_path}", flush=True)
            print(
                f"  Optimized: {active_fids} active features, saved {len(top_fids)} ids",
                flush=True,
            )
            results.append({"label": f"SSV_K{k}", "mean": None, "scores": [], **meta})

        payload = {
            "method": "sae_ssv",
            "trait": cfg["trait"],
            "run_id": cfg["run_id"],
            "layer": layer,
            "sae_id": sae_id,
            "optim": {
                "n_iter": args.n_iter,
                "lr": args.lr,
                "lambda_lm": args.lambda_lm,
                "beta": args.beta,
                "lm_loss_real": args.lm_loss_real,
                "lm_batch_size": args.lm_batch_size,
                "lm_max_tokens": args.lm_max_tokens,
                "normalize_dist": args.normalize_dist,
            },
            "data": {
                "n_samples": int(z_all.shape[0]),
                "n_pos": int((y_all > 0.5).sum()),
                "n_neg": int((y_all <= 0.5).sum()),
            },
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved {out_path}", flush=True)
        return

    # --- Generation & judging helpers ---

    def gen_text(ids, attn, hook_fn=None):
        handle = None
        if hook_fn is not None:
            handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=pad_id, use_cache=True,
            )
        if handle is not None:
            handle.remove()
        return tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def judge_reply(prompt, reply):
        if args.skip_judge or len(reply.strip()) < 20:
            return None
        try:
            js = score_transcript(judge_instr, neg_sys, prompt, reply)
            return int(js.score)
        except Exception as exc:
            print(f"  judge error: {exc}", flush=True)
            return None

    def run_condition_sae(label, v_opt, qs, meta=None):
        """Generate with SAE encode+steer+decode hook — steer all positions."""
        v_masked = v_opt.to(device=dev).float()
        scores, samples = [], []
        for qi, prompt in enumerate(qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook_fn = sae_steer_hook_fn(sae, v_masked, prompt_len=0)
            reply = gen_text(ids, attn, hook_fn)
            s = judge_reply(prompt, reply)
            scores.append(s)
            samples.append({"q_idx": qi, "score": s, "reply": reply[:300]})
            print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean} (SAE hook eval)", flush=True)
        row = {"label": label, "mean": mean, "scores": scores, "eval_mode": "sae_hook", "samples": samples}
        if meta:
            row.update(meta)
        return row

    def run_condition(label, direction, alpha_eff, qs, meta=None):
        d = direction.to(device=dev, dtype=dtype).view(1, 1, -1)
        scores, samples = [], []
        for qi, prompt in enumerate(qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            hook = _steering_hook_fn(alpha_eff, d, steer_last_token_only=False, hook_calls=[0])
            reply = gen_text(ids, attn, hook)
            s = judge_reply(prompt, reply)
            scores.append(s)
            samples.append({"q_idx": qi, "score": s, "reply": reply[:300]})
            print(f"  [{label}] Q{qi+1} score={s}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = round(sum(valid) / len(valid), 1) if valid else None
        print(f"  [{label}] MEAN={mean} alpha_eff={alpha_eff:.3f}", flush=True)
        row = {"label": label, "mean": mean, "scores": scores, "alpha_effective": round(alpha_eff, 4), "samples": samples}
        if meta:
            row.update(meta)
        return row

    results = []

    # --- Baseline & Dense CAA ---
    if not args.skip_ref:
        print("=== BASELINE ===", flush=True)
        base_scores = []
        for qi, prompt in enumerate(eval_qs):
            msgs = [{"role": "system", "content": neg_sys}, {"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(dev)
            attn = torch.ones_like(ids)
            reply = gen_text(ids, attn)
            s = judge_reply(prompt, reply)
            base_scores.append(s)
            print(f"  [BASELINE] Q{qi+1} score={s}", flush=True)
        valid = [s for s in base_scores if s is not None]
        results.append({"label": "BASELINE", "mean": round(sum(valid)/len(valid), 1) if valid else None, "scores": base_scores})

        if v_layer is not None:
            print("=== DENSE_CAA ===", flush=True)
            results.append(run_condition("DENSE_CAA", v_layer, args.alpha, eval_qs))

    # --- SAE-SSV optimization at each K ---
    for k in ks:
        k = min(k, d_sae)
        print(f"\n=== SAE-SSV K={k} ===", flush=True)

        top_k_idx = np.argsort(f_stats)[::-1][:k].copy()
        feature_mask = torch.zeros(d_sae, dtype=torch.float32)
        feature_mask[top_k_idx] = 1.0

        v_opt, traj = optimize_v_steer(
            z_neg_means, W_dec, mu_pos, mu_neg, feature_mask,
            **_optim_kwargs(),
        )
        if traj is not None:
            traj_path = out_path.parent / f"ssv_trajectory_{cfg['trait']}_K{k}.json"
            traj_out = {
                "trait": cfg["trait"],
                "k": k,
                "feature_ids": top_k_idx.tolist(),
                "n_iter": args.n_iter,
                "iterations": traj,
                "final": {
                    "feature_ids": None,
                    "feature_weights": None,
                    "cosine_vs_dense": None,
                },
            }
            traj_path.write_text(json.dumps(traj_out))
            print(f"  Saved trajectory: {traj_path}", flush=True)

        v_residual = (W_dec.T @ v_opt.cpu().float()).float()
        raw_norm = float(v_residual.norm())
        if raw_norm > 1e-8:
            v_residual = v_residual * (steer_norm / raw_norm)

        active_mask = v_opt.abs() > 1e-8
        active_fids = int(active_mask.sum())
        top_active = torch.argsort(v_opt.abs(), descending=True)
        top_fids = [int(f) for f in top_active if active_mask[f]]
        top_weights = [round(float(v_opt[f]), 6) for f in top_active if active_mask[f]]

        meta = {
            "k": k,
            "method": "sae_ssv",
            "n_active_features": active_fids,
            "feature_ids": top_fids,
            "feature_weights": top_weights,
            "norm_raw": round(raw_norm, 4),
            "norm_after_scale": round(float(v_residual.norm()), 4),
            "steer_norm": round(steer_norm, 4),
        }
        if v_layer is not None:
            cos = F.cosine_similarity(v_layer.unsqueeze(0), v_residual.unsqueeze(0)).item()
            meta["cosine_vs_dense"] = round(cos, 4)

        label = f"SSV_K{k}"
        print(
            f"  Optimized: {active_fids} active features, norm={raw_norm:.4f}, "
            f"cos_dense={meta.get('cosine_vs_dense', 'N/A')}",
            flush=True,
        )
        if args.optimize_only:
            row = {"label": label, "mean": None, "scores": [], "alpha_effective": args.alpha, **meta}
            results.append(row)
        elif args.lm_loss_real:
            row = run_condition_sae(label, v_opt.to(device=dev).float(), eval_qs, meta=meta)
            results.append(row)
        else:
            row = run_condition(label, v_residual, args.alpha, eval_qs, meta=meta)
            results.append(row)

    # --- Save ---
    payload = {
        "method": "sae_ssv",
        "trait": cfg["trait"],
        "run_id": cfg["run_id"],
        "layer": layer,
        "sae_id": sae_id,
        "alpha_base": args.alpha,
        "n_questions": len(eval_qs),
        "optim": {
            "n_iter": args.n_iter,
            "lr": args.lr,
            "lambda_lm": args.lambda_lm,
            "beta": args.beta,
            "lm_loss_real": args.lm_loss_real,
            "lm_batch_size": args.lm_batch_size,
            "lm_max_tokens": args.lm_max_tokens,
            "normalize_dist": args.normalize_dist,
        },
        "data": {
            "n_samples": int(z_all.shape[0]),
            "n_pos": int((y_all > 0.5).sum()),
            "n_neg": int((y_all <= 0.5).sum()),
        },
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        extra = ""
        if "k" in r:
            extra = f" k={r['k']} cos={r.get('cosine_vs_dense', 'N/A')}"
        print(f"  {r['label']:>12s}  mean={r['mean']}  scores={r['scores']}{extra}")
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
