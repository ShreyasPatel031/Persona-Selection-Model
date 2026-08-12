#!/usr/bin/env python3
"""
Logit lens for SSV-selected 262k SAE features (decoder column -> lm_head).

Builds layer-level caches shared across traits on the same layer, then writes
per-trait feature lens JSON and optional combined-vector lens at selected K.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHARED_DIR = REPO / "persona_runs" / "_shared"
TOP_K = 8
SAVE_EVERY = 100
VECTOR_KS = (100, 512, 1000)

TRAIT_SSV: dict[str, Path] = {
    "good": REPO / "persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json",
    "evil": REPO / "persona_runs/dnd_evil/sae/sae_ssv_results_262k_l16.json",
    "lawful": REPO / "persona_runs/dnd_lawful/sae/sae_ssv_results_262k_l15.json",
    "chaotic": REPO / "persona_runs/dnd_chaotic/sae/sae_ssv_results_262k_l15.json",
}


def cache_path(layer: int) -> Path:
    return SHARED_DIR / f"l{layer}_262k_logit_lens_cache.json"


def trait_lens_path(trait: str, layer: int) -> Path:
    cfg = resolve_trait(trait)
    return cfg["sae_dir"] / f"ssv_feature_logit_lens_262k_l{layer}.json"


def vector_lens_path(trait: str, layer: int) -> Path:
    cfg = resolve_trait(trait)
    return cfg["sae_dir"] / f"ssv_vector_logit_lens_262k_l{layer}.json"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def collect_fids_from_ssv(ssv_path: Path) -> set[int]:
    doc = json.loads(ssv_path.read_text(encoding="utf-8"))
    fids: set[int] = set()
    for row in doc.get("results", []):
        if "k" not in row:
            continue
        fids.update(int(f) for f in row.get("feature_ids", []))
    return fids


def omp_sweep_path(trait: str, layer: int) -> Path:
    cfg = resolve_trait(trait)
    return cfg["sae_dir"] / f"ssv_omp_dsweep_l{layer}.json"


def omp_lens_path(trait: str, layer: int) -> Path:
    cfg = resolve_trait(trait)
    return cfg["sae_dir"] / f"ssv_omp_feature_logit_lens_262k_l{layer}.json"


def collect_fids_from_omp(trait: str, *, layer: int | None = None) -> set[int]:
    lyr = layer if layer is not None else int(resolve_trait(trait)["layer"])
    path = omp_sweep_path(trait, lyr)
    if not path.is_file():
        raise FileNotFoundError(f"Missing OMP d-sweep for trait {trait}: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    fids: set[int] = set()
    for row in doc.get("results", []):
        if row.get("d") is None:
            continue
        fids.update(int(f) for f in row.get("feature_ids", []))
    omp_list = doc.get("omp_features") or []
    fids.update(int(f) for f in omp_list)
    return fids


def collect_fids_from_omp_traits(traits: list[str], *, layer: int) -> set[int]:
    all_fids: set[int] = set()
    for trait in traits:
        all_fids.update(collect_fids_from_omp(trait, layer=layer))
    return all_fids


def collect_fids_from_decomp(trait: str, *, layer: int, top_k: int) -> set[int]:
    """Top-K feature IDs from OMP decomposition by |coefficient|."""
    cfg = resolve_trait(trait)
    path = cfg["sae_dir"] / f"omp_decomposition_262k_l{layer}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OMP decomposition for {trait}: {path}")
    rows = sorted(
        json.loads(path.read_text(encoding="utf-8")).get("decomposition") or [],
        key=lambda r: abs(float(r.get("coefficient", 0))),
        reverse=True,
    )[:top_k]
    return {int(r["feature_id"]) for r in rows}


def collect_fids_from_decomp_traits(traits: list[str], *, layer: int, top_k: int) -> set[int]:
    all_fids: set[int] = set()
    for trait in traits:
        all_fids.update(collect_fids_from_decomp(trait, layer=layer, top_k=top_k))
    return all_fids


def collect_fids_for_traits(traits: list[str]) -> tuple[set[int], dict[str, Path]]:
    all_fids: set[int] = set()
    ssv_paths: dict[str, Path] = {}
    for trait in traits:
        path = TRAIT_SSV.get(trait)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing SSV for trait {trait}: {path}")
        ssv_paths[trait] = path
        all_fids.update(collect_fids_from_ssv(path))
    return all_fids, ssv_paths


def load_k_row(ssv_path: Path, k: int) -> tuple[list[int], list[float]]:
    doc = json.loads(ssv_path.read_text(encoding="utf-8"))
    for row in doc.get("results", []):
        if int(row.get("k", -1)) == k:
            fids = [int(f) for f in row.get("feature_ids", [])]
            wts = [float(w) for w in row.get("feature_weights", [])]
            return fids, wts
    raise KeyError(f"K={k} not in {ssv_path}")


def _token_pairs(tokenizer, idx: torch.Tensor, vals: torch.Tensor) -> list[list]:
    out: list[list] = []
    for j, v in zip(idx.tolist(), vals.tolist()):
        tok = tokenizer.decode([int(j)]).strip()
        out.append([tok, round(float(v), 3)])
    return out


def effective_lm_head(model) -> torch.Tensor:
    """Fold final RMSNorm gain into unembedding (Gemma Scope 2 tutorial)."""
    raw = model.lm_head.weight.detach().float()
    norm_w = None
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "norm"):
            norm_w = inner.language_model.norm.weight.detach().float()
        elif hasattr(inner, "norm"):
            norm_w = inner.norm.weight.detach().float()
    if norm_w is None:
        logger.warning("Could not find final RMSNorm; using raw lm_head")
        return raw
    return raw * (1.0 + norm_w)


def compute_lens_for_fid(
    fid: int,
    W_dec: torch.Tensor,
    lm_head: torch.Tensor,
    tokenizer,
    top_k: int = TOP_K,
) -> dict:
    logits = lm_head @ W_dec[fid]
    top_vals, top_idx = torch.topk(logits, top_k)
    bot_vals, bot_idx = torch.topk(-logits, top_k)
    return {
        "fid": fid,
        "top_tokens": _token_pairs(tokenizer, top_idx, top_vals),
        "top_suppress": _token_pairs(tokenizer, bot_idx, bot_vals),
    }


def seed_cache_from_legacy_good(cache: dict[str, dict]) -> dict[str, dict]:
    legacy = REPO / "persona_runs/dnd_good_scale/sae/ssv_k100_feature_logit_lens.json"
    if not legacy.is_file():
        return cache
    rows = json.loads(legacy.read_text(encoding="utf-8"))
    for row in rows:
        fid = str(int(row["fid"]))
        if fid in cache:
            continue
        cache[fid] = {
            "fid": int(row["fid"]),
            "top_tokens": row.get("top_tokens", [])[:TOP_K],
            "top_suppress": row.get("top_suppress") or row.get("bot_tokens", [])[:TOP_K],
        }
    return cache


def ensure_lens_cache(
    layer: int,
    fids: set[int],
    *,
    force: bool = False,
) -> dict[str, dict]:
    path = cache_path(layer)
    cache = {} if force else load_cache(path)
    if layer == 16 and not force:
        cache = seed_cache_from_legacy_good(cache)
    missing = sorted(fid for fid in fids if force or str(fid) not in cache)
    if not missing:
        logger.info("Layer %d cache complete (%d features)", layer, len(cache))
        return cache

    logger.info("Loading model + L%d 262k SAE on CPU (%d fids to compute)...", layer, len(missing))
    cpu = torch.device("cpu")
    model, tokenizer, _ = load_model_and_tokenizer(None, device=cpu)
    sae_id = f"layer_{layer}_width_262k_l0_small"
    sae, _ = load_sae_for_layer(
        cpu,
        release=SAE_RELEASE,
        sae_id=sae_id,
        hidden_state_index=layer + 1,
    )
    W_dec = sae.W_dec.detach().float()
    lm_head = effective_lm_head(model)
    del model, sae

    for i, fid in enumerate(missing):
        row = compute_lens_for_fid(fid, W_dec, lm_head, tokenizer)
        cache[str(fid)] = row
        if (i + 1) % SAVE_EVERY == 0 or i + 1 == len(missing):
            logger.info("  layer %d: %d/%d", layer, i + 1, len(missing))
            save_cache(path, cache)

    save_cache(path, cache)
    logger.info("Wrote cache %s (%d features)", path, len(cache))
    return cache


def export_decomp_trait_lens(
    trait: str,
    cache: dict[str, dict],
    *,
    layer: int,
    top_k: int,
) -> Path:
    fids = sorted(collect_fids_from_decomp(trait, layer=layer, top_k=top_k))
    rows_out: list[dict] = []
    for fid in fids:
        entry = cache.get(str(fid))
        if not entry:
            continue
        rows_out.append(
            {
                "fid": fid,
                "top_tokens": entry.get("top_tokens", []),
                "top_suppress": entry.get("top_suppress", entry.get("bot_tokens", [])),
            }
        )
    out = omp_lens_path(trait, layer)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d decomp top-%d features)", out, len(rows_out), top_k)
    return out


def export_omp_trait_lens(
    trait: str,
    cache: dict[str, dict],
    *,
    layer: int,
) -> Path:
    fids = sorted(collect_fids_from_omp(trait, layer=layer))
    rows_out: list[dict] = []
    for fid in fids:
        entry = cache.get(str(fid))
        if not entry:
            continue
        rows_out.append(
            {
                "fid": fid,
                "top_tokens": entry.get("top_tokens", []),
                "top_suppress": entry.get("top_suppress", entry.get("bot_tokens", [])),
            }
        )
    out = omp_lens_path(trait, layer)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d OMP features)", out, len(rows_out))
    return out


def export_trait_lens(
    trait: str,
    ssv_path: Path,
    cache: dict[str, dict],
    *,
    layer: int,
) -> Path:
    fids = sorted(collect_fids_from_ssv(ssv_path))
    doc = json.loads(ssv_path.read_text(encoding="utf-8"))
    rows = [r for r in doc.get("results", []) if "k" in r]
    ref = next((r for r in rows if int(r["k"]) == 100), max(rows, key=lambda r: int(r["k"])))
    ref_fids = [int(f) for f in ref.get("feature_ids", [])]
    ref_wts = [float(w) for w in ref.get("feature_weights", [])]
    rank_map = {fid: i + 1 for i, fid in enumerate(ref_fids)}
    wt_map = dict(zip(ref_fids, ref_wts))

    rows_out: list[dict] = []
    for fid in fids:
        entry = cache.get(str(fid))
        if not entry:
            continue
        w = wt_map.get(fid)
        rows_out.append(
            {
                "fid": fid,
                "rank": rank_map.get(fid),
                "weight": round(w, 4) if w is not None else None,
                "sign": "+" if w is not None and w > 0 else ("-" if w is not None else None),
                "top_tokens": entry.get("top_tokens", []),
                "top_suppress": entry.get("top_suppress", entry.get("bot_tokens", [])),
            }
        )

    out = trait_lens_path(trait, layer)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d features)", out, len(rows_out))
    return out


def export_vector_lens(
    trait: str,
    ssv_path: Path,
    *,
    layer: int,
    ks: tuple[int, ...] = VECTOR_KS,
) -> Path:
    logger.info("Vector logit lens for %s at K=%s", trait, ks)
    cpu = torch.device("cpu")
    model, tokenizer, _ = load_model_and_tokenizer(None, device=cpu)
    sae_id = f"layer_{layer}_width_262k_l0_small"
    sae, _ = load_sae_for_layer(
        cpu,
        release=SAE_RELEASE,
        sae_id=sae_id,
        hidden_state_index=layer + 1,
    )
    W_dec = sae.W_dec.detach().float()
    lm_head = effective_lm_head(model)
    del model, sae

    results: list[dict] = []
    for k in ks:
        try:
            fids, wts = load_k_row(ssv_path, k)
        except KeyError:
            logger.warning("Skip vector K=%d for %s (not in SSV)", k, trait)
            continue
        cols = W_dec[fids]
        w = torch.tensor(wts, dtype=torch.float32)
        v = (cols.T @ w).float()
        logits = lm_head @ v
        top_vals, top_idx = torch.topk(logits, 20)
        bot_vals, bot_idx = torch.topk(-logits, 20)
        results.append(
            {
                "k": k,
                "n_features": len(fids),
                "vector_norm": round(float(v.norm().item()), 4),
                "top_promote": _token_pairs(tokenizer, top_idx, top_vals),
                "top_suppress": _token_pairs(tokenizer, bot_idx, bot_vals),
            }
        )

    out = vector_lens_path(trait, layer)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"trait": trait, "layer": layer, "results": results}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return out


def parse_traits(s: str) -> list[str]:
    return [t.strip().lower() for t in s.split(",") if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="SSV feature logit lens (262k residual SAE)")
    ap.add_argument("--layer", type=int, choices=[15, 16], help="SAE layer (15 or 16)")
    ap.add_argument("--traits", type=str, help="Comma-separated traits (good,evil,...)")
    ap.add_argument("--fids-from", type=str, dest="fids_from", help="Alias: traits whose SSV fids to cache")
    ap.add_argument(
        "--omp-traits",
        type=str,
        help="Comma-separated traits: cache + export logit lens for OMP d-sweep feature IDs",
    )
    ap.add_argument(
        "--decomp-top-k",
        type=int,
        default=0,
        help="With --decomp-traits: top-K features from omp_decomposition per trait",
    )
    ap.add_argument(
        "--decomp-traits",
        type=str,
        help="Comma-separated traits: cache logit lens for top-K OMP decomposition features",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="Recompute cache entries")
    ap.add_argument("--skip-cache", action="store_true", help="Only export from existing cache")
    ap.add_argument("--vector-only", action="store_true", help="Only run combined vector lens")
    ap.add_argument("--vector-ks", default="100,512,1000", help="K levels for vector lens")
    args = ap.parse_args()

    omp_traits = parse_traits(args.omp_traits) if args.omp_traits else []
    decomp_traits = parse_traits(args.decomp_traits) if args.decomp_traits else []
    trait_str = args.traits or args.fids_from
    traits = parse_traits(trait_str) if trait_str else []

    if not traits and not omp_traits and not decomp_traits:
        ap.error("Provide --traits, --fids-from, --omp-traits, or --decomp-traits")
    if decomp_traits and args.decomp_top_k <= 0:
        ap.error("--decomp-top-k required with --decomp-traits")

    all_trait_keys = list(dict.fromkeys(traits + omp_traits + decomp_traits))
    layers = {resolve_trait(t)["layer"] for t in all_trait_keys}
    if len(layers) != 1:
        ap.error(f"Traits must share one layer; got {layers}")
    layer = args.layer if args.layer is not None else layers.pop()
    for t in all_trait_keys:
        if resolve_trait(t)["layer"] != layer:
            ap.error(f"Trait {t} is L{resolve_trait(t)['layer']}, expected L{layer}")

    all_fids: set[int] = set()
    ssv_paths: dict[str, Path] = {}
    if traits:
        all_fids_ssv, ssv_paths = collect_fids_for_traits(traits)
        all_fids.update(all_fids_ssv)
    if omp_traits:
        all_fids.update(collect_fids_from_omp_traits(omp_traits, layer=layer))
    if decomp_traits:
        all_fids.update(
            collect_fids_from_decomp_traits(decomp_traits, layer=layer, top_k=args.decomp_top_k)
        )

    vector_ks = tuple(int(x) for x in args.vector_ks.split(",") if x.strip())

    if not args.vector_only:
        if args.skip_cache:
            cache = load_cache(cache_path(layer))
        else:
            cache = ensure_lens_cache(layer, all_fids, force=args.force)
        for trait in traits:
            export_trait_lens(trait, ssv_paths[trait], cache, layer=layer)
        for trait in omp_traits:
            export_omp_trait_lens(trait, cache, layer=layer)
        for trait in decomp_traits:
            export_decomp_trait_lens(trait, cache, layer=layer, top_k=args.decomp_top_k)

    if (traits and not args.skip_cache) and (args.vector_only or not omp_traits):
        for trait in traits:
            export_vector_lens(
                trait,
                ssv_paths[trait],
                layer=layer,
                ks=vector_ks,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
