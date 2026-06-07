#!/usr/bin/env python3
"""
Corpus-based auto-interpretation for SSV-selected SAE features.

Streams a public corpus (default: monology/pile-uncopyrighted), caches top
activating token contexts per feature, generates Gemini explanations, and
scores them with a lightweight detection test.

This is the lightweight alternative to delphi when Neuronpedia has no data for
our L16 residual 262k SAE.
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, resolve_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SSV = REPO / "persona_runs/dnd_good_scale/sae/sae_ssv_full_sweep_262k_l16.json"
DEFAULT_OUT = REPO / "persona_runs/dnd_good_scale/sae/ssv_corpus_interp.json"
DEFAULT_MODEL = os.environ.get("SAE_AUTOINTERP_MODEL", "gemini-2.5-flash")
CONTEXT_WINDOW = 8
MAX_SEQ_LEN = 256
TOP_K_CONTEXTS = 20
TOP_K_NEGATIVES = 5
DEFAULT_N_TOKENS = 2_000_000  # ~2M tokens; increase to 10M for full plan


@dataclass(order=True)
class _ActItem:
    neg_activation: float
    feature_id: int = field(compare=False)
    activation: float = field(compare=False)
    context: str = field(compare=False)
    token: str = field(compare=False)


def extract_ssv_features(ssv_path: Path, k_levels: list[int] | None = None) -> dict[int, float]:
    """Union of feature IDs from SSV results, with max |weight| across K levels."""
    doc = json.loads(ssv_path.read_text(encoding="utf-8"))
    weights: dict[int, float] = {}
    for row in doc.get("results", []):
        k = row.get("k")
        if k_levels and k not in k_levels:
            continue
        fids = row.get("feature_ids") or []
        wts = row.get("feature_weights") or []
        if not wts and fids:
            wts = [1.0] * len(fids)
        for fid, w in zip(fids, wts):
            fid = int(fid)
            weights[fid] = max(weights.get(fid, 0.0), abs(float(w)))
    return weights


def context_snippet(token_strs: list[str], idx: int, window: int = CONTEXT_WINDOW) -> str:
    start = max(0, idx - window)
    end = min(len(token_strs), idx + window + 1)
    parts = []
    for i in range(start, end):
        tok = token_strs[i].replace("\n", "\\n")
        parts.append(f">>>{tok}<<<" if i == idx else tok)
    return "".join(parts)


class FeatureContextCache:
    """Per-feature min-heaps for top activating contexts + low-act negatives."""

    def __init__(self, feature_ids: list[int], top_k: int = TOP_K_CONTEXTS):
        self.top_k = top_k
        self.heaps: dict[int, list[_ActItem]] = {f: [] for f in feature_ids}
        self.negatives: dict[int, list[_ActItem]] = {f: [] for f in feature_ids}
        self.n_updates = 0

    def update(
        self,
        z: torch.Tensor,
        token_strs: list[str],
        target_fids: set[int],
    ) -> None:
        fid_list = sorted(target_fids)
        z_sub = z[:, fid_list]  # (T, n_fids)
        for fi, fid in enumerate(fid_list):
            col = z_sub[:, fi]
            for ti in range(col.shape[0]):
                act = float(col[ti].item())
                if act <= 0:
                    continue
                ctx = context_snippet(token_strs, ti)
                tok = token_strs[ti].strip()
                item = _ActItem(-act, feature_id=fid, activation=act, context=ctx, token=tok)
                heap = self.heaps[fid]
                if len(heap) < self.top_k:
                    heapq.heappush(heap, item)
                elif act > -heap[0].neg_activation:
                    heapq.heapreplace(heap, item)
                self.n_updates += 1

    def finalize_negatives(self, rng: random.Random) -> None:
        """Sample low-activation contexts from bottom of heaps as hard negatives."""
        for fid, heap in self.heaps.items():
            if not heap:
                continue
            sorted_items = sorted(heap, key=lambda x: x.activation)
            self.negatives[fid] = sorted_items[:TOP_K_NEGATIVES]

    def top_examples(self, fid: int) -> list[dict]:
        heap = self.heaps.get(fid, [])
        items = sorted(heap, key=lambda x: x.activation, reverse=True)
        return [
            {"activation": round(x.activation, 3), "token": x.token, "context": x.context}
            for x in items
        ]

    def neg_examples(self, fid: int) -> list[dict]:
        return [
            {"activation": round(x.activation, 3), "token": x.token, "context": x.context}
            for x in self.negatives.get(fid, [])
        ]


def load_corpus_chunks(
    dataset_repo: str,
    dataset_split: str,
    dataset_column: str,
    max_tokens: int,
    seq_len: int,
    tokenizer,
    seed: int,
) -> list[torch.Tensor]:
    from datasets import load_dataset

    logger.info("Loading corpus %s split=%s column=%s", dataset_repo, dataset_split, dataset_column)
    try:
        ds = load_dataset(dataset_repo, split=dataset_split, streaming=True, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Failed to load %s: %s; falling back to wikitext", dataset_repo, exc)
        dataset_repo = "wikitext"
        dataset_split = "train"
        dataset_column = "text"
        ds = load_dataset(dataset_repo, "wikitext-103-v1", split=dataset_split, streaming=True)
    rng = random.Random(seed)
    chunks: list[torch.Tensor] = []
    token_budget = 0

    for row in ds:
        text = row.get(dataset_column) or row.get("text") or ""
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(ids) < 32:
            continue
        for start in range(0, len(ids), seq_len):
            chunk = ids[start : start + seq_len]
            if len(chunk) < 32:
                continue
            chunks.append(torch.tensor(chunk, dtype=torch.long))
            token_budget += len(chunk)
            if token_budget >= max_tokens:
                logger.info("Collected %d chunks (~%d tokens)", len(chunks), token_budget)
                return chunks
        if len(chunks) % 500 == 0 and chunks:
            logger.info("  ... %d chunks, ~%d tokens", len(chunks), token_budget)

    logger.info("Corpus exhausted: %d chunks, ~%d tokens", len(chunks), token_budget)
    rng.shuffle(chunks)
    return chunks


def cache_activations(
    model,
    tokenizer,
    dev,
    sae,
    layer: int,
    chunks: list[torch.Tensor],
    target_fids: set[int],
    batch_size: int = 1,
) -> FeatureContextCache:
    from app.persona.steering_demo import _language_model_layers

    cache = FeatureContextCache(sorted(target_fids))
    layers = _language_model_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = layers[layer].register_forward_hook(_hook)

    try:
        for bi in range(0, len(chunks), batch_size):
            batch = chunks[bi : bi + batch_size]
            max_len = max(c.shape[0] for c in batch)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros_like(input_ids)
            for i, c in enumerate(batch):
                input_ids[i, : c.shape[0]] = c
                attn[i, : c.shape[0]] = 1
            input_ids = input_ids.to(dev)
            attn = attn.to(dev)

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            hs = captured["h"].float()

            for i, c in enumerate(batch):
                seq_len = c.shape[0]
                h = hs[i, :seq_len, :].cpu()
                z = sae.encode(h.unsqueeze(0))[0].float()
                token_strs = [tokenizer.decode([tid]) for tid in c.tolist()]
                cache.update(z, token_strs, target_fids)

            if (bi // batch_size) % 25 == 0:
                logger.info(
                    "Cached batch %d/%d (updates=%d)",
                    bi // batch_size,
                    len(chunks) // batch_size + 1,
                    cache.n_updates,
                )
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    cache.finalize_negatives(random.Random(42))
    return cache


def build_explanation_prompt(fid: int, pos_examples: list[dict], neg_examples: list[dict]) -> str:
    lines = [
        "You are interpreting a sparse autoencoder (SAE) feature from Gemma-3-4B-IT, layer 16 residual stream.",
        f"Feature index: F{fid}.",
        "",
        "ACTIVATING examples (>>>token<<< marks strongest activation):",
    ]
    for ex in pos_examples[:12]:
        lines.append(f"  act={ex['activation']:.2f}  {ex['context']}")
    lines.append("")
    lines.append("LOW-ACTIVATION examples from similar contexts (should NOT activate this feature):")
    for ex in neg_examples[:5]:
        lines.append(f"  act={ex['activation']:.2f}  {ex['context']}")
    lines.extend([
        "",
        "In one concise sentence (under 15 words), describe what concept or pattern",
        "this feature detects. Focus on semantic meaning, not token identity.",
        "Reply with ONLY the one-sentence interpretation.",
    ])
    return "\n".join(lines)


def build_detection_prompt(fid: int, explanation: str, examples: list[dict]) -> str:
    labeled = []
    for i, ex in enumerate(examples):
        labeled.append(f"Example {i+1} (activation={ex['activation']:.2f}):\n  {ex['context']}")
    lines = [
        f"Feature F{fid} explanation: {explanation}",
        "",
        "For each example below, answer YES if the explanation matches the highlighted >>>token<<<, else NO.",
        "Reply as JSON list of booleans, e.g. [true, false, true].",
        "",
        *labeled,
    ]
    return "\n".join(lines)


def gemini_generate(prompt: str, project_id: str, model_name: str, max_tokens: int = 256) -> str:
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    from app.persona.config import DEFAULT_VERTEX_LOCATION

    vertexai.init(project=project_id, location=os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION))
    model = GenerativeModel(model_name)
    cfg = GenerationConfig(temperature=0.2, max_output_tokens=max_tokens)
    out = model.generate_content(prompt, generation_config=cfg)
    text = (out.text or "").strip()
    if not text:
        raise RuntimeError("Empty Gemini response")
    return text


def score_detection(
    fid: int,
    explanation: str,
    pos_examples: list[dict],
    neg_examples: list[dict],
    project_id: str,
    model_name: str,
) -> dict:
    """Lightweight detection score: can Gemini identify high vs low activation contexts?"""
    if len(pos_examples) < 3:
        return {"detection_accuracy": None, "n_tested": 0}

    test_set = pos_examples[:5] + neg_examples[:3]
    random.Random(fid).shuffle(test_set)
    expected = [ex["activation"] >= 1.0 for ex in test_set]

    prompt = build_detection_prompt(fid, explanation, test_set)
    try:
        raw = gemini_generate(prompt, project_id, model_name, max_tokens=128)
        # Parse JSON-ish response
        import re
        m = re.search(r"\[[^\]]+\]", raw)
        if not m:
            return {"detection_accuracy": None, "n_tested": len(test_set), "raw": raw[:200]}
        preds = json.loads(m.group().replace("true", "True").replace("false", "False"))
        preds = [bool(p) for p in preds[: len(test_set)]]
        if len(preds) != len(expected):
            return {"detection_accuracy": None, "n_tested": len(test_set), "raw": raw[:200]}
        acc = sum(p == e for p, e in zip(preds, expected)) / len(expected)
        return {"detection_accuracy": round(acc, 3), "n_tested": len(test_set)}
    except Exception as exc:
        return {"detection_accuracy": None, "n_tested": len(test_set), "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", default="good")
    ap.add_argument("--ssv", type=Path, default=DEFAULT_SSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=None, help="Save/load activation context cache")
    ap.add_argument("--k-levels", default="100,512", help="SSV K levels to union features from")
    ap.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS)
    ap.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dataset", default="monology/pile-uncopyrighted")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--dataset-column", default="text")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00"))
    ap.add_argument("--skip-cache", action="store_true", help="Load --cache instead of re-caching")
    ap.add_argument("--skip-explain", action="store_true", help="Only cache activations")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = cfg["layer"]
    sae_id = cfg["sae_id"]
    k_levels = [int(x.strip()) for x in args.k_levels.split(",") if x.strip()]
    feature_weights = extract_ssv_features(args.ssv, k_levels)
    if not feature_weights:
        logger.error("No features found in %s for K levels %s", args.ssv, k_levels)
        return 1

    target_fids = set(feature_weights.keys())
    logger.info("Target features: %d from SSV K=%s", len(target_fids), k_levels)

    cache_path = args.cache or (args.out.parent / "ssv_corpus_cache.json")

    if args.skip_cache and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        contexts = cached["contexts"]
        logger.info("Loaded cache for %d features from %s", len(contexts), cache_path)
    else:
        model, tokenizer, dev = load_model_and_tokenizer()
        sae, _ = load_sae_for_layer(
            torch.device("cpu"), release=SAE_RELEASE, sae_id=sae_id, hidden_state_index=cfg["hs_index"],
        )
        chunks = load_corpus_chunks(
            args.dataset, args.dataset_split, args.dataset_column,
            args.n_tokens, args.seq_len, tokenizer, args.seed,
        )
        if not chunks:
            logger.error("No corpus chunks collected")
            return 1
        feat_cache = cache_activations(model, tokenizer, dev, sae, layer, chunks, target_fids, args.batch_size)
        contexts = {str(fid): {"pos": feat_cache.top_examples(fid), "neg": feat_cache.neg_examples(fid)} for fid in target_fids}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"contexts": contexts, "n_tokens": args.n_tokens}, indent=2) + "\n")
        logger.info("Wrote activation cache %s", cache_path)
        del model, sae
        if args.skip_explain:
            return 0

    existing = {}
    if args.out.is_file() and not args.force:
        existing = json.loads(args.out.read_text(encoding="utf-8")).get("features", {})

    results: dict[str, dict] = {}
    n_new = 0
    for fid in sorted(target_fids, key=lambda f: feature_weights[f], reverse=True):
        key = str(fid)
        ctx = contexts.get(key, {})
        pos = ctx.get("pos", [])
        neg = ctx.get("neg", [])

        if not args.force and key in existing and existing[key].get("interpretation"):
            results[key] = existing[key]
            continue

        if not pos:
            results[key] = {
                "interpretation": "No activations found in corpus",
                "source": "no_activations",
                "ssv_weight": feature_weights[fid],
                "top_examples": [],
            }
            continue

        prompt = build_explanation_prompt(fid, pos, neg)
        try:
            interpretation = gemini_generate(prompt, args.project, args.model).split("\n")[0].strip().rstrip(".")
            source = "corpus_gemini"
            n_new += 1
        except Exception as exc:
            logger.error("F%d explain failed: %s", fid, exc)
            interpretation = f"Interpretation failed: {exc}"
            source = "error"

        entry = {
            "interpretation": interpretation,
            "source": source,
            "ssv_weight": round(feature_weights[fid], 6),
            "top_examples": pos[:8],
            "neg_examples": neg[:3],
        }

        if not args.skip_score and source == "corpus_gemini":
            entry["detection"] = score_detection(fid, interpretation, pos, neg, args.project, args.model)

        results[key] = entry
        logger.info("F%d -> %s (det=%s)", fid, interpretation[:60], entry.get("detection", {}).get("detection_accuracy"))

        # Incremental save
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "meta": {
                "method": "ssv_corpus_interp",
                "trait": args.trait,
                "layer": layer,
                "sae_id": sae_id,
                "n_features": len(results),
                "k_levels": k_levels,
                "n_tokens": args.n_tokens,
                "corpus": args.dataset,
            },
            "features": results,
        }, indent=2) + "\n", encoding="utf-8")

    logger.info("Wrote %s (%d features, %d new)", args.out, len(results), n_new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
