#!/usr/bin/env python3
"""Build OMP bubble viz JSON from persona_runs/*/sae/ssv_omp_dsweep_l*.json + corpus interp + logit lens."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "app/static"
SHARED = REPO / "persona_runs" / "_shared"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from trait_sae_config import resolve_trait


def _neuronpedia_url(layer: int, fid: int) -> str:
    return f"https://www.neuronpedia.org/gemma-3-4b-it/{layer}-gemmascope-2-transcoder-262k/{fid}"


def _load_lens_rows(path: Path | None) -> dict[int, dict]:
    if not path or not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["fid"]): r for r in rows}


def load_lens_by_fid(trait: str, layer: int, needed: set[int]) -> dict[int, dict]:
    """Merge OMP lens, SSV trait lens, and shared layer cache for requested fids."""
    cfg = resolve_trait(trait)
    sae_dir = cfg["sae_dir"]
    merged: dict[int, dict] = {}
    for path in (
        sae_dir / f"ssv_omp_feature_logit_lens_262k_l{layer}.json",
        sae_dir / f"ssv_feature_logit_lens_262k_l{layer}.json",
    ):
        merged.update(_load_lens_rows(path))
    cache_path = SHARED / f"l{layer}_262k_logit_lens_cache.json"
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for fid in needed:
            entry = cache.get(str(fid))
            if entry:
                merged[int(fid)] = entry
    return {fid: merged[fid] for fid in needed if fid in merged}


def _load_interp_file(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for key, entry in (doc.get("features") or {}).items():
        out[int(key)] = entry
    return out


def load_corpus_interp(trait: str, layer: int) -> dict[int, dict]:
    cfg = resolve_trait(trait)
    return _load_interp_file(cfg["sae_dir"] / "ssv_omp_corpus_interp.json")


def load_lens_interp(trait: str, layer: int) -> dict[int, dict]:
    cfg = resolve_trait(trait)
    return _load_interp_file(cfg["sae_dir"] / "ssv_omp_lens_interp.json")


_VERBOSE_PREFIXES = [
    "This feature activates on ",
    "This feature detects ",
    "This feature fires on ",
    "This feature responds to ",
    "This feature identifies ",
    "This feature captures ",
    "Activates on ",
    "Detects ",
]


def _strip_verbose(text: str) -> str:
    """Strip common Gemini verbose prefixes to get a concise noun-phrase label."""
    for prefix in _VERBOSE_PREFIXES:
        if text.startswith(prefix):
            remainder = text[len(prefix):]
            if remainder:
                return remainder[0].upper() + remainder[1:]
    return text


def _good_interp(entry: dict | None) -> str:
    if not entry:
        return ""
    text = (entry.get("interpretation") or "").strip()
    if not text:
        return ""
    if entry.get("source") in ("error", "no_activations"):
        return ""
    if text.startswith("Interpretation failed") or text.startswith("No activations found"):
        return ""
    return _strip_verbose(text)


def _good_lens_title(entry: dict | None) -> str:
    if not entry:
        return ""
    title = (entry.get("title") or "").strip()
    if not title or entry.get("source") in ("error",):
        return ""
    if title.lower() == "polysemantic":
        return ""
    return title


def _good_lens_desc(entry: dict | None) -> str:
    if not entry:
        return ""
    desc = (entry.get("description") or "").strip()
    if not desc or entry.get("source") in ("error",):
        return ""
    return _strip_verbose(desc)


def feature_label(
    fid: int,
    interp_by_fid: dict[int, dict],
    lens_interp_by_fid: dict[int, dict],
    lens: dict | None,
) -> tuple[str, str, str]:
    """Return (title, description, label_source).
    
    Priority: lens_gemini title/desc > corpus_gemini (as desc, first 3 words as title) > logit_lens tokens.
    """
    from ssv_lens_themes import label_from_lens

    li_entry = lens_interp_by_fid.get(fid)
    li_title = _good_lens_title(li_entry)
    li_desc = _good_lens_desc(li_entry)

    corpus = _good_interp(interp_by_fid.get(fid))

    if li_title:
        desc = li_desc or corpus or ""
        return li_title, desc, "lens_gemini"
    if corpus:
        words = corpus.split()
        title = " ".join(words[:3]) if len(words) > 3 else corpus
        return title, corpus, "corpus_gemini"
    lens_label = label_from_lens(lens)
    if lens_label:
        return lens_label.split(",")[0].strip(), lens_label, "logit_lens"
    return "", "", "none"


def _load_omp_decomposition(sae_dir: Path, layer: int) -> list[dict]:
    """Load OMP decomposition sorted by |coefficient| descending."""
    path = sae_dir / f"omp_decomposition_262k_l{layer}.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("decomposition") or []
    return sorted(rows, key=lambda r: abs(float(r.get("coefficient", 0))), reverse=True)


def _omp_features_at_k(decomposition: list[dict], k: int) -> tuple[list[int], list[float]]:
    """Top-K features from the decomposition."""
    top = decomposition[:k]
    return [int(r["feature_id"]) for r in top], [float(r["coefficient"]) for r in top]


def _load_judged_scores_by_k(path: Path) -> dict[int, dict]:
    """Load mean_trait + scores keyed by K from a k_sweep results file."""
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for row in doc.get("results") or []:
        k = row.get("k") if row.get("k") is not None else row.get("d")
        if k is None or row.get("label") == "dense_ref":
            continue
        out[int(k)] = {
            "mean_trait": row.get("mean_trait") if "mean_trait" in row else row.get("mean"),
            "scores": row.get("scores") or [],
        }
    return out


def _overlay_judged_scores(data: dict, scores_by_k: dict[int, dict], *, source: str) -> None:
    """Replace mean_trait/scores on k_levels and comparison_curves when judged data exists."""
    if not scores_by_k:
        return
    for lvl in data.get("k_levels") or []:
        k = int(lvl["k"])
        judged = scores_by_k.get(k)
        if not judged:
            continue
        if judged.get("mean_trait") is not None:
            lvl["mean_trait"] = judged["mean_trait"]
        if judged.get("scores"):
            lvl["scores"] = judged["scores"]
    curve = data.get("comparison_curves", {}).get("omp") or []
    for pt in curve:
        k = int(pt["d"])
        judged = scores_by_k.get(k)
        if judged and judged.get("mean_trait") is not None:
            pt["mean"] = judged["mean_trait"]
    if curve:
        data.setdefault("comparison_curves", {})["omp"] = curve
    data.setdefault("meta", {})["judged_scores_source"] = source


def _normalize_sweep(doc: dict, sae_dir: Path, layer: int) -> list[dict]:
    """Normalize K sweep or old dsweep rows into a common format with d/feature_ids/feature_weights/mean/scores."""
    results = doc.get("results") or []
    first = next((r for r in results if r.get("k") is not None or r.get("d") is not None), None)
    if not first:
        return []

    has_features = bool(first.get("feature_ids"))
    uses_k_key = "k" in first and "d" not in first

    if has_features and not uses_k_key:
        return results

    decomposition = _load_omp_decomposition(sae_dir, layer)
    if not decomposition:
        print(f"  WARN: no OMP decomposition at L{layer} — cannot look up features for K sweep rows")
        return results

    normalized = []
    for row in results:
        if row.get("label") == "dense_ref":
            continue
        k = int(row.get("k") or row.get("d"))
        fids, weights = _omp_features_at_k(decomposition, k)
        normalized.append({
            "d": k,
            "feature_ids": fids,
            "feature_weights": weights,
            "mean": row.get("mean_trait") if "mean_trait" in row else row.get("mean"),
            "scores": row.get("scores") or [],
            "scale": row.get("scale"),
            "cosine_vs_dense": row.get("cosine_vs_dense"),
            "best_cal_score": row.get("best_cal_score"),
        })
    return normalized


def build_trait(
    trait: str,
    sweep_path: Path,
    lens_by_fid: dict[int, dict],
    interp_by_fid: dict[int, dict],
    lens_interp_by_fid: dict[int, dict],
) -> dict:
    from ssv_lens_themes import suppress_label_from_lens, theme_from_label, theme_from_lens

    doc = json.loads(sweep_path.read_text(encoding="utf-8"))
    cfg = resolve_trait(trait)
    layer = int(doc.get("layer", cfg["layer"]))
    sae_dir = cfg["sae_dir"]

    rows = _normalize_sweep(doc, sae_dir, layer)

    curve: list[dict] = []
    k_levels: list[dict] = []
    all_fids: set[int] = set()

    for row in sorted(rows, key=lambda r: int(r["d"])):
        d = int(row["d"])
        fids = [int(x) for x in row.get("feature_ids") or []]
        weights = [float(w) for w in row.get("feature_weights") or []]
        all_fids.update(fids)

        curve.append({"d": d, "mean": row.get("mean")})

        max_abs = max((abs(w) for w in weights), default=1.0) or 1.0
        features = []
        for rank, (fid, wt) in enumerate(zip(fids, weights), start=1):
            lens = lens_by_fid.get(fid)
            title, desc, label_source = feature_label(fid, interp_by_fid, lens_interp_by_fid, lens)
            theme = theme_from_label(title or desc, trait) if label_source in ("corpus_gemini", "lens_gemini") else theme_from_lens(lens, trait)
            abs_w = abs(wt)
            interp_entry = interp_by_fid.get(fid) or {}
            lens_score = lens["top_tokens"][0][1] if lens and lens.get("top_tokens") else 0.0
            features.append(
                {
                    "fid": fid,
                    "rank": rank,
                    "weight": round(wt, 4),
                    "abs_weight": round(abs_w, 4),
                    "importance": round(abs_w / max_abs, 4),
                    "lens_score": round(float(lens_score), 4),
                    "sign": "pos" if wt >= 0 else "neg",
                    "title": title,
                    "description": desc,
                    "label": title,
                    "label_source": label_source,
                    "theme": theme,
                    "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
                    "top_suppress": (
                        lens.get("top_suppress") or lens.get("bot_tokens") or []
                    )[:8]
                    if lens
                    else [],
                    "suppress_label": suppress_label_from_lens(lens),
                    "detection_accuracy": (interp_entry.get("detection") or {}).get("detection_accuracy"),
                }
            )

        k_levels.append(
            {
                "k": d,
                "ranking": "omp",
                "n_active": len(fids),
                "scale": row.get("scale"),
                "cosine_vs_dense": row.get("cosine_vs_dense"),
                "mean_trait": row.get("mean"),
                "scores": row.get("scores") or [],
                "best_cal_score": row.get("best_cal_score"),
                "features": features,
            }
        )

    curve = sorted(curve, key=lambda x: int(x["d"]))
    d_values = sorted({int(lvl["k"]) for lvl in k_levels})
    n_corpus = sum(1 for fid in all_fids if _good_interp(interp_by_fid.get(fid)))
    n_lens_interp = sum(
        1 for fid in all_fids
        if not _good_interp(interp_by_fid.get(fid)) and _good_interp(lens_interp_by_fid.get(fid))
    )
    n_lens_tokens = sum(
        1 for fid in all_fids
        if not _good_interp(interp_by_fid.get(fid))
        and not _good_interp(lens_interp_by_fid.get(fid))
        and feature_label(fid, interp_by_fid, lens_interp_by_fid, lens_by_fid.get(fid))[0]
    )

    feature_meta = {}
    for fid in sorted(all_fids):
        lens = lens_by_fid.get(fid)
        title, desc, label_source = feature_label(fid, interp_by_fid, lens_interp_by_fid, lens)
        theme = theme_from_label(title or desc, trait) if label_source in ("corpus_gemini", "lens_gemini") else theme_from_lens(lens, trait)
        interp_entry = interp_by_fid.get(fid) or {}
        lens_score = lens["top_tokens"][0][1] if lens and lens.get("top_tokens") else 0.0
        feature_meta[str(fid)] = {
            "fid": fid,
            "title": title,
            "description": desc,
            "label": title,
            "label_source": label_source,
            "lens_score": round(float(lens_score), 4),
            "theme": theme,
            "top_tokens": lens.get("top_tokens", [])[:8] if lens else [],
            "top_suppress": (
                lens.get("top_suppress") or lens.get("bot_tokens") or []
            )[:8]
            if lens
            else [],
            "suppress_label": suppress_label_from_lens(lens),
            "neuronpedia": _neuronpedia_url(layer, fid),
            "detection_accuracy": (interp_entry.get("detection") or {}).get("detection_accuracy"),
        }

    return {
        "meta": {
            "trait": trait,
            "layer": layer,
            "method": "omp_sae_hook",
            "rankings": ["omp"],
            "d_values": d_values,
            "note": "OMP d-sweep. Labels: corpus Gemini > lens Gemini > logit lens tokens.",
            "n_features_corpus_interp": n_corpus,
            "n_features_lens_interp": n_lens_interp,
            "n_features_logit_lens_fallback": n_lens_tokens,
            "n_omp_features": len(all_fids),
            "sae_id": doc.get("sae_id") or cfg.get("sae_id"),
            "alpha_reference": doc.get("alpha_reference") or doc.get("alpha_dense") or cfg.get("alpha"),
            "early_stopped": bool(doc.get("early_stopped")),
            "n_questions": doc.get("n_questions"),
        },
        "comparison_curves": {
            "omp": curve,
        },
        "k_levels": sorted(k_levels, key=lambda x: int(x["k"])),
        "feature_meta": feature_meta,
    }


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for trait in ("good", "evil", "lawful", "chaotic", "male", "female"):
        cfg = resolve_trait(trait)
        layer = int(cfg["layer"])
        sae_dir = cfg["sae_dir"]

        k_sweep_path = sae_dir / f"omp_k_sweep_l{layer}_20q.json"
        emd_path = sae_dir / f"omp_k_sweep_l{layer}_20q_emd.json"
        old_dsweep_path = sae_dir / f"ssv_omp_dsweep_l{layer}.json"
        if emd_path.is_file():
            sweep_path = emd_path
            print(f"{trait}: using EMD K sweep (20Q) {emd_path.name}")
        elif k_sweep_path.is_file():
            sweep_path = k_sweep_path
            print(f"{trait}: using K sweep (20Q) {k_sweep_path.name}")
        elif old_dsweep_path.is_file():
            sweep_path = old_dsweep_path
            print(f"{trait}: using old dsweep {old_dsweep_path.name}")
        else:
            print(f"SKIP {trait}: no sweep file found")
            continue
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        needed: set[int] = set()
        for row in sweep.get("results") or []:
            needed.update(int(f) for f in row.get("feature_ids") or [])
        if not needed:
            decomp = _load_omp_decomposition(sae_dir, layer)
            max_k = max((int(r.get("k") or r.get("d") or 0) for r in sweep.get("results") or [] if r.get("label") != "dense_ref"), default=0)
            needed.update(int(r["feature_id"]) for r in decomp[:max_k])
        lens_by_fid = load_lens_by_fid(trait, layer, needed)
        interp_by_fid = load_corpus_interp(trait, layer)
        lens_interp_by_fid = load_lens_interp(trait, layer)
        data = build_trait(trait, sweep_path, lens_by_fid, interp_by_fid, lens_interp_by_fid)
        if emd_path.is_file():
            emd_scores = _load_judged_scores_by_k(emd_path)
            if emd_scores:
                _overlay_judged_scores(data, emd_scores, source=emd_path.name)
                print(f"  overlaid EMD judged scores from {emd_path.name} ({len(emd_scores)} K levels)")
        out = STATIC / f"ssv_bubble_viz_omp_data_{trait}.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        manifest[trait] = out.name
        meta = data["meta"]
        print(
            f"Wrote {out.name} "
            f"(d={meta['d_values']}, "
            f"corpus={meta['n_features_corpus_interp']}, "
            f"lens_interp={meta['n_features_lens_interp']}, "
            f"lens_tokens={meta['n_features_logit_lens_fallback']}, "
            f"total={meta['n_omp_features']})"
        )

    if manifest:
        (STATIC / "ssv_bubble_viz_omp_manifest.json").write_text(
            json.dumps({"traits": manifest, "default": "good"}, indent=2),
            encoding="utf-8",
        )
        print("Wrote ssv_bubble_viz_omp_manifest.json")


if __name__ == "__main__":
    main()
