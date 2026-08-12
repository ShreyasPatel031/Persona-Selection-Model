#!/usr/bin/env python3
"""
Interpret OMP SAE features via logit lens tokens + Gemini.

Uses pre-computed W_dec·lm_head token lists from the shared lens cache.
No model loading, no corpus -- just Gemini API calls.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from trait_sae_config import resolve_trait

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHARED = REPO / "persona_runs" / "_shared"
DEFAULT_MODEL = os.environ.get("SAE_AUTOINTERP_MODEL", "gemini-2.5-flash")


def collect_fids_from_decomp(trait: str, *, layer: int, top_k: int) -> set[int]:
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


def format_token_pairs(pairs: list, *, limit: int = 12) -> str:
    if not pairs:
        return "(none)"
    parts = []
    for tok, score in pairs[:limit]:
        tok_s = str(tok).strip().replace("\n", "\\n")
        parts.append(f"{tok_s} ({float(score):+.2f})")
    return ", ".join(parts)


def build_prompt(fid: int, entry: dict, *, layer: int, trait: str) -> str:
    boost = entry.get("top_tokens") or []
    suppress = entry.get("top_suppress") or entry.get("bot_tokens") or []
    return "\n".join([
        f"You are interpreting a sparse autoencoder (SAE) feature from Gemma-3-4B-IT, layer {layer} residual stream.",
        f"This feature appears in an OMP decomposition of the '{trait}' persona steering vector.",
        f"Feature index: F{fid}.",
        "",
        "The logit lens (W_dec dot lm_head, with RMSNorm folded in) shows these tokens get a logit boost:",
        f"  {format_token_pairs(boost)}",
        "",
        "And these tokens get suppressed:",
        f"  {format_token_pairs(suppress)}",
        "",
        "Reply with exactly two lines, no other text:",
        "Line 1 — TITLE: A 1–3 word noun-phrase label for this feature (e.g. 'Empathy', 'Dark humor', 'Legal terms').",
        "Line 2 — DESC: One sentence (under 20 words) explaining what concept this feature detects.",
        "",
        "If the tokens are mostly garbage, sub-token artifacts, or <unused> slots:",
        "TITLE: Polysemantic",
        "DESC: Sub-token feature with no clear semantic concept.",
    ])


def _get_response_text(out) -> str:
    """Extract full text from a Gemini response, handling thinking models."""
    chunks: list[str] = []
    try:
        if out.text:
            chunks.append(out.text.strip())
    except ValueError:
        pass
    for cand in getattr(out, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                t = getattr(part, "text", None)
                if t and str(t).strip():
                    chunks.append(str(t).strip())
    if not chunks:
        raise RuntimeError("Empty Gemini response")
    return "\n".join(chunks)


def _parse_title_desc(raw: str) -> tuple[str, str]:
    """Parse TITLE: ... / DESC: ... from Gemini response."""
    import re
    title = ""
    desc = ""
    for line in raw.split("\n"):
        line = line.strip()
        m_title = re.match(r"(?:TITLE:\s*|Line\s*1[:\s—-]+(?:TITLE:\s*)?)", line, re.IGNORECASE)
        m_desc = re.match(r"(?:DESC:\s*|Line\s*2[:\s—-]+(?:DESC:\s*)?)", line, re.IGNORECASE)
        if m_title:
            title = line[m_title.end():].strip().strip('"').strip("'").rstrip(".")
        elif m_desc:
            desc = line[m_desc.end():].strip().strip('"').strip("'").rstrip(".")
        elif not title and not desc and len(line) > 2 and len(line) <= 40:
            title = line.strip('"').strip("'").rstrip(".")
        elif title and not desc and len(line) > 10:
            desc = line.strip('"').strip("'").rstrip(".")

    if not title and not desc:
        clean = raw.strip().split("\n")[0].strip().strip('"').strip("'").rstrip(".")
        if len(clean) <= 40:
            title = clean
        else:
            title = clean[:30].rsplit(" ", 1)[0]
            desc = clean

    return title[:40], desc[:150]


def gemini_generate(prompt: str, project_id: str, model_name: str) -> tuple[str, str]:
    """Returns (title, description)."""
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    from app.persona.config import DEFAULT_VERTEX_LOCATION

    vertexai.init(
        project=project_id,
        location=os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
    )
    model = GenerativeModel(model_name)
    cfg = GenerationConfig(temperature=0.2, max_output_tokens=4096)
    out = model.generate_content(prompt, generation_config=cfg)
    raw = _get_response_text(out)
    return _parse_title_desc(raw)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traits", default="good,evil,lawful,chaotic")
    ap.add_argument("--decomp-top-k", type=int, default=20,
                    help="Use top-K OMP decomposition features instead of dsweep FIDs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]

    for trait in traits:
        cfg = resolve_trait(trait)
        layer = cfg["layer"]

        if args.decomp_top_k > 0:
            fids = collect_fids_from_decomp(trait, layer=layer, top_k=args.decomp_top_k)
        else:
            sweep_path = cfg["sae_dir"] / f"ssv_omp_dsweep_l{layer}.json"
            if not sweep_path.is_file():
                logger.warning("SKIP %s: missing %s", trait, sweep_path)
                continue
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
            fids = set()
            for row in sweep.get("results", []):
                fids.update(int(f) for f in row.get("feature_ids", []))

        cache_path = SHARED / f"l{layer}_262k_logit_lens_cache.json"
        lens_cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}

        out_path = cfg["sae_dir"] / "ssv_omp_lens_interp.json"
        existing: dict[str, dict] = {}
        if out_path.is_file() and not args.force:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get("features", {})

        results: dict[str, dict] = {}
        n_new = 0
        for fid in sorted(fids):
            key = str(fid)
            if not args.force and key in existing and existing[key].get("title"):
                results[key] = existing[key]
                continue

            entry = lens_cache.get(key)
            if not entry:
                results[key] = {"interpretation": "", "source": "missing_lens"}
                continue

            prompt = build_prompt(fid, entry, layer=layer, trait=trait)
            try:
                title, desc = gemini_generate(prompt, args.project, args.model)
                source = "lens_gemini"
                n_new += 1
            except Exception as exc:
                logger.error("F%d failed: %s", fid, exc)
                title, desc = "", f"Interpretation failed: {exc}"
                source = "error"

            results[key] = {"title": title, "description": desc, "source": source}
            logger.info("F%d -> [%s] %s", fid, title, desc[:60])

            out_path.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "meta": {
                    "method": "lens_gemini",
                    "trait": trait,
                    "layer": layer,
                    "model": args.model,
                    "n_features": len(results),
                },
                "features": results,
            }
            out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        logger.info("Wrote %s (%d features, %d new)", out_path, len(results), n_new)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
