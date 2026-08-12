#!/usr/bin/env python3
"""Interpret SAE features via Gemini using logit-lens token lists."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LENS = REPO / "app/static/logit_lens_l16_all.json"
DEFAULT_SWEEP = REPO / "persona_runs/dnd_good_scale/sae/alpha_sweep_analysis.json"
DEFAULT_OUT = REPO / "app/static/feature_interpretations.json"
DEFAULT_MODEL = os.environ.get("SAE_AUTOINTERP_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 4096


MAX_SENTENCE_CHARS = 120


def _looks_like_sentence(line: str) -> bool:
    t = line.strip().strip('"').strip("'")
    if not t or len(t) < 12:
        return False
    if len(t) > MAX_SENTENCE_CHARS:
        return False
    if t.count("`") >= 2:
        return False
    if sum(1 for c in t if ord(c) > 0xA000) > 3:
        return False
    if t in {",", ".", "Poly", "This", "Detects"}:
        return False
    if "(10 words)" in t or "This is good" in t:
        return False
    return True


def _clean_sentence(line: str) -> str:
    t = line.strip().strip('"').strip("'")
    for marker in (" (10 words)", " - This is good", '." ('):
        if marker in t:
            t = t.split(marker)[0]
    return t.rstrip(".\"' ")


def _extract_response_text(out) -> str:
    try:
        if out.text:
            chunks = [out.text.strip()]
        else:
            chunks = []
    except ValueError:
        chunks = []
    for cand in out.candidates or []:
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                t = getattr(part, "text", None)
                if t and str(t).strip():
                    chunks.append(str(t).strip())
    if not chunks:
        raise RuntimeError("Empty Vertex auto-interp response.")

    candidates: list[str] = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            line = _clean_sentence(line.strip().strip("-"))
            if _looks_like_sentence(line):
                candidates.append(line)

    if candidates:
        # Prefer explicit polysemantic fallback or concise Detects/Recognizes lines
        for pref in ("Polysemantic sub-token feature", "Detects ", "Recognizes ", "This feature "):
            for c in candidates:
                if c.startswith(pref):
                    return c[:MAX_SENTENCE_CHARS]
        return candidates[-1][:MAX_SENTENCE_CHARS]

    raise RuntimeError("No valid one-sentence interpretation in response.")


def explain_logit_lens_feature(
    prompt: str,
    *,
    project_id: str,
    model_name: str,
) -> str:
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    from app.persona.config import DEFAULT_VERTEX_LOCATION

    vertexai.init(project=project_id, location=os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION))
    model = GenerativeModel(model_name)
    cfg = GenerationConfig(temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS)
    out = model.generate_content(prompt, generation_config=cfg)
    text = _extract_response_text(out)
    if not _looks_like_sentence(text):
        raise RuntimeError(f"Truncated or low-quality response: {text!r}")
    return text


def feature_ids_from_sweep(sweep_path: Path) -> list[int]:
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    fids: set[int] = set()
    for pq in sweep.get("per_question", []):
        for rows in (pq.get("top_features_by_alpha") or {}).values():
            for row in rows:
                fids.add(int(row["feature_id"]))
    return sorted(fids)


def format_token_pairs(pairs: list, *, limit: int = 12) -> str:
    if not pairs:
        return "(none)"
    parts = []
    for tok, score in pairs[:limit]:
        tok_s = str(tok).strip().replace("\n", "\\n")
        parts.append(f"{tok_s} ({float(score):+.2f})")
    return ", ".join(parts)


def build_logit_lens_prompt(feature_id: int, row: dict) -> str:
    boost = row.get("top_boost") or []
    suppress = row.get("top_suppress") or []
    return "\n".join(
        [
            "You are interpreting a sparse autoencoder (SAE) feature from Gemma-2-4B-IT, layer 16.",
            f"Feature index: F{feature_id}.",
            "",
            "The logit lens (W_dec dot lm_head) shows these tokens get a logit boost when this feature is active:",
            f"  {format_token_pairs(boost)}",
            "",
            "And these tokens get suppressed:",
            f"  {format_token_pairs(suppress)}",
            "",
            "In one concise sentence (under 15 words), describe what concept or behavior this feature detects.",
            "Use plain English, no markdown, no quotes.",
            "Reply with ONLY the one-sentence interpretation. Do not explain your reasoning.",
            "If the tokens are mostly garbage, sub-token artifacts, or <unused> slots,",
            'say exactly: Polysemantic sub-token feature (no clear concept).',
        ]
    )


def load_out(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(doc.get("features"), dict):
        return doc["features"]
    return {k: v for k, v in doc.items() if k.isdigit()}


def save_out(path: Path, features: dict[str, dict], *, meta: dict | None = None) -> None:
    doc = {
        "meta": meta or {},
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def is_cached_ok(entry: dict, fid: int) -> bool:
    cached = entry.get("interpretation", "")
    if entry.get("source") == "error":
        return False
    if cached.startswith(f"F{fid}: interpretation failed"):
        return False
    return _looks_like_sentence(cached)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00"))
    ap.add_argument("--force", action="store_true", help="Reinterpret even if cached")
    args = ap.parse_args()

    if not args.lens.is_file():
        logger.error("Missing logit lens file: %s", args.lens)
        return 1
    if not args.sweep.is_file():
        logger.error("Missing sweep file: %s", args.sweep)
        return 1

    lens = json.loads(args.lens.read_text(encoding="utf-8"))
    fids = feature_ids_from_sweep(args.sweep)
    out = load_out(args.out)

    meta = {
        "model": args.model,
        "project": args.project,
        "lens": str(args.lens),
        "sweep": str(args.sweep),
        "n_features": len(fids),
    }

    logger.info("Interpreting %d features via %s", len(fids), args.model)
    for i, fid in enumerate(fids, start=1):
        key = str(fid)
        if not args.force and key in out and out[key].get("interpretation"):
            if is_cached_ok(out[key], fid):
                logger.info("[%d/%d] F%s cached: %s", i, len(fids), fid, out[key]["interpretation"][:60])
                continue

        row = lens.get(key)
        if not row:
            out[key] = {
                "interpretation": f"F{fid}: no logit-lens data available.",
                "source": "missing_lens",
            }
            save_out(args.out, out, meta=meta)
            logger.warning("[%d/%d] F%s missing from logit lens", i, len(fids), fid)
            continue

        prompt = build_logit_lens_prompt(fid, row)
        try:
            interpretation = explain_logit_lens_feature(
                prompt,
                project_id=args.project,
                model_name=args.model,
            )
            source = "gemini"
        except Exception as e:
            logger.error("[%d/%d] F%s Vertex failed: %s", i, len(fids), fid, e)
            interpretation = f"F{fid}: interpretation failed ({e})"
            source = "error"

        out[key] = {"interpretation": interpretation, "source": source}
        save_out(args.out, out, meta=meta)
        logger.info("[%d/%d] F%s -> %s", i, len(fids), fid, interpretation)

    save_out(args.out, out, meta=meta)
    logger.info("Wrote %s (%d features)", args.out, len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
