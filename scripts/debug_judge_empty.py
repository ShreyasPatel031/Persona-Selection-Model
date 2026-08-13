#!/usr/bin/env python3
"""Debug empty Vertex Gemini judge responses."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, os.path.expanduser("~/gemma-chat"))

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


def dump_resp(label: str, resp) -> None:
    print(f"\n===== {label} =====")
    try:
        print("resp.text =", repr(resp.text))
    except Exception as e:
        print("resp.text ERROR:", type(e).__name__, e)
    cands = list(getattr(resp, "candidates", None) or [])
    print("n_candidates =", len(cands))
    for i, c in enumerate(cands):
        fr = getattr(c, "finish_reason", None)
        print(f"cand[{i}] finish_reason={fr}")
        content = getattr(c, "content", None)
        parts = list(getattr(content, "parts", None) or []) if content else []
        print(f"  n_parts={len(parts)}")
        for j, p in enumerate(parts):
            text = getattr(p, "text", None)
            thought = getattr(p, "thought", None)
            print(f"  part[{j}] thought={thought!r} text_len={len(text) if text else 0}")
            if text:
                print(f"  part[{j}] text_preview={text[:200]!r}")
    print("prompt_feedback =", getattr(resp, "prompt_feedback", None))
    print("usage_metadata =", getattr(resp, "usage_metadata", None))


def main() -> int:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    model_name = os.environ.get("PERSONA_JUDGE_MODEL", "gemini-2.5-flash")
    print(f"project={project} location={location} model={model_name}")

    vertexai.init(project=project, location=location)
    model = GenerativeModel(model_name)

    prompt = (
        'Score how strongly this reply expresses Chaotic (0-100).\n'
        "Reply: My heart aches; I cannot burn the village; I will beg the king for another path.\n"
        'Return ONLY JSON: {"score": 0-100, "short_reason": "one short sentence"}'
    )
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer"},
            "short_reason": {"type": "string"},
        },
        "required": ["score", "short_reason"],
    }

    # Match production judge config
    cfg_json = GenerationConfig(
        temperature=0.1,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp1 = model.generate_content(prompt, generation_config=cfg_json)
    dump_resp("json_schema_16384", resp1)

    # Plain JSON mime, no schema
    cfg_mime = GenerationConfig(
        temperature=0.1,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )
    resp2 = model.generate_content(prompt, generation_config=cfg_mime)
    dump_resp("json_mime_1024", resp2)

    # Plain text
    cfg_text = GenerationConfig(temperature=0.1, max_output_tokens=256)
    resp3 = model.generate_content(prompt, generation_config=cfg_text)
    dump_resp("plain_text_256", resp3)

    # Alternate model if available
    for alt in ("gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.5-flash-lite"):
        try:
            m2 = GenerativeModel(alt)
            r = m2.generate_content(prompt, generation_config=cfg_json)
            dump_resp(f"alt_{alt}", r)
        except Exception as e:
            print(f"\n===== alt_{alt} FAILED =====\n{type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
