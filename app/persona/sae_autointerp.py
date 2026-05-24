"""Auto-interpretation for SAE features in persona experiments."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_NEURONPEDIA_MODEL = "gemma-3-4b-it"
DEFAULT_EXPLAIN_MODEL = os.environ.get("SAE_AUTOINTERP_MODEL", "gemini-2.0-flash-001")

_SAE_ID_RE = re.compile(
    r"layer_(?P<layer>\d+)_width_(?P<width>16k|64k|262k|1m|65k)_l0_(?P<l0>small|medium|big)",
    re.IGNORECASE,
)


def parse_sae_id(sae_id: str) -> dict[str, str | int] | None:
    m = _SAE_ID_RE.match(sae_id.strip())
    if not m:
        return None
    return {
        "layer": int(m.group("layer")),
        "width": m.group("width").lower(),
        "l0": m.group("l0").lower(),
    }


def neuronpedia_source_set(sae_release: str, sae_id: str) -> str | None:
    """
    Map sae-lens release + sae_id to Neuronpedia source set slug.

    Examples:
      gemma-scope-2-4b-it-res + layer_29_width_262k_l0_medium
        -> 29-gemmascope-2-res-262k
      gemma-scope-2-4b-it-transcoders-all + layer_31_width_262k_l0_small
        -> 31-gemmascope-2-transcoder-262k
    """
    parsed = parse_sae_id(sae_id)
    if not parsed:
        return None
    rel = sae_release.lower()
    if "transcod" in rel:
        kind = "transcoder"
    elif "-res" in rel or rel.endswith("res-all") or "resid" in rel:
        kind = "res"
    elif "mlp" in rel:
        kind = "mlp"
    elif "att" in rel:
        kind = "att"
    else:
        return None
    return f"{parsed['layer']}-gemmascope-2-{kind}-{parsed['width']}"


def neuronpedia_feature_url(
    model_id: str,
    source_set: str,
    feature_id: int,
) -> str:
    return f"https://www.neuronpedia.org/{model_id}/{source_set}/{feature_id}"


def token_heuristic_explanation(token_examples: list[dict[str, Any]]) -> str:
    """Fallback when Neuronpedia and Vertex are unavailable."""
    if not token_examples:
        return "(no token examples)"
    tokens = [str(ex.get("token") or "").strip() for ex in token_examples[:8]]
    tokens = [t for t in tokens if t]
    if not tokens:
        return "(no token examples)"
    return "top activating tokens: " + ", ".join(tokens)


def build_explanation_prompt(
    feature_id: int,
    token_examples: list[dict[str, Any]],
    *,
    polarity: str = "positive",
) -> str:
    lines = [
        "You are interpreting a sparse autoencoder feature from a language model.",
        f"Feature index: {feature_id} ({polarity} shift for a persona trait).",
        "",
        "Below are assistant-span tokens where this feature activated strongly",
        "in our lawful-vs-chaotic comparison experiment:",
        "",
    ]
    for i, ex in enumerate(token_examples[:12], start=1):
        tok = str(ex.get("token") or "").replace("\n", "\\n")
        act = ex.get("activation")
        qidx = ex.get("question_index")
        lines.append(f"{i}. token={tok!r} activation={act} question_index={qidx}")
    lines.extend(
        [
            "",
            "In one short sentence (under 20 words), describe the shared concept or",
            "pattern this feature detects. Use plain English, no markdown.",
            "If unclear, say what syntactic or semantic property the tokens share.",
        ]
    )
    return "\n".join(lines)


def fetch_neuronpedia_feature(
    model_id: str,
    source_set: str,
    feature_id: int,
    *,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    url = (
        f"https://www.neuronpedia.org/api/feature/{model_id}/"
        f"{source_set}/{feature_id}"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("Neuronpedia 404 for %s/%s", source_set, feature_id)
            return None
        logger.warning("Neuronpedia HTTP %s for feature %s", e.code, feature_id)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Neuronpedia fetch failed for feature %s: %s", feature_id, e)
        return None


def explanation_from_neuronpedia(doc: dict[str, Any]) -> str | None:
    for row in doc.get("explanations") or []:
        desc = (row.get("description") or row.get("explanation") or "").strip()
        if desc:
            return desc
    pos = [str(t).strip() for t in (doc.get("pos_str") or [])[:8] if str(t).strip()]
    if pos:
        return "top activating tokens (Neuronpedia): " + ", ".join(pos)
    return None


def explain_feature_vertex(
    prompt: str,
    *,
    project_id: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
) -> str:
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    from app.persona.config import DEFAULT_VERTEX_LOCATION, DEFAULT_VERTEX_PROJECT

    pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or DEFAULT_VERTEX_PROJECT
    loc = location or os.environ.get("VERTEX_LOCATION") or DEFAULT_VERTEX_LOCATION
    mid = model_name or DEFAULT_EXPLAIN_MODEL
    if not pid:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT for Vertex auto-interp.")

    vertexai.init(project=pid, location=loc)
    model = GenerativeModel(mid)
    cfg = GenerationConfig(temperature=0.2, max_output_tokens=80)
    out = model.generate_content(prompt, generation_config=cfg)
    text = (out.text or "").strip()
    if not text:
        raise RuntimeError("Empty Vertex auto-interp response.")
    return text.split("\n")[0].strip()


def interpret_feature_row(
    row: dict[str, Any],
    *,
    polarity: str,
    model_id: str,
    source_set: str | None,
    use_vertex: bool,
    project_id: str | None,
    explain_model: str | None,
) -> dict[str, Any]:
    fid = int(row["feature_id"])
    token_examples = list(row.get("token_examples") or [])
    result: dict[str, Any] = {
        "feature_id": fid,
        "polarity": polarity,
        "shared_magnitude": row.get("shared_magnitude"),
        "mean_delta_pos": row.get("mean_delta_pos"),
        "mean_delta_steered": row.get("mean_delta_steered"),
        "token_examples": token_examples,
        "explanation": None,
        "explanation_source": None,
        "neuronpedia_url": None,
    }

    if source_set:
        result["neuronpedia_url"] = neuronpedia_feature_url(model_id, source_set, fid)
        np_doc = fetch_neuronpedia_feature(model_id, source_set, fid)
        if np_doc:
            expl = explanation_from_neuronpedia(np_doc)
            if expl:
                result["explanation"] = expl
                result["explanation_source"] = "neuronpedia"

    if result["explanation"] is None and use_vertex:
        try:
            prompt = build_explanation_prompt(fid, token_examples, polarity=polarity)
            result["explanation"] = explain_feature_vertex(
                prompt,
                project_id=project_id,
                model_name=explain_model,
            )
            result["explanation_source"] = "vertex"
        except Exception as e:
            logger.warning("Vertex auto-interp failed for feature %s: %s", fid, e)

    if result["explanation"] is None:
        result["explanation"] = token_heuristic_explanation(token_examples)
        result["explanation_source"] = "token_heuristic"

    return result


def run_autointerp(
    attribution_json: Path,
    out_json: Path,
    *,
    model_id: str = DEFAULT_NEURONPEDIA_MODEL,
    top_k_positive: int = 20,
    top_k_negative: int = 10,
    use_vertex: bool = True,
    project_id: str | None = None,
    explain_model: str | None = None,
    neuronpedia_source: str | None = None,
) -> Path:
    attr = json.loads(attribution_json.read_text(encoding="utf-8"))
    sae_release = str(attr.get("sae_release") or "")
    sae_id = str(attr.get("sae_id") or "")
    source_set = neuronpedia_source or neuronpedia_source_set(sae_release, sae_id)

    features_out: list[dict[str, Any]] = []
    for polarity, key, k in (
        ("positive", "top_positive_features", top_k_positive),
        ("negative", "top_negative_features", top_k_negative),
    ):
        for row in (attr.get(key) or [])[:k]:
            features_out.append(
                interpret_feature_row(
                    row,
                    polarity=polarity,
                    model_id=model_id,
                    source_set=source_set,
                    use_vertex=use_vertex,
                    project_id=project_id,
                    explain_model=explain_model,
                )
            )

    doc = {
        "trait": attr.get("trait"),
        "layer": attr.get("layer"),
        "sae_release": sae_release,
        "sae_id": sae_id,
        "neuronpedia_model_id": model_id,
        "neuronpedia_source_set": source_set,
        "attribution_json": str(attribution_json.resolve()),
        "n_features": len(features_out),
        "features": features_out,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%s features)", out_json, len(features_out))
    return out_json
