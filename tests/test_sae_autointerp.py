"""Tests for SAE auto-interp helpers."""

from __future__ import annotations

from app.persona.sae_autointerp import (
    build_explanation_prompt,
    explanation_from_neuronpedia,
    neuronpedia_source_set,
    parse_sae_id,
    token_heuristic_explanation,
)


def test_parse_sae_id():
    p = parse_sae_id("layer_29_width_262k_l0_medium")
    assert p == {"layer": 29, "width": "262k", "l0": "medium"}
    assert parse_sae_id("bad") is None


def test_neuronpedia_source_set_res():
    assert (
        neuronpedia_source_set(
            "gemma-scope-2-4b-it-res",
            "layer_29_width_262k_l0_medium",
        )
        == "29-gemmascope-2-res-262k"
    )
    assert (
        neuronpedia_source_set(
            "gemma-scope-2-4b-it-res-all",
            "layer_31_width_16k_l0_small",
        )
        == "31-gemmascope-2-res-16k"
    )


def test_neuronpedia_source_set_transcoder():
    assert (
        neuronpedia_source_set(
            "gemma-scope-2-4b-it-transcoders-all",
            "layer_31_width_262k_l0_small",
        )
        == "31-gemmascope-2-transcoder-262k"
    )


def test_token_heuristic_explanation():
    expl = token_heuristic_explanation(
        [{"token": " diligent"}, {"token": " respect"}]
    )
    assert "diligent" in expl
    assert token_heuristic_explanation([]) == "(no token examples)"


def test_explanation_from_neuronpedia_pos_str_fallback():
    expl = explanation_from_neuronpedia({"pos_str": [" which", " whereby"]})
    assert expl is not None
    assert "which" in expl


def test_build_explanation_prompt_includes_tokens():
    prompt = build_explanation_prompt(
        55,
        [{"token": " respect", "activation": 1.0, "question_index": 0}],
        polarity="positive",
    )
    assert "Feature index: 55" in prompt
    assert "respect" in prompt
