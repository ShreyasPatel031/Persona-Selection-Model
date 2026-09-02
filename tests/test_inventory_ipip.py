"""Unit tests for IPIP-50 item handling and constrained-Likert scoring helpers."""

from __future__ import annotations

import pytest

from app.persona.intensity_ladder import _likert_from_logits
from app.persona.inventory_ipip import (
    IPIP_50,
    LIKERT_OPTIONS,
    TRAITS,
    items_for_traits,
    item_user_message,
    response_validity,
    reverse_scored,
    score_traits,
)


def test_inventory_has_ten_items_per_trait_with_both_keyings():
    assert len(IPIP_50) == 50
    for trait in TRAITS:
        items = [it for it in IPIP_50 if it.trait == trait]
        assert len(items) == 10, trait
        assert any(it.keyed == 1 for it in items)
        assert any(it.keyed == -1 for it in items)


def test_items_for_traits_filters_and_rejects_unknown():
    only = items_for_traits(["openness"])
    assert {it.trait for it in only} == {"openness"}
    assert len(items_for_traits(None)) == 50
    with pytest.raises(ValueError):
        items_for_traits(["charisma"])


def test_item_user_message_presents_full_response_scale():
    msg = item_user_message(IPIP_50[0])
    assert IPIP_50[0].text in msg
    for opt in LIKERT_OPTIONS:
        assert f"{opt}. " in msg


def test_reverse_scored_flips_only_negative_keyed_items():
    assert reverse_scored(5, 1) == 5.0
    assert reverse_scored(5, -1) == 1.0
    assert reverse_scored(3, -1) == 3.0
    with pytest.raises(ValueError):
        reverse_scored(6, 1)


def test_score_traits_averages_reverse_keyed_responses():
    responses = [
        {"trait": "extraversion", "keyed": 1, "value": 5},
        {"trait": "extraversion", "keyed": -1, "value": 1},
        {"trait": "openness", "keyed": 1, "value": 2},
        {"trait": "openness", "keyed": 1, "value": None},
    ]
    scores = score_traits(responses)
    assert scores["extraversion"] == 5.0
    assert scores["openness"] == 2.0
    assert response_validity(responses) == 0.75


def test_score_traits_ignores_all_invalid_trait():
    assert score_traits([{"trait": "openness", "keyed": 1, "value": None}]) == {}


def test_likert_from_logits_picks_argmax_over_option_tokens_only():
    logits = [0.0] * 10
    logits[7] = 9.0  # not an option token: must be ignored
    logits[3] = 2.0
    option_ids = {"1": [1], "2": [2], "3": [3], "4": [4], "5": [5, 99]}

    import torch

    assert _likert_from_logits(torch.tensor(logits), option_ids) == 3
    assert _likert_from_logits(torch.tensor([0.0] * 3), {"1": [50]}) is None
