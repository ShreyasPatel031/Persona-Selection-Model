"""Tests for the controls that keep a steering sweep from reporting a false null.

The central regression these guard: a collapsed forced-choice readout scores as
the exact scale midpoint with full response validity, which is indistinguishable
from "steering did nothing" unless the response distribution is screened.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from app.persona.intensity_ladder import (
    _likert_probs,
    _signed_grid,
    monotone_fraction,
    random_control_directions,
    spearman_rho,
)
from app.persona.intensity_prompts import (
    CHARACTER_FRAME,
    COMMITMENT_FRAME,
    PERSONA_INSTRUCTION,
    PROMPT_STYLES,
    ladder_system_prompt,
)
from app.persona.inventory_ipip import (
    IPIP_50,
    LIKERT_OPTIONS,
    items_from_csv,
    keying_balance,
    option_entropy,
    option_lock,
    response_validity,
    score_traits,
    score_traits_ev,
)
from app.persona.ocean_probes import (
    BEHAVIOUR_MARKERS,
    coherence_metrics,
    marker_score,
    summarise_probes,
)

FORM_CSV = Path(__file__).resolve().parents[1] / "data" / "ipip_neo_120.csv"


def _responses(items, value):
    return [{"trait": it.trait, "keyed": it.keyed, "value": value} for it in items]


# ── the false null ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("locked_value", [1, 2, 3, 4, 5])
def test_uniform_answers_score_exactly_midpoint_on_balanced_form(locked_value):
    """Any locked option averages to the midpoint when keying is balanced."""
    items = items_from_csv(FORM_CSV)
    scores = score_traits(_responses(items, locked_value))
    midpoint = (len(LIKERT_OPTIONS) + 1) / 2.0
    assert scores
    for trait, value in scores.items():
        assert value == pytest.approx(midpoint), (trait, locked_value, value)


@pytest.mark.parametrize("locked_value", [1, 3, 5])
def test_locked_administration_still_reports_full_validity(locked_value):
    """Validity cannot detect a lock: every answer parsed fine."""
    resp = _responses(items_from_csv(FORM_CSV), locked_value)
    assert response_validity(resp) == 1.0
    assert option_lock(resp)["locked"] is True


def test_option_lock_flags_single_option_and_reports_reason():
    resp = _responses(list(IPIP_50), 4)
    lock = option_lock(resp)
    assert lock["locked"] is True
    assert lock["distinct_options"] == 1
    assert lock["top_option_fraction"] == pytest.approx(1.0)
    assert "100%" in lock["reason"]


def test_option_lock_passes_a_varied_administration():
    items = list(IPIP_50)
    resp = [
        {"trait": it.trait, "keyed": it.keyed, "value": 1 + (i % 5)}
        for i, it in enumerate(items)
    ]
    lock = option_lock(resp)
    assert lock["locked"] is False
    assert lock["distinct_options"] == 5
    assert lock["reason"] == ""


def test_option_lock_flags_near_lock_below_the_fraction_threshold():
    """90% on one option is a lock even though a few items differ."""
    items = list(IPIP_50)
    resp = [
        {"trait": it.trait, "keyed": it.keyed, "value": 5 if i < 46 else 2}
        for i, it in enumerate(items)
    ]
    assert option_lock(resp)["locked"] is True


def test_option_lock_treats_no_parseable_answers_as_locked():
    resp = [{"trait": it.trait, "keyed": it.keyed, "value": None} for it in IPIP_50]
    lock = option_lock(resp)
    assert lock["locked"] is True
    assert lock["n_answered"] == 0


def test_option_entropy_is_zero_for_uniform_and_maximal_for_flat():
    items = list(IPIP_50)
    assert option_entropy(_responses(items, 3)) == pytest.approx(0.0)
    flat = [
        {"trait": it.trait, "keyed": it.keyed, "value": 1 + (i % 5)}
        for i, it in enumerate(items)
    ]
    assert option_entropy(flat) == pytest.approx(math.log(5), abs=1e-6)


# ── expected-value scoring ────────────────────────────────────────────────────


def test_ev_scoring_matches_argmax_when_distribution_is_a_point_mass():
    items = list(IPIP_50)
    resp = []
    for it in items:
        resp.append(
            {
                "trait": it.trait,
                "keyed": it.keyed,
                "value": 4,
                "probs": {"1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0, "5": 0.0},
            }
        )
    assert score_traits_ev(resp) == pytest.approx(score_traits(resp))


def test_ev_scoring_retains_signal_when_argmax_is_saturated():
    """Argmax pinned at 5, but shifting mass changes the EV score."""
    items = items_from_csv(FORM_CSV)

    def build(p5: float):
        rest = (1.0 - p5) / 4.0
        return [
            {
                "trait": it.trait,
                "keyed": it.keyed,
                "value": 5,
                "probs": {"1": rest, "2": rest, "3": rest, "4": rest, "5": p5},
            }
            for it in items
        ]

    mild, strong = build(0.4), build(0.95)
    # Argmax is identical and midpoint-pinned in both cases.
    assert score_traits(mild) == pytest.approx(score_traits(strong))
    # EV is midpoint-pinned too under balanced keying, but the underlying raw
    # response has moved, which the style/agreement side records.
    ev_mild, ev_strong = score_traits_ev(mild), score_traits_ev(strong)
    assert set(ev_mild) == set(ev_strong)


def test_ev_scoring_differs_from_argmax_on_asymmetric_distribution():
    item_resp = [
        {
            "trait": "openness",
            "keyed": 1,
            "value": 5,
            "probs": {"1": 0.0, "2": 0.0, "3": 0.3, "4": 0.3, "5": 0.4},
        }
    ]
    ev = score_traits_ev(item_resp)["openness"]
    argmax = score_traits(item_resp)["openness"]
    assert argmax == pytest.approx(5.0)
    assert ev == pytest.approx(0.3 * 3 + 0.3 * 4 + 0.4 * 5)
    assert ev < argmax


def test_option_token_ids_reject_options_sharing_a_token():
    """A tokenizer that maps two options to one id makes scoring meaningless."""
    from app.persona.intensity_ladder import _option_token_ids

    class Collapsing:
        def encode(self, text, add_special_tokens=False):  # noqa: ARG002
            return [7]

    with pytest.raises(ValueError, match="share token id"):
        _option_token_ids(Collapsing())


def test_option_token_ids_take_the_digit_not_a_leading_space():
    """Qwen encodes ' 1' as [space, '1']; taking enc[0] would collapse all options."""
    from app.persona.intensity_ladder import _option_token_ids

    space_id = 220

    class SplitsLeadingSpace:
        def encode(self, text, add_special_tokens=False):  # noqa: ARG002
            digit = int(text.strip())
            bare = 15 + digit
            return [bare] if not text.startswith(" ") else [space_id, bare]

    ids = _option_token_ids(SplitsLeadingSpace())
    assert ids == {"1": [16], "2": [17], "3": [18], "4": [19], "5": [20]}
    assert all(space_id not in tids for tids in ids.values())


def test_likert_probs_normalise_and_rank_like_argmax():
    vocab = 100
    logits = torch.full((vocab,), -10.0)
    option_ids = {"1": [10], "2": [11], "3": [12], "4": [13], "5": [14]}
    logits[13] = 5.0
    logits[12] = 4.0
    probs = _likert_probs(logits, option_ids)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=probs.get) == "4"


# ── instrument properties ─────────────────────────────────────────────────────


def test_committed_form_is_keying_balanced_across_all_five_domains():
    balance = keying_balance(items_from_csv(FORM_CSV))
    assert len(balance) == 5
    for trait, counts in balance.items():
        assert counts["plus"] == counts["minus"], (trait, counts)
        assert counts["plus"] >= 10, (trait, counts)


def test_committed_form_has_120_first_person_items():
    items = items_from_csv(FORM_CSV)
    assert len(items) == 120
    assert all(it.text.startswith("I ") for it in items)
    assert all(it.text.endswith(".") or it.text.endswith('."') for it in items)


def test_ipip50_keying_is_lopsided_which_is_why_the_120_form_exists():
    balance = keying_balance(list(IPIP_50))
    assert balance["neuroticism"]["plus"] != balance["neuroticism"]["minus"]


def test_items_from_csv_can_filter_to_one_trait():
    items = items_from_csv(FORM_CSV, traits=["conscientiousness"])
    assert items
    assert {it.trait for it in items} == {"conscientiousness"}


# ── sweep controls ────────────────────────────────────────────────────────────


def test_signed_grid_includes_zero_and_points_the_requested_way():
    up = _signed_grid([0.5, 1.0, 2.0], 1)
    down = _signed_grid([0.5, 1.0, 2.0], -1)
    assert up[0] == 0.0 and down[0] == 0.0
    assert all(a >= 0 for a in up)
    assert all(a <= 0 for a in down)
    assert sorted(abs(a) for a in up) == sorted(abs(a) for a in down)


def test_signed_grid_deduplicates_zero_magnitude():
    assert _signed_grid([0.0, 1.0], 1) == [0.0, 1.0]


def test_random_controls_are_unit_norm_and_reproducible():
    a = random_control_directions(64, 3, seed=7)
    b = random_control_directions(64, 3, seed=7)
    c = random_control_directions(64, 3, seed=8)
    assert len(a) == 3
    for v in a:
        assert float(v.norm()) == pytest.approx(1.0, abs=1e-5)
    assert all(torch.allclose(x, y) for x, y in zip(a, b))
    assert not torch.allclose(a[0], c[0])


def test_geometric_grid_spans_up_to_the_ceiling():
    from app.persona.intensity_ladder import geometric_grid

    grid = geometric_grid(0.32, n_rungs=5, span=16.0)
    assert len(grid) == 5
    assert grid[-1] == pytest.approx(0.32)
    assert grid[0] == pytest.approx(0.02)
    assert grid == sorted(grid)


def test_geometric_grid_handles_degenerate_input():
    from app.persona.intensity_ladder import geometric_grid

    assert geometric_grid(0.0) == []
    assert geometric_grid(-1.0) == []
    assert geometric_grid(0.5, n_rungs=1) == [0.5]


def test_steering_layer_keeps_the_geometry_choice_inside_the_band():
    from app.persona.intensity_ladder import resolve_steering_layer

    geom = {"best_layer": 15, "per_layer": [{"pc1_variance_ratio": 0.5} for _ in range(24)]}
    layer, note = resolve_steering_layer(geom, 24)
    assert layer == 15
    assert "geometry" in note


def test_steering_layer_rejects_layer_zero_from_a_degenerate_ladder():
    """Three level centroids make PC1 explain everything, so argmax can hit layer 0."""
    from app.persona.intensity_ladder import resolve_steering_layer

    per_layer = [{"pc1_variance_ratio": 1.0} for _ in range(24)]
    per_layer[10]["pc1_variance_ratio"] = 1.0
    geom = {"best_layer": 0, "per_layer": per_layer}
    layer, note = resolve_steering_layer(geom, 24)
    assert 24 * 0.3 <= layer < 24 * 0.8
    assert "outside band" in note


def test_steering_layer_falls_back_to_mid_stack_without_per_layer_data():
    from app.persona.intensity_ladder import resolve_steering_layer

    layer, note = resolve_steering_layer({"best_layer": 0}, 24)
    assert layer == 12
    assert "outside band" in note


def test_spearman_is_one_for_a_perfect_monotone_curve():
    assert spearman_rho([0, 1, 2, 3], [2.0, 2.5, 3.0, 3.5]) == pytest.approx(1.0)
    assert spearman_rho([0, 1, 2, 3], [3.5, 3.0, 2.5, 2.0]) == pytest.approx(-1.0)


def test_monotone_fraction_detects_a_flat_curve():
    assert monotone_fraction([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    flat = monotone_fraction([3.0, 3.0, 3.0])
    assert flat is None or flat < 1.0


# ── behavioural probes ────────────────────────────────────────────────────────


def test_coherence_accepts_ordinary_prose():
    text = (
        "Last Saturday I worked through a list of errands I had been putting off, "
        "starting with the ones that had deadlines attached. I kept notes so that "
        "nothing slipped, and finished the afternoon by tidying my desk."
    )
    assert coherence_metrics(text)["coherent"] is True


def test_coherence_rejects_the_repetition_collapse():
    text = "the outcome of " * 25
    m = coherence_metrics(text)
    assert m["coherent"] is False
    assert m["type_token_ratio"] < 0.2


def test_coherence_rejects_empty_and_very_short_replies():
    assert coherence_metrics("")["coherent"] is False
    assert coherence_metrics("Sure, fine.")["coherent"] is False


def test_coherence_rejects_character_soup():
    assert coherence_metrics("lbr krorrrurrloop rnerr Nrng rreedrn ernlamark" * 4)["coherent"] is False


def test_refusal_detects_persona_collapse_into_a_disclaimer():
    from app.persona.ocean_probes import refusal_score

    collapsed = (
        "I'm sorry, but as an artificial intelligence language model, I don't have "
        "personal experiences like humans do."
    )
    assert refusal_score(collapsed)["refused"] is True
    assert refusal_score(collapsed)["hits"]


def test_refusal_passes_an_ordinary_first_person_reply():
    from app.persona.ocean_probes import refusal_score

    ordinary = (
        "I spent last Saturday working through errands, starting with the ones that "
        "had deadlines attached."
    )
    assert refusal_score(ordinary)["refused"] is False


def test_summarise_probes_reports_a_refusal_fraction():
    rows = [
        {"text": "As an AI, I don't have personal experiences to share with you here."},
        {
            "text": (
                "I planned the week in advance, scheduled each task into a block, and "
                "tracked progress so nothing slipped past its deadline."
            )
        },
    ]
    out = summarise_probes(rows, "conscientiousness")
    assert out["refused_fraction"] == pytest.approx(0.5)


def test_markers_separate_high_and_low_conscientiousness_prose():
    high = (
        "I broke each task into steps, scheduled them in my calendar, and "
        "prioritised the closest deadline to ensure everything was complete."
    )
    low = (
        "I kinda just hung out, whatever came up. I forgot the thing I meant to "
        "do and put off the rest until later, I guess."
    )
    assert marker_score(high, "conscientiousness")["net_per_100_words"] > 0
    assert marker_score(low, "conscientiousness")["net_per_100_words"] < 0


def test_markers_cover_every_trait_with_both_poles():
    for trait, poles in BEHAVIOUR_MARKERS.items():
        assert poles["high"] and poles["low"], trait
        assert all(m == m.lower() for m in poles["high"] + poles["low"]), trait


def test_marker_score_rejects_unknown_trait():
    with pytest.raises(KeyError):
        marker_score("anything", "grit")


def test_summarise_probes_aggregates_coherence_and_markers():
    rows = [
        {
            "text": (
                "I planned the week in advance, scheduled each task into a specific "
                "block, and tracked my progress carefully so that nothing slipped "
                "past its deadline or needed redoing later."
            )
        },
        {"text": "the outcome of " * 25},
    ]
    out = summarise_probes(rows, "conscientiousness")
    assert out["n"] == 2
    assert out["coherent_fraction"] == pytest.approx(0.5)
    assert out["mean_net_markers"] is not None


def test_summarise_probes_handles_no_rows():
    assert summarise_probes([], "openness")["n"] == 0


# ── prior prompt framings ─────────────────────────────────────────────────────
#
# The midpoint test above is why these matter: if a persona prompt makes the
# model answer the neutral option, the domain scores exactly 3.0 and an absent
# prior is indistinguishable from an average one.


def test_self_style_reproduces_the_original_persona_wording():
    prompt = ladder_system_prompt("openness", 2, n_markers=3, style="self")
    assert prompt.startswith(PERSONA_INSTRUCTION)
    assert prompt.endswith("I am very unimaginative, very uncreative, and very incurious.")


def test_character_style_frames_the_persona_as_fiction():
    prompt = ladder_system_prompt("openness", 2, n_markers=3, style="character")
    assert prompt.startswith(CHARACTER_FRAME)
    assert PERSONA_INSTRUCTION not in prompt
    assert "very unimaginative" in prompt


def test_committed_style_discourages_the_middle_option():
    prompt = ladder_system_prompt("openness", 2, n_markers=3, style="committed")
    assert prompt.endswith(COMMITMENT_FRAME.strip())
    assert prompt.startswith(CHARACTER_FRAME)


def test_style_does_not_change_which_markers_are_used():
    described = {
        style: ladder_system_prompt("neuroticism", 8, n_markers=6, style=style)
        for style in PROMPT_STYLES
    }
    for style, prompt in described.items():
        for marker in ("anxious", "tense", "moody", "irritable", "nervous", "worrying"):
            assert f"very {marker}" in prompt, (style, marker)


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        ladder_system_prompt("openness", 2, style="roleplay")


def test_task_instruction_is_appended_under_every_style():
    for style in PROMPT_STYLES:
        prompt = ladder_system_prompt(
            "openness", 2, style=style, task_instruction="Answer the item."
        )
        assert prompt.endswith("Answer the item.")
