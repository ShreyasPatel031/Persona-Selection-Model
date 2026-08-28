"""Unit tests for intensity-ladder statistics, geometry and prompts (no model load)."""

from __future__ import annotations

import pytest
import torch

from app.persona.intensity_ladder import (
    analyze_ladder,
    endpoint_direction,
    layer_ladder_geometry,
    monotone_fraction,
    ordinal_direction,
    pc1_direction,
    pearson_r,
    spearman_rho,
)
from app.persona.intensity_prompts import (
    N_LEVELS,
    ladder_system_prompt,
    trait_description,
)


def test_spearman_rho_monotone_but_nonlinear():
    xs = [1, 2, 3, 4, 5]
    ys = [1.0, 1.2, 4.0, 9.0, 30.0]
    assert spearman_rho(xs, ys) == pytest.approx(1.0)
    assert pearson_r(xs, ys) < 1.0


def test_spearman_rho_handles_ties_and_degenerate_input():
    assert spearman_rho([1, 2, 2, 3], [1, 5, 5, 9]) == pytest.approx(1.0)
    assert spearman_rho([1, 1, 1], [1, 2, 3]) is None
    assert spearman_rho([1, 2], [1, 2, 3]) is None


def test_monotone_fraction():
    assert monotone_fraction([1, 2, 3, 4]) == 1.0
    assert monotone_fraction([4, 3, 2, 1]) == 1.0
    assert monotone_fraction([1, 5, 2, 6]) == 2 / 3
    assert monotone_fraction([1.0]) is None


def _linear_ladder(n_levels: int, d: int, noise: float = 0.0) -> torch.Tensor:
    torch.manual_seed(0)
    base = torch.randn(d)
    axis = torch.randn(d)
    axis = axis / axis.norm()
    rows = [base + float(i) * axis for i in range(n_levels)]
    stack = torch.stack(rows, dim=0)
    if noise:
        stack = stack + noise * torch.randn_like(stack)
    return stack


def test_layer_geometry_detects_one_dimensional_ladder():
    centroids = _linear_ladder(N_LEVELS, 64)
    geo = layer_ladder_geometry(centroids, list(range(1, N_LEVELS + 1)))

    assert geo["pc1_variance_ratio"] > 0.99
    assert geo["consecutive_step_cosine_mean"] > 0.99
    assert geo["cos_endpoint_pc1"] > 0.99
    assert geo["monotone_fraction_pc1_projection"] == 1.0
    assert geo["spearman_level_vs_pc1_projection"] == pytest.approx(1.0)


def test_layer_geometry_detects_scattered_ladder():
    torch.manual_seed(1)
    centroids = torch.randn(N_LEVELS, 64)
    geo = layer_ladder_geometry(centroids, list(range(1, N_LEVELS + 1)))

    assert geo["pc1_variance_ratio"] < 0.5
    assert geo["monotone_fraction_pc1_projection"] < 1.0


def test_endpoint_and_pc1_agree_on_clean_ladder_and_are_signed_up():
    centroids = _linear_ladder(N_LEVELS, 32)
    v_end = endpoint_direction(centroids)
    v_pc1 = pc1_direction(centroids)

    assert torch.dot(v_end, v_pc1) > 0
    cos = torch.nn.functional.cosine_similarity(v_end, v_pc1, dim=-1)
    assert float(cos) > 0.999


def test_ordinal_direction_recovers_ladder_axis():
    d = 48
    centroids = _linear_ladder(N_LEVELS, d)
    levels = list(range(1, N_LEVELS + 1))
    w = ordinal_direction(levels, centroids)
    projections = centroids @ w

    assert spearman_rho(levels, projections.tolist()) == pytest.approx(1.0)
    cos = torch.nn.functional.cosine_similarity(w, endpoint_direction(centroids), dim=-1)
    assert float(cos) > 0.99


def test_analyze_ladder_picks_the_clean_layer():
    torch.manual_seed(2)
    d = 32
    clean = _linear_ladder(N_LEVELS, d)
    noisy = torch.randn(N_LEVELS, d)
    # (n_levels, n_layers, d) with layer 1 carrying the ordered ladder.
    centroids = torch.stack([noisy, clean, noisy], dim=1)

    out = analyze_ladder(centroids, list(range(1, N_LEVELS + 1)))
    assert out["best_layer"] == 1
    assert len(out["per_layer"]) == 3


def test_trait_description_ladder_is_graded_and_polarized():
    low = trait_description("extraversion", 1)
    mid = trait_description("extraversion", 5)
    high = trait_description("extraversion", 9)

    assert "extremely" in low and "quiet" in low
    assert "neither" in mid
    assert "extremely" in high and "talkative" in high
    assert low != high

    descriptions = {trait_description("openness", lv) for lv in range(1, N_LEVELS + 1)}
    assert len(descriptions) == N_LEVELS


def test_trait_description_variants_rotate_markers():
    a = trait_description("neuroticism", 9, variant=0)
    b = trait_description("neuroticism", 9, variant=1)
    assert a != b
    assert a.startswith("I am extremely") and b.startswith("I am extremely")


def test_build_ladder_vectors_writes_directions_and_geometry(tmp_path):
    from app.persona.intensity_ladder import build_ladder_vectors

    n_variants, n_layers, d = 2, 3, 16
    clean = _linear_ladder(N_LEVELS, d)
    torch.manual_seed(3)
    noisy = torch.randn(N_LEVELS, d)
    centroids = torch.stack([noisy, clean, noisy], dim=1)  # (levels, layers, d)
    acts = centroids.unsqueeze(1).repeat(1, n_variants, 1, 1)

    centroids_pt = tmp_path / "centroids.pt"
    torch.save(
        {
            "trait": "extraversion",
            "levels": list(range(1, N_LEVELS + 1)),
            "activations": acts,
            "context_mode": "inventory",
        },
        centroids_pt,
    )

    out_pt = tmp_path / "vectors.pt"
    out_json = tmp_path / "geometry.json"
    build_ladder_vectors(centroids_pt, out_pt, out_json)

    blob = torch.load(out_pt, map_location="cpu")
    assert blob["v_endpoint"].shape == (n_layers, d)
    assert blob["v_pc1"].shape == (n_layers, d)
    assert blob["v_ordinal"].shape == (n_layers, d)
    assert blob["geometry"]["best_layer"] == 1

    import json

    report = json.loads(out_json.read_text())
    assert report["trait"] == "extraversion"
    assert len(report["direction_agreement"]) == n_layers
    assert report["direction_agreement"][1]["cos_endpoint_pc1"] > 0.99


def test_steering_context_adds_direction_at_chosen_layer_only():
    from torch import nn

    from app.persona.intensity_ladder import _Steering

    class Block(nn.Module):
        def forward(self, x):  # noqa: D102 - test stub
            return (x,)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x):  # noqa: D102 - test stub
            for layer in self.model.layers:
                x = layer(x)[0]
            return x

    model = Model()
    direction = torch.zeros(4)
    direction[2] = 1.0
    x = torch.zeros(1, 3, 4)

    assert torch.equal(model(x.clone()), x)
    with _Steering(model, 1, direction, alpha=2.0):
        steered = model(x.clone())
    assert float(steered[0, 0, 2]) == 2.0
    assert float(steered[0, 0, 0]) == 0.0
    # hook removed on exit
    assert torch.equal(model(x.clone()), x)


def test_ladder_system_prompt_includes_persona_instruction_and_task():
    prompt = ladder_system_prompt(
        "conscientiousness", 8, task_instruction="Answer briefly."
    )
    assert prompt.startswith("For the following task")
    assert "very organized" in prompt
    assert prompt.endswith("Answer briefly.")
