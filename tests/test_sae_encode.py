"""Unit tests for SAE encode / attribution helpers (no model load)."""

from __future__ import annotations

import torch

from app.persona.sae_common import compute_feature_attribution


def test_compute_feature_attribution_shared_sign():
    d_sae = 8
    questions = []
    for _ in range(3):
        z_neg = torch.zeros(d_sae)
        z_pos = torch.zeros(d_sae)
        z_st = torch.zeros(d_sae)
        z_pos[1] = 2.0
        z_pos[3] = -1.5
        z_st[1] = 1.0
        z_st[3] = -0.8
        questions.append(
            {
                "z_pos_mean": z_pos,
                "z_neg_mean": z_neg,
                "z_steered": {"2": z_st},
            }
        )

    attr = compute_feature_attribution(questions, steered_alpha_key="2", top_k=5)
    pos_ids = [r["feature_id"] for r in attr["top_positive_features"]]
    neg_ids = [r["feature_id"] for r in attr["top_negative_features"]]
    assert 1 in pos_ids
    assert 3 in neg_ids
    assert attr["n_questions"] == 3
