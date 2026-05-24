"""Unit tests for SAE causality diagnostics (mock SAE, no model load)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.persona.sae_causality import (
    check_hf_sae_checkpoint,
    cosine_similarity,
    full_reconstruction_metrics,
    latent_topk_mask,
    magnitude_matched_alpha,
    sparse_direction_from_latent,
    topk_reconstruction_sweep,
)


class _MockSAE:
    def __init__(self, d_in: int = 4, d_sae: int = 8) -> None:
        self.cfg = SimpleNamespace(d_in=d_in, d_sae=d_sae)
        self.W_dec = torch.eye(d_in, d_sae)[:, :d_sae]
        self._dev = torch.device("cpu")

    def parameters(self):
        yield self.W_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 1, d_in) -> latent picks first d_in dims
        z = x[..., : self.cfg.d_sae].clone()
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (1, 1, d_sae) -> (1, 1, d_in)
        out = torch.zeros(*z.shape[:-1], self.cfg.d_in, dtype=z.dtype, device=z.device)
        k = min(self.cfg.d_in, self.cfg.d_sae)
        out[..., :k] = z[..., :k]
        return out


def test_cosine_similarity_identical():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_full_reconstruction_metrics_identity_basis():
    sae = _MockSAE(d_in=4, d_sae=4)
    v = torch.tensor([1.0, 0.5, -0.25, 2.0])
    m = full_reconstruction_metrics(sae, v)
    assert m["cosine_full"] > 0.99
    assert m["n_active_features"] >= 1


def test_topk_sweep_monotonic_k():
    sae = _MockSAE(d_in=4, d_sae=4)
    v = torch.tensor([1.0, 0.5, -0.25, 2.0])
    rows = topk_reconstruction_sweep(sae, v, ks=[1, 2, 4])
    assert rows[0]["k"] == 1
    assert rows[-1]["cosine"] >= rows[0]["cosine"]


def test_latent_topk_mask():
    z = torch.tensor([0.1, 5.0, -3.0, 0.2])
    z2 = latent_topk_mask(z, 2)
    assert (z2.abs() > 0).sum().item() == 2


def test_sparse_direction_normalize_flag():
    sae = _MockSAE(d_in=4, d_sae=8)
    z = torch.zeros(8)
    z[0] = 1.0
    z[2] = 2.0
    raw = sparse_direction_from_latent(sae, z, normalize=False)
    normed = sparse_direction_from_latent(sae, z, normalize=True)
    assert abs(raw.norm().item() - 1.0) > 0.01
    assert abs(normed.norm().item() - 1.0) < 1e-5


def test_magnitude_matched_alpha():
    dense = torch.tensor([3.0, 4.0])  # norm 5
    sparse = torch.tensor([1.0, 0.0])  # norm 1
    a = magnitude_matched_alpha(dense, sparse, 2.0, normalize_sparse=False)
    assert abs(a - 10.0) < 1e-5
    a_norm = magnitude_matched_alpha(dense, sparse, 2.0, normalize_sparse=True)
    assert abs(a_norm - 10.0) < 1e-5


def test_check_hf_sae_checkpoint_layer31_262k():
    r = check_hf_sae_checkpoint(
        "gemma-scope-2-4b-it-res-all",
        "layer_31_width_262k_l0_medium",
    )
    assert "exists" in r
    assert r.get("sae_id") == "layer_31_width_262k_l0_medium"
