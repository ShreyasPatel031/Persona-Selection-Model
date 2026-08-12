# Misleading SAE Decompositions of Steering Vectors

**Authors:** Mayne et al. (2024)  
**Venue:** arXiv  
**URL:** https://arxiv.org/abs/2411.08790  
**Status:** complete

## Key claims

- Decomposing a steering vector into SAE features via OMP or similar yields misleading results.
- High cosine between decomposition and original vector does not imply equivalent steering effect.
- Negative SAE coefficients are out-of-distribution — models rarely see negative latents during training.
- Random linear combinations in SAE subspace can match cosine but fail to steer.

## Concepts introduced or grounded

- [OMP decomposition](../concepts/omp-decomposition.md)
- [Non-identifiability](../concepts/non-identifiability.md)
- [Per-feature clamp dead end](../concepts/per-feature-clamp-dead-end.md)

## In this repo

- `scripts/sae_structure_synthesis.py` — literature section + experiment interpretation
- `scripts/omp_decompose.py`, `scripts/omp_uniqueness_test.py` — OMP experiments
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — OMP non-uniqueness noted
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — good/chaotic negative-coeff failure

## Notes

Also discussed as OpenReview [QRpzG4b5dz](../papers/openreview-sae-decomposition.md).
