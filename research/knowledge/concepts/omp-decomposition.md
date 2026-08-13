# OMP Decomposition

**Slug:** `omp-decomposition`  
**Level:** method  
**Status:** complete

## Definition

Orthogonal Matching Pursuit (OMP) greedily decomposes dense persona vector v_dense into a sparse combination of SAE decoder columns. Used to find minimal feature sets that approximate v_dense — but high cosine ≠ equivalent steering, and negative coefficients are OOD.

## Prerequisites (parents)

- [SAE W_enc / W_dec](sae-enc-dec.md)
- [Dense CAA steering](dense-caa-steering.md)

## Used by (children)

- [Non-identifiability](non-identifiability.md)
- [Contrastive diff vs trait content](contrastive-diff-vs-trait-content.md)

## Papers

- [Mayne et al. 2024](../papers/mayne-2024-sae-decomposition.md)
- [OpenReview SAE decomposition](../papers/openreview-sae-decomposition.md)

## In this repo

- `scripts/omp_decompose.py`, `omp_uniqueness_test.py`, `ssv_omp_dsweep.py`
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — evil/lawful work; good/chaotic suppression axis

## Notes / open questions

Feature 87091: lawful +1158.5, chaotic −1158.5 — shared suppression axis, not independent traits.
