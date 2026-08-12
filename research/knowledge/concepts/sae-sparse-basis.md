# SAE Sparse Basis

**Slug:** `sae-sparse-basis`  
**Level:** method  
**Status:** complete

## Definition

A sparse autoencoder (SAE) learns an overcomplete dictionary that encodes residual-stream activations h into sparse latents z, then decodes back: h ≈ W_dec^T z + error. Pretrained Gemma Scope 2 SAEs provide a fixed sparse basis for interpretability and steering experiments.

## Prerequisites (parents)

- [Residual stream](../axioms/residual-stream.md)
- [Superposition](../axioms/superposition.md)

## Used by (children)

- [SAE W_enc / W_dec](sae-enc-dec.md)
- [Reconstruction error / dark matter](reconstruction-error-dark-matter.md)
- [F-stat feature ranking](f-stat-feature-ranking.md)
- [SAE-SSV](sae-ssv.md)
- [Encode-modify-decode clamp](encode-modify-decode-clamp.md)

## Papers

- [Gemma Scope 2 / SAELens](../papers/gemma-scope-2-saelens.md)

## In this repo

- `app/phase2.py`, `app/persona/sae_common.py`, `sae_encode.py`
- Widths: 16k and 262k features; hook typically L15 for trait work

## Notes / open questions

SAE is a change of basis, not a guaranteed monosemantic decomposition.
