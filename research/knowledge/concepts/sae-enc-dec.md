# SAE W_enc / W_dec

**Slug:** `sae-enc-dec`  
**Level:** method  
**Status:** complete

## Definition

**W_enc** maps residual activations h → sparse latents z (with ReLU sparsity). **W_dec** maps latents back: each column W_dec[f] is the decoder direction for feature f in residual space. Steering decode: v_res = W_dec^T v where v is a weight vector in SAE space.

## Prerequisites (parents)

- [SAE sparse basis](sae-sparse-basis.md)
- [Hidden states](../axioms/hidden-states.md)

## Used by (children)

- [SAE-SSV](sae-ssv.md)
- [OMP decomposition](omp-decomposition.md)
- [Logit lens features](logit-lens-features.md)
- [Encode-modify-decode clamp](encode-modify-decode-clamp.md)

## Papers

- [Gemma Scope 2 / SAELens](../papers/gemma-scope-2-saelens.md)
- [He et al. 2025](../papers/he-2025-sae-ssv.md)

## In this repo

- `app/persona/sae_common.py` — encode/decode utilities
- `scripts/sae_ssv_optimize.py` — `W_dec^T @ v`

## Notes / open questions

Negative latent values are OOD for ReLU SAEs — explains failure modes with negative OMP/SSV coefficients.
