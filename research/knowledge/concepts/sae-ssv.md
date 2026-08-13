# SAE-SSV

**Slug:** `sae-ssv`  
**Level:** method  
**Status:** complete

## Definition

SAE-SSV (Supervised Steering Vector) optimizes a joint weight vector v over top-K F-stat SAE features using the **L_steer** objective, then decodes v_res = W_dec^T v for residual add steering. Unlike independent per-feature clamps, SSV coordinates signed weights across many polysemantic features — the breakthrough sparse method for D&D traits.

## Prerequisites (parents)

- [F-stat feature ranking](f-stat-feature-ranking.md)
- [SAE W_enc / W_dec](sae-enc-dec.md)
- [Residual add steering](residual-add-steering.md)
- [Polysemanticity](../axioms/polysemanticity.md)

## Used by (children)

- [Logit lens features](logit-lens-features.md) — bubble viz interpretation

## Papers

- [He et al. 2025](../papers/he-2025-sae-ssv.md)

## In this repo

- `scripts/sae_ssv_optimize.py` — full pipeline
- `scripts/_remote_sae_ssv_all_traits.sh`
- Output: `persona_runs/<run_id>/sae/sae_ssv_results_262k_l{L}.json`
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — success on all four traits

## Notes / open questions

Optimal K varies by trait (Lawful peaks ~50–100; Evil ~750+). Chaotic can have negative cosine to dense v at low K.
