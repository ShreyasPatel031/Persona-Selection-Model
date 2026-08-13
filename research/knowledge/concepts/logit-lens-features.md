# Logit Lens Features

**Slug:** `logit-lens-features`  
**Level:** method  
**Status:** complete

## Definition

Logit lens features interprets individual SAE decoder columns W_dec[f] by projecting through lm_head to top vocabulary tokens. Provides semantic labels for bubble viz, cluster analysis, and human-readable feature reports — correlational, not causal.

## Prerequisites (parents)

- [Logit lens](../axioms/logit-lens.md)
- [SAE W_enc / W_dec](sae-enc-dec.md)

## Used by (children)

_None — leaf interpretability tool._

## Papers

- [Gemma Scope 2 / SAELens](../papers/gemma-scope-2-saelens.md)

## In this repo

- `scripts/ssv_feature_logit_lens.py`, `ssv_lens_themes.py`
- `app/static/ssv_bubble_viz.html` — K-slider with logit-lens labels
- `scripts/ssv_cluster_causal.py` — cluster + ablation

## Notes / open questions

Labels from Gemma Scope 2 pretrained logit lens; themes clustered in `ssv_lens_themes.py`.
