# Logit Lens

**Slug:** `logit-lens`  
**Level:** foundation  
**Status:** complete

## Definition

The logit lens projects an internal activation h through the model's unembedding matrix W_U (and often the final layer norm) to obtain a distribution over vocabulary tokens. It answers: "if this hidden state were read out directly, which tokens would the model favor?" Used to interpret SAE decoder columns and steering directions.

## Prerequisites (parents)

- [Hidden states](hidden-states.md) — h is projected to logits
- [Autoregressive LM](autoregressive-lm.md) — W_U comes from the LM head

## Used by (children)

- [Logit lens features](../concepts/logit-lens-features.md)
- [Output-side feature selection](../concepts/output-side-feature-selection.md)
- [Steering Target Atoms](../concepts/steering-target-atoms.md)

## Papers

- [nostalgebraist — Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpKCwpCuMF) (blog, external)

## In this repo

- `scripts/ssv_feature_logit_lens.py` — decoder column → lm_head top tokens
- `scripts/ssv_lens_themes.py` — theme clustering from lens labels
- `app/static/ssv_bubble_viz.html` — bubble viz uses logit-lens labels from Gemma Scope 2

## Notes / open questions

Logit lens is correlational/interpretive, not causal. High top-token similarity does not prove a feature **drives** behavior (see output-side vs input-side distinction).
