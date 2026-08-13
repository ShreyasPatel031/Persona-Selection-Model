# SAEs Are Good for Steering — If You Select the Right Features

**Authors:** Arad et al. (2025)  
**Venue:** EMNLP 2025  
**URL:** _(venue paper)_  
**Status:** complete

## Key claims

- **Input features** (high activation when concept is present) and **output features** (whose decoder directions push logits toward concept) rarely co-occur.
- Output-side feature selection yields 2–3× better steering than input-side selection.
- Feature selection timing and computational role matter as much as correlation strength.

## Concepts introduced or grounded

- [Output-side feature selection](../concepts/output-side-feature-selection.md)
- [F-stat feature ranking](../concepts/f-stat-feature-ranking.md) — input-side baseline we used
- [Logit lens features](../concepts/logit-lens-features.md)

## In this repo

- `scripts/causal_feature_screen.py` — motivation for output-side selection
- `scripts/sae_parallel_ruling.py` — Arad-style output relevance ranking
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — input vs output discussion

## Notes

F-stat on stored activations is input-side; we have not cleanly isolated output-score filtering before steering.
