# Output-Side Feature Selection

**Slug:** `output-side-feature-selection`  
**Level:** method  
**Status:** complete

## Definition

Output-side feature selection ranks SAE features by their effect on model **outputs** (logits, gradients toward target tokens) rather than input activation strength alone. Identifies "driver" features vs "detector" features that correlate with but do not cause behavior.

## Prerequisites (parents)

- [Logit lens](../axioms/logit-lens.md)
- [SAE sparse basis](sae-sparse-basis.md)
- [F-stat feature ranking](f-stat-feature-ranking.md) — input-side baseline

## Used by (children)

- [Logit lens features](logit-lens-features.md)

## Papers

- [Arad et al. 2025](../papers/arad-2025-sae-steering-features.md)
- [Gur-Arieh et al. 2025](../papers/gur-arieh-2025-output-features.md)
- [GradSAE 2025](../papers/gradsae-2025.md)

## In this repo

- `scripts/causal_feature_screen.py` — GradSAE Phase A
- `scripts/sae_parallel_ruling.py` — Arad-style output relevance
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — not yet cleanly isolated in steering pipeline

## Notes / open questions

Identified as epistemically correct next step before steering; F-stat pool remains input-side.
