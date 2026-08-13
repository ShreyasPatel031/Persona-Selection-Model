# Correlation vs Causation

**Slug:** `correlation-vs-causation`  
**Level:** meta  
**Status:** complete

## Definition

Correlation (feature active when trait present) does not imply causation (intervening on feature changes trait). F-stat ranks correlates on stored activations; steering tests causal sufficiency; ablation tests necessity. Each answers a different epistemic question.

## Prerequisites (parents)

- [F-stat feature ranking](f-stat-feature-ranking.md)
- [Causal intervention on activations](../axioms/causal-intervention-on-activations.md)

## Used by (children)

- [CorrSteer](corrsteer.md)
- [Output-side feature selection](output-side-feature-selection.md)

## Papers

- [Arad et al. 2025](../papers/arad-2025-sae-steering-features.md)
- [Soo et al. 2025 CorrSteer](../papers/soo-2025-corrsteer.md)

## In this repo

- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — Part 3, Part 6

## Notes / open questions

F-stat + optimize + steer is still sufficiency on diff-derived targets, not full causal loop.
