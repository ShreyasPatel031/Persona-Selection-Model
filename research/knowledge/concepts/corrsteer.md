# CorrSteer

**Slug:** `corrsteer`  
**Level:** method  
**Status:** draft

## Definition

CorrSteer selects SAE features by correlating their activations **during token-by-token generation** with task outcome scores, then intervenes on selected features. Differs from offline F-stat on precomputed contrastive activations.

## Prerequisites (parents)

- [Correlation vs causation](correlation-vs-causation.md)
- [SAE sparse basis](sae-sparse-basis.md)
- [F-stat feature ranking](f-stat-feature-ranking.md) — what we did instead

## Used by (children)

_None — proposed next step._

## Papers

- [Soo et al. 2025](../papers/soo-2025-corrsteer.md)

## In this repo

- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — Part 3; proposed for good trait
- **Not implemented**

## Notes / open questions

Identified as epistemically correct next step for prior-resident good.
