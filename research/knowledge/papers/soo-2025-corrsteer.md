# CorrSteer: Correlation During Generation for SAE Steering

**Authors:** Soo et al. (2025)  
**Venue:** arXiv / workshop  
**URL:** _(cited in checkpoint 002)_  
**Status:** complete

## Key claims

- Correlate SAE feature activations **during token-by-token generation** with task outcome scores.
- Select features active while the model is actually producing target behavior, then intervene.
- Differs from F-stat on precomputed contrastive activations (offline correlation).

## Concepts introduced or grounded

- [CorrSteer](../concepts/corrsteer.md)
- [Correlation vs causation](../concepts/correlation-vs-causation.md)
- [F-stat feature ranking](../concepts/f-stat-feature-ranking.md) — what we did instead

## In this repo

- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — Part 3 F-stat vs CorrSteer distinction; proposed replication for good trait

## Notes

Not yet implemented in this repo; identified as epistemically correct next step for prior-resident good.
