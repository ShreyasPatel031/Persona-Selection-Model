# F-stat Feature Ranking

**Slug:** `f-stat-feature-ranking`  
**Level:** method  
**Status:** complete

## Definition

F-stat feature ranking scores each SAE latent by ANOVA F-statistic separating pos vs neg activation distributions from contrastive rollouts. Top-K F-stat features form the candidate subspace for SAE-SSV optimization. This is **input-side, offline correlation** — not generation-time CorrSteer.

## Prerequisites (parents)

- [SAE sparse basis](sae-sparse-basis.md)
- [Step C rollouts and judge](step-c-rollouts-judge.md)

## Used by (children)

- [SAE-SSV](sae-ssv.md)
- [Output-side feature selection](output-side-feature-selection.md) — contrast / alternative

## Papers

- [He et al. 2025](../papers/he-2025-sae-ssv.md)
- [Arad et al. 2025](../papers/arad-2025-sae-steering-features.md) — input vs output critique

## In this repo

- `scripts/sae_ssv_optimize.py` — F-stat on 262k features, top-K pool (e.g. 1024)
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — F-stat did not save good interpretability

## Notes / open questions

F-stat selects detectors; Arad argues output-side drivers steer better.
