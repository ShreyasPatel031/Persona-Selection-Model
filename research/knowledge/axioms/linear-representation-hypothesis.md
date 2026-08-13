# Linear Representation Hypothesis

**Slug:** `linear-representation-hypothesis`  
**Level:** foundation  
**Status:** complete

## Definition

The linear representation hypothesis holds that many semantic and behavioral features of language models are encoded as **directions** (or low-dimensional subspaces) in activation space. Steering by adding a vector v to h is meaningful because concepts correspond to approximately linear structure in the residual stream.

## Prerequisites (parents)

- [Hidden states](hidden-states.md) — directions live in d_model space
- [Superposition](superposition.md) — many features may share dimensions

## Used by (children)

- [Contrastive activation averaging](../concepts/contrastive-activation-averaging.md)
- [Residual add steering](../concepts/residual-add-steering.md)
- [Dense CAA steering](../concepts/dense-caa-steering.md)
- [Non-identifiability](../concepts/non-identifiability.md) — many equivalent directions

## Papers

- [Park et al. — The Linear Representation Hypothesis](https://arxiv.org/abs/2311.03643) (external)
- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — persona vectors as linear directions

## In this repo

- Entire persona pipeline assumes v_ℓ is a steerable direction at layer ℓ
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — linear steering works behaviorally but direction is non-unique

## Notes / open questions

Linear steering is sufficient for behavioral control but does not guarantee unique or interpretable internal representations.
