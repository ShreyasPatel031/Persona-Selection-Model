# Dense CAA Steering

**Slug:** `dense-caa-steering`  
**Level:** method  
**Status:** complete

## Definition

Dense CAA steering applies the full d_model-dimensional persona vector v_ℓ via residual add at the causally selected layer. It is the behavioral baseline against which all sparse SAE methods are compared.

## Prerequisites (parents)

- [Contrastive activation averaging](contrastive-activation-averaging.md)
- [Residual add steering](residual-add-steering.md)
- [Causal layer selection](causal-layer-selection.md)

## Used by (children)

- [Quality gates](quality-gates.md)
- [D&D alignment grid](dnd-alignment-grid.md)
- [OMP decomposition](omp-decomposition.md) — OMP decomposes dense v
- [SAE-SSV](sae-ssv.md) — norm-matched calibration target

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md)

## In this repo

- `app/persona/steering_demo.py`, `coherence_alpha_sweep.py`
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — works for all four traits
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — Good at α=2 → 95.4

## Notes / open questions

Dense CAA is necessary but not sufficient for interpretable sparse decomposition.
