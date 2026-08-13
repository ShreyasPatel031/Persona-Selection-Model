# Per-Feature Clamp Dead End

**Slug:** `per-feature-clamp-dead-end`  
**Level:** method  
**Status:** complete

## Definition

**Negative result:** independent per-feature encode-modify-decode clamps (STA top-K, z_steer−z_base top-K, OMP decode, STA projection) fail to elicit multi-neuron D&D traits. Only single-neuron phenomena (French positive control) succeed. Joint optimization (SAE-SSV) is required.

## Prerequisites (parents)

- [Encode-modify-decode clamp](encode-modify-decode-clamp.md)
- [Steering Target Atoms](steering-target-atoms.md)
- [OMP decomposition](omp-decomposition.md)
- [Polysemanticity](../axioms/polysemanticity.md)

## Used by (children)

_None — terminal negative result._

## Papers

- [Mayne et al. 2024](../papers/mayne-2024-sae-decomposition.md)

## In this repo

- `scripts/sae_clamp_experiment.py` — phases A/B/C
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — summary table

## Notes / open questions

Document dead ends explicitly — they constrain what sparse steering methods are worth pursuing.
