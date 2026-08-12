# Steering Target Atoms (STA)

**Slug:** `steering-target-atoms`  
**Level:** method  
**Status:** complete

## Definition

Steering Target Atoms (STA) ranks SAE features by gradient-based attribution to a steering target — which atoms causally contribute to moving behavior toward a goal. Alternative to F-stat for pre-steering feature selection.

## Prerequisites (parents)

- [SAE sparse basis](sae-sparse-basis.md)
- [Causal intervention on activations](../axioms/causal-intervention-on-activations.md)
- [Logit lens](../axioms/logit-lens.md)

## Used by (children)

- [Per-feature clamp dead end](per-feature-clamp-dead-end.md)

## Papers

- [Bricken STA](../papers/bricken-sta.md)

## In this repo

- `app/persona/sae_common.py` — `compute_sta_attribution()`
- `scripts/sta_phase1_sweep.py`, `sta_projection_test.py`

## Notes / open questions

STA top-K independent clamps did not recover multi-neuron traits.
