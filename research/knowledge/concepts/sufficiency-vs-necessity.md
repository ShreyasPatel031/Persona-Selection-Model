# Sufficiency vs Necessity

**Slug:** `sufficiency-vs-necessity`  
**Level:** meta  
**Status:** complete

## Definition

**Sufficiency:** adding/intervening on feature F elicits trait behavior (does F cause trait appearance?). **Necessity:** ablating F degrades trait behavior that was already present (is F required?). Addition tests sufficiency; ablation tests necessity. The correct test depends on whether the trait is prior-resident.

## Prerequisites (parents)

- [Causal intervention on activations](../axioms/causal-intervention-on-activations.md)
- [Prior-resident traits](prior-resident-traits.md)

## Used by (children)

- [Post-intervention recovery](post-intervention-recovery.md)

## Papers

- [Nanda & Heimersheim](../papers/nanda-heimersheim-patching.md)
- [Cui et al. 2026](../papers/cui-2026-sae-interventions.md)

## In this repo

- `scripts/ablation_necessity_sweep.py` — proposed necessity battery
- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — evil=sufficiency, good=necessity

## Notes / open questions

We ran sufficiency (addition/residual add) on diff-derived features; ablation for good is identified but not done.
