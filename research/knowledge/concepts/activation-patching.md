# Activation Patching

**Slug:** `activation-patching`  
**Level:** method  
**Status:** draft

## Definition

Activation patching replaces activations from a "source" forward pass into a "target" pass at specified layers and positions to test causal contribution of those activations to outputs. Attribution patching weights patches by gradient attribution.

## Prerequisites (parents)

- [Causal intervention on activations](../axioms/causal-intervention-on-activations.md)
- [Hidden states](../axioms/hidden-states.md)

## Used by (children)

- [Sufficiency vs necessity](sufficiency-vs-necessity.md)

## Papers

- [Nanda & Heimersheim](../papers/nanda-heimersheim-patching.md)

## In this repo

- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — reference only; not implemented

## Notes / open questions

Candidate methodology for necessity tests without contrastive diff targets.
