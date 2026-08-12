# Contrastive Diff vs Trait Content

**Slug:** `contrastive-diff-vs-trait-content`  
**Level:** meta  
**Status:** complete

## Definition

Contrastive persona vectors measure **transition from model prior** (prior → trait), not absolute trait content. For prior-resident Good, v_dense primarily encodes what must be *suppressed* (anti-lawful axis), not what "goodness" IS internally. OMP/SSV toward v_dense therefore interpret diff, not trait modules.

## Prerequisites (parents)

- [Contrastive activation averaging](contrastive-activation-averaging.md)
- [Prior-resident traits](prior-resident-traits.md)
- [Non-identifiability](non-identifiability.md)

## Used by (children)

_None — synthesizing meta concept._

## Papers

- [Non-identifiability 2026](../papers/non-identifiability-2026.md)

## In this repo

- [Checkpoint 002](../checkpoints/002-interpretability-causation-steering-conflict.md) — Part 6 refinement
- Feature 87091 lawful/chaotic opposite signs

## Notes / open questions

To interpret good-as-resident: correlation-select good features → **ablate** on neutral prompts.
