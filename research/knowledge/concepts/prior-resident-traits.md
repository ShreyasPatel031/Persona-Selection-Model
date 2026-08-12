# Prior-Resident Traits

**Slug:** `prior-resident-traits`  
**Level:** meta  
**Status:** complete

## Definition

A prior-resident trait is already expressed in the model's default behavior (e.g. Good in RLHF-tuned Gemma-IT). Contrastive extraction then measures **suppression of default alignment**, not acquisition of trait content. Addition-based steering stacks on an already-saturated prior and misleads interpretability.

## Prerequisites (parents)

- [Contrastive activation averaging](contrastive-activation-averaging.md)
- [Reconstruction error / dark matter](reconstruction-error-dark-matter.md)

## Used by (children)

- [Sufficiency vs necessity](sufficiency-vs-necessity.md)
- [Contrastive diff vs trait content](contrastive-diff-vs-trait-content.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — contrastive framing
- [Checkpoint 002 empirical analysis](../checkpoints/002-interpretability-causation-steering-conflict.md)

## In this repo

- Good trait: dense CAA works (α=2 → 95.4) but sparse OMP/SSV interpret as anti-lawful suppression
- Evil trait: not prior-resident — sufficiency/addition is correct test

## Notes / open questions

| Trait | In prior? | Correct test |
|-------|-----------|--------------|
| evil | no | sufficiency (add) |
| good | yes | necessity (ablate) |
