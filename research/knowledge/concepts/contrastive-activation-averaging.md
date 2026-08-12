# Contrastive Activation Averaging

**Slug:** `contrastive-activation-averaging`  
**Level:** method  
**Status:** complete

## Definition

Contrastive activation averaging (CAA) computes a direction v = mean(h_pos) − mean(h_neg) from hidden states of positive vs negative behavioral examples. It is the core extraction method for persona vectors in Chen et al.

## Prerequisites (parents)

- [Hidden states](../axioms/hidden-states.md)
- [Linear representation hypothesis](../axioms/linear-representation-hypothesis.md)

## Used by (children)

- [Step D vector extraction](step-d-vector-extraction.md)
- [Dense CAA steering](dense-caa-steering.md)
- [Contrastive diff vs trait content](contrastive-diff-vs-trait-content.md)
- [OMP decomposition](omp-decomposition.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — §2.2

## In this repo

- `app/persona/activations.py` — extraction over assistant-span tokens
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — dense CAA baseline

## Notes / open questions

CAA measures **transition from prior**, not absolute trait content — critical for prior-resident traits like Good.
