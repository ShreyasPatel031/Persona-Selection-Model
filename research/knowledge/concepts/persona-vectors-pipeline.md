# Persona Vectors Pipeline (Steps B/C/D)

**Slug:** `persona-vectors-pipeline`  
**Level:** project  
**Status:** complete

## Definition

The Persona Vectors pipeline extracts and validates steerable trait directions from a language model in three steps: **B** (trait artifact bundle), **C** (contrastive rollouts + judge filtering), **D** (vector extraction + quality gates). It is the foundation for all dense and sparse steering work in this repo.

## Prerequisites (parents)

- [Contrastive activation averaging](contrastive-activation-averaging.md)
- [Vertex judge behavioral scoring](vertex-judge-behavioral-scoring.md)
- [Quality gates](quality-gates.md)

## Used by (children)

- [D&D alignment grid](dnd-alignment-grid.md)
- [SAE-SSV](sae-ssv.md)
- [Gate self-chat experiment](gate-self-chat.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md)

## In this repo

- `app/persona/run.py` — CLI: `step-b`, `step-c`, `step-d`, `quality-gates`
- `README.md` — production requirements
- `docs/directory_structure.md` — `persona_runs/` layout

## Notes / open questions

Outputs live in gitignored `persona_runs/<run_id>/`.
