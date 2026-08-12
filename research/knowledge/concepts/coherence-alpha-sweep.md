# Coherence Alpha Sweep

**Slug:** `coherence-alpha-sweep`  
**Level:** project  
**Status:** complete

## Definition

Coherence alpha sweep calibrates steering strength α by sweeping α values at the selected layer and measuring both trait expression and response coherence (fluency/sensibleness). Gate 3 requires coherence ≥ 80 alongside trait ≥ 75.

## Prerequisites (parents)

- [Residual add steering](residual-add-steering.md)
- [Vertex judge behavioral scoring](vertex-judge-behavioral-scoring.md)

## Used by (children)

- [Quality gates](quality-gates.md)
- [D&D alignment grid](dnd-alignment-grid.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md)

## In this repo

- `app/persona/coherence_alpha_sweep.py`
- `app/persona/vector_compose.py` — `calibrate` command

## Notes / open questions

Good trait validated at α=2 (95.4 mean) per checkpoint 002.
