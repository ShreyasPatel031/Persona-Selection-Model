# Vector Composition

**Slug:** `vector-composition`  
**Level:** project  
**Status:** complete

## Definition

Vector composition combines multiple persona directions with weights, optional orthogonalization, and norm budgets to steer toward compound traits. Supports dual-axis α calibration and Pareto-style tradeoffs between competing alignments.

## Prerequisites (parents)

- [Dense CAA steering](dense-caa-steering.md)
- [Linear representation hypothesis](../axioms/linear-representation-hypothesis.md)

## Used by (children)

- [D&D alignment grid](dnd-alignment-grid.md)
- [Gate self-chat experiment](gate-self-chat.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — base steering; composition extended locally

## In this repo

- `app/persona/vector_compose.py`
- `app/persona/grid_nine.py`

## Notes / open questions

Orthogonalization reduces interference when combining Lawful + Chaotic axes.
