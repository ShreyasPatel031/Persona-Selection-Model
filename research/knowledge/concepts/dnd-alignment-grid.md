# D&D Alignment Grid

**Slug:** `dnd-alignment-grid`  
**Level:** project  
**Status:** complete

## Definition

The D&D alignment grid composes four persona vectors — Good, Evil, Lawful, Chaotic — on a dual-axis 2×2 (or extended 3×3) grid. Each grid point applies weighted combinations of trait vectors at calibrated α values to elicit compound alignments (e.g. Lawful Good).

## Prerequisites (parents)

- [Persona Vectors pipeline](persona-vectors-pipeline.md)
- [Dense CAA steering](dense-caa-steering.md)
- [Vector composition](vector-composition.md)
- [Coherence alpha sweep](coherence-alpha-sweep.md)

## Used by (children)

- [Gate self-chat experiment](gate-self-chat.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — multi-trait composition (extended in this repo)

## In this repo

- `app/persona/grid_nine.py`, `dnd_playground.py`
- `app/persona/vector_compose.py` — `dnd-grid`, `calibrate`
- `scripts/dnd_gemma_mvp.sh` — end-to-end VM workflow

## Notes / open questions

All four traits also targeted by SAE-SSV sparse steering (checkpoint 001).
