# Quality Gates (0–3)

**Slug:** `quality-gates`  
**Level:** project  
**Status:** complete

## Definition

Automated quality gates validate persona vectors before steering use. **Gate 0:** data sufficiency (enough kept rollouts). **Gate 1:** pos/neg separability (projection margin). **Gate 2:** causal layer selection (Appendix B.4 α sweep). **Gate 3:** steering validation + coherence floor (≥75 trait, ≥80 coherence).

## Prerequisites (parents)

- [Causal layer selection](causal-layer-selection.md)
- [Vertex judge behavioral scoring](vertex-judge-behavioral-scoring.md)
- [Coherence alpha sweep](coherence-alpha-sweep.md)
- [Dense CAA steering](dense-caa-steering.md)

## Used by (children)

- [Persona Vectors pipeline](persona-vectors-pipeline.md)
- [D&D alignment grid](dnd-alignment-grid.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — thresholds in `quality_gates.py`

## In this repo

- `app/persona/quality_gates.py` — `PAPER_*` constants, gate runners
- Output: `persona_runs/<run_id>/eval/validation_report.json`

## Notes / open questions

`recommended_layer` and `recommended_alpha` from validation_report override `trait_sae_config.py` defaults.
