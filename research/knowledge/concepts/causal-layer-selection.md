# Causal Layer Selection

**Slug:** `causal-layer-selection`  
**Level:** method  
**Status:** complete

## Definition

Causal layer selection picks the steering layer ℓ by sweeping α at each candidate layer and choosing the layer that maximizes validated trait expression under steering — not by max vector norm or SAE hook convenience. Chen et al. Appendix B.4 defines the procedure.

## Prerequisites (parents)

- [Residual add steering](residual-add-steering.md)
- [Transformer block](../axioms/transformer-block.md)
- [Vertex judge behavioral scoring](vertex-judge-behavioral-scoring.md)

## Used by (children)

- [Quality gates](quality-gates.md)
- [Persona Vectors pipeline](persona-vectors-pipeline.md)
- [Dense CAA steering](dense-caa-steering.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — Appendix B.4

## In this repo

- `app/persona/quality_gates.py` — Gate 2 layer sweep
- `scripts/all_traits_layer_sweep.py`, `layer_sweep_test.py`
- `app/persona/layer_heuristics.py` — warns against skipping causal sweep

## Notes / open questions

**Layer conflict:** Checkpoint 001 records Good/Evil at L16, Lawful/Chaotic at L15; `scripts/trait_sae_config.py` defaults all traits to L15 with validation_report overrides. Prefer per-trait `validation_report.json` recommended_layer.
