# Step D: Vector Extraction

**Slug:** `step-d-vector-extraction`  
**Level:** project  
**Status:** complete

## Definition

Step D extracts persona vectors v_ℓ for each layer ℓ by mean-pooling assistant-span hidden states from kept pos and neg rollouts and computing v_ℓ = mean(h_pos) − mean(h_neg). Saves `persona_vectors.pt` for steering and SAE decomposition.

## Prerequisites (parents)

- [Step C rollouts and judge](step-c-rollouts-judge.md)
- [Contrastive activation averaging](contrastive-activation-averaging.md)
- [Hidden states](../axioms/hidden-states.md)

## Used by (children)

- [Causal layer selection](causal-layer-selection.md)
- [Dense CAA steering](dense-caa-steering.md)
- [F-stat feature ranking](f-stat-feature-ranking.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — §2.2

## In this repo

- `app/persona/activations.py` — `extract_persona_vectors()`
- Output: `persona_runs/<run_id>/vectors/persona_vectors.pt`

## Notes / open questions

Pooling restricted to assistant tokens, not user/system spans.
