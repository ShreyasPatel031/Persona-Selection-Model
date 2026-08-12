# Project-Only Subtree

Curated view: concepts directly used in the Persona Selection Model pipeline, from axioms to deployment.

## Reading order (recommended)

1. [Autoregressive LM](../axioms/autoregressive-lm.md) → [Residual stream](../axioms/residual-stream.md) → [Hidden states](../axioms/hidden-states.md)
2. [Contrastive activation averaging](../concepts/contrastive-activation-averaging.md) → [Persona Vectors pipeline](../concepts/persona-vectors-pipeline.md)
3. [Step B](../concepts/step-b-trait-bundle.md) → [Step C](../concepts/step-c-rollouts-judge.md) → [Step D](../concepts/step-d-vector-extraction.md)
4. [Quality gates](../concepts/quality-gates.md) → [Dense CAA steering](../concepts/dense-caa-steering.md)
5. [D&D alignment grid](../concepts/dnd-alignment-grid.md)

## SAE extension path

After dense pipeline works:

1. [SAE sparse basis](../concepts/sae-sparse-basis.md) → [F-stat feature ranking](../concepts/f-stat-feature-ranking.md) → [SAE-SSV](../concepts/sae-ssv.md)
2. Read [Per-feature clamp dead end](../concepts/per-feature-clamp-dead-end.md) before trying clamp/OMP shortcuts
3. Read [Checkpoint 002 epistemic limits](../concepts/prior-resident-traits.md) before claiming interpretability

## Key papers for this subtree

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — pipeline
- [He et al. 2025](../papers/he-2025-sae-ssv.md) — sparse breakthrough
- [Mayne et al. 2024](../papers/mayne-2024-sae-decomposition.md) — what not to claim
