# Polysemanticity

**Slug:** `polysemanticity`  
**Level:** foundation  
**Status:** complete

## Definition

A neuron or SAE feature is **polysemantic** if it activates across multiple unrelated contexts or concepts. Polysemantic features are common under superposition. Trait steering that requires **many neurons firing together with coordinated signed weights** is evidence of polysemantic, distributed representation of alignment traits.

## Prerequisites (parents)

- [Superposition](superposition.md)

## Used by (children)

- [SAE-SSV](../concepts/sae-ssv.md) — joint multi-feature optimization
- [Per-feature clamp dead end](../concepts/per-feature-clamp-dead-end.md)
- [Monosemanticity claim](../concepts/monosemanticity-claim.md) — ideal vs reality

## Papers

- [Schubert et al. — OpenAI sparse autoencoders](https://openai.com/index/extracting-concepts-from-gpt-4/) (external)

## In this repo

- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — Good/Evil/Lawful/Chaotic require K joint SAE features
- Bubble viz shows polysemantic clusters (e.g. Good K=100: manipulation suppression + care amplification)

## Notes / open questions

French positive control (single SAE neuron) works because one feature is sufficient — traits are not single-neuron phenomena.
