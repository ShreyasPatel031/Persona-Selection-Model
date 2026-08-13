# Superposition

**Slug:** `superposition`  
**Level:** foundation  
**Status:** complete

## Definition

Superposition is the phenomenon where a model represents more features than it has dimensions by encoding many features as nearly-orthogonal directions that overlap in activation space. A single neuron or dimension may participate in multiple concepts, making naive one-neuron-one-concept interpretation unreliable.

## Prerequisites (parents)

- [Linear representation hypothesis](linear-representation-hypothesis.md)
- [Hidden states](hidden-states.md)

## Used by (children)

- [Polysemanticity](polysemanticity.md)
- [SAE sparse basis](../concepts/sae-sparse-basis.md) — SAEs attempt to disentangle superposition
- [Monosemanticity claim](../concepts/monosemanticity-claim.md)

## Papers

- [Elhage et al. — Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (external)

## In this repo

- Motivation for using SAEs instead of raw neurons for trait steering
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — multi-neuron joint steering needed for traits

## Notes / open questions

Superposition explains why single-feature SAE clamps fail for polysemantic D&D traits.
