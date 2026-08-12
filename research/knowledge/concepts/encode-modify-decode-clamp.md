# Encode-Modify-Decode Clamp

**Slug:** `encode-modify-decode-clamp`  
**Level:** method  
**Status:** complete

## Definition

Encode-modify-decode clamp: encode h → z, modify one or more z_i (set, clamp, offset), decode h' = W_dec^T z'. **Full replacement** mode (Anthropic/Templeton style) replaces h entirely with decode(z'). Tests whether sparse feature edits alone can steer behavior.

## Prerequisites (parents)

- [SAE W_enc / W_dec](sae-enc-dec.md)
- [Causal intervention on activations](../axioms/causal-intervention-on-activations.md)

## Used by (children)

- [Per-feature clamp dead end](per-feature-clamp-dead-end.md)

## Papers

- [Gemma Scope 2 / SAELens](../papers/gemma-scope-2-saelens.md)

## In this repo

- `app/persona/sae_causality.py` — `sae_steer_hook_fn`, `full_replacement` mode
- `scripts/sae_clamp_experiment.py` — French positive control (single neuron)

## Notes / open questions

Works for **one-neuron** phenomena (French); fails for multi-neuron D&D traits without joint optimization (SSV).
