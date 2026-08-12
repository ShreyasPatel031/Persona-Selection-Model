# Transformer Block

**Slug:** `transformer-block`  
**Level:** axiom  
**Status:** complete

## Definition

A transformer block at layer ℓ consists of multi-head self-attention followed by an MLP (feed-forward network), each with layer norm and residual connections. Blocks compose depth: the model stacks L such blocks to transform token embeddings into final representations.

## Prerequisites (parents)

- [Autoregressive LM](autoregressive-lm.md)
- [Residual stream](residual-stream.md)

## Used by (children)

- [Hidden states](hidden-states.md)
- [Causal layer selection](../concepts/causal-layer-selection.md) — which block to steer at

## Papers

_General architecture literature._

## In this repo

- Gemma-3-4B-IT has 34 layers (ℓ = 0 … 33); most persona work uses ℓ ≈ 15–16
- `app/persona/layer_heuristics.py` — warns against heuristic layer pick without causal sweep

## Notes / open questions

Layer index conventions: this repo uses 0-based layer numbers matching model config and SAE hook names.
