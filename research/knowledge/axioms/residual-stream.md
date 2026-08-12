# Residual Stream

**Slug:** `residual-stream`  
**Level:** axiom  
**Status:** complete

## Definition

The residual stream is the main information highway in a transformer: a sequence of d_model-dimensional vectors, one per token position, that each sublayer (attention, MLP) reads from and writes to via skip connections. Steering and persona vectors operate by adding directions directly into this stream at a chosen layer.

## Prerequisites (parents)

- [Autoregressive LM](autoregressive-lm.md) — the stream carries token representations through the network

## Used by (children)

- [Hidden states](hidden-states.md)
- [Linear representation hypothesis](linear-representation-hypothesis.md)
- [Residual add steering](../concepts/residual-add-steering.md)
- [SAE sparse basis](../concepts/sae-sparse-basis.md)
- [Contrastive activation averaging](../concepts/contrastive-activation-averaging.md)

## Papers

- [Elhage et al. — A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — canonical residual-stream framing (external)

## In this repo

- `app/persona/steering_demo.py` — `h += α · v` hook on residual
- `app/persona/activations.py` — captures hidden states from residual stream
- `scripts/sae_ssv_optimize.py` — decodes SAE weights back into residual space

## Notes / open questions

Hook points in this repo are typically post-attention or block output at layer ℓ; exact hook name varies by SAE checkpoint (see Gemma Scope 2 config).
