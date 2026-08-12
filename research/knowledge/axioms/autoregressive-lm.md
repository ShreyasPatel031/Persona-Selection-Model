# Autoregressive Language Model

**Slug:** `autoregressive-lm`  
**Level:** axiom  
**Status:** complete

## Definition

An autoregressive language model (LM) predicts the next token given all prior tokens in a sequence. Generation proceeds left-to-right: each forward pass produces logits over the vocabulary, a token is sampled or argmax-selected, appended to the context, and the process repeats.

## Prerequisites (parents)

_None — root axiom._

## Used by (children)

- [Residual stream](residual-stream.md)
- [Hidden states](hidden-states.md)
- [Transformer block](transformer-block.md)
- [Logit lens](../concepts/logit-lens-features.md)

## Papers

_General ML literature; no single canonical paper in this repo._

## In this repo

- `google/gemma-3-4b-it` — primary model for all persona and SAE experiments
- `app/main.py` — FastAPI chat server loading Gemma via Transformers

## Notes / open questions

All steering and interpretability work in this project assumes a decoder-only transformer LM unless stated otherwise.
