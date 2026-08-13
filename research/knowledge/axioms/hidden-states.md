# Hidden States

**Slug:** `hidden-states`  
**Level:** axiom  
**Status:** complete

## Definition

Hidden states **h_ℓ** are the d_model-dimensional vectors in the residual stream at layer ℓ and token position t. Persona vectors are computed as differences of mean hidden states over assistant-span tokens from contrastive rollout pairs.

## Prerequisites (parents)

- [Residual stream](residual-stream.md) — h_ℓ lives in the stream at layer ℓ
- [Transformer block](transformer-block.md) — blocks transform h_ℓ → h_{ℓ+1}

## Used by (children)

- [Contrastive activation averaging](../concepts/contrastive-activation-averaging.md)
- [Persona vector extraction](../concepts/step-d-vector-extraction.md)
- [SAE sparse basis](../concepts/sae-sparse-basis.md) — SAEs encode h into sparse z

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — §2.2 mean pooling over assistant tokens

## In this repo

- `app/persona/activations.py` — `extract_persona_vectors()`, assistant-span mean pooling
- `app/persona/sae_encode.py` — encode hidden states into SAE latents

## Notes / open questions

Pooling is over **assistant** tokens only (not user/system), matching Chen et al. §2.2.
