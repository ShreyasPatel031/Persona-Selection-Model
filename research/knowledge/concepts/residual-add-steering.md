# Residual Add Steering

**Slug:** `residual-add-steering`  
**Level:** method  
**Status:** complete

## Definition

Residual add steering modifies generation by adding a scaled direction to hidden states during the forward pass: **h ← h + α·v** at hook layer ℓ. α controls steering strength; v may be a dense persona vector or decoded SAE direction.

## Prerequisites (parents)

- [Causal intervention on activations](../axioms/causal-intervention-on-activations.md)
- [Residual stream](../axioms/residual-stream.md)
- [Linear representation hypothesis](../axioms/linear-representation-hypothesis.md)

## Used by (children)

- [Dense CAA steering](dense-caa-steering.md)
- [SAE-SSV](sae-ssv.md)
- [Coherence alpha sweep](coherence-alpha-sweep.md)
- [Quality gates](quality-gates.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — §3.2

## In this repo

- `app/persona/steering_demo.py` — dense steering hook
- `scripts/sae_ssv_optimize.py` — `sae_steer_hook_fn` with decoded v_res
- `app/main.py` — optional persona steer in `/chat`

## Notes / open questions

Chen et al. use raw α·v without unit-normalizing v. Neg system prompt used during eval to suppress default alignment.
