# Causal Intervention on Activations

**Slug:** `causal-intervention-on-activations`  
**Level:** foundation  
**Status:** complete

## Definition

Causal intervention on activations means modifying hidden states during a forward pass (add, replace, ablate, patch) and observing the effect on downstream behavior. Unlike correlational analysis of stored activations, interventions test whether a representation **causes** a model output or behavior change.

## Prerequisites (parents)

- [Hidden states](hidden-states.md) — what we intervene on
- [Autoregressive LM](autoregressive-lm.md) — generation provides the behavioral readout

## Used by (children)

- [Residual add steering](../concepts/residual-add-steering.md)
- [Encode-modify-decode clamp](../concepts/encode-modify-decode-clamp.md)
- [Sufficiency vs necessity](../concepts/sufficiency-vs-necessity.md)
- [Activation patching](../concepts/activation-patching.md)

## Papers

- [Nanda & Heimersheim](../papers/nanda-heimersheim-patching.md) — patching as causal tool
- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — steering as residual intervention

## In this repo

- `app/persona/sae_causality.py` — `sae_steer_hook_fn` (clamp, full_replacement modes)
- `scripts/ablation_necessity_sweep.py` — necessity tests via ablation
- `scripts/sae_clamp_experiment.py` — per-feature causal clamp experiments

## Notes / open questions

Addition-based steering tests **sufficiency** ("does adding v elicit trait?"). Ablation tests **necessity** ("does removing feature break trait?"). The correct test depends on whether the trait is prior-resident.
