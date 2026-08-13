# Gemma Scope 2 and SAELens

**Authors:** Google DeepMind (Gemma Scope 2); SAELens maintainers  
**Venue:** tooling / pretrained releases  
**URL:** https://decoderesearch.github.io/SAELens/  
**Status:** complete

## Key claims

- Public pretrained sparse autoencoders for Gemma models at multiple layers and widths (16k, 262k).
- SAELens provides encode/decode, hook integration, and logit-lens utilities.
- Default in this repo: `gemma-scope-2-4b-it-res`, layer 22 16k (Phase 2); trait SAE work often uses L15 262k.

## Concepts introduced or grounded

- [SAE sparse basis](../concepts/sae-sparse-basis.md)
- [SAE W_enc / W_dec](../concepts/sae-enc-dec.md)
- [Logit lens features](../concepts/logit-lens-features.md)

## In this repo

- `app/phase2.py` — SAE loading via SAELens
- `scripts/trait_sae_config.py` — trait → run_id, layer, SAE width mapping
- `app/static/phase2.html`, bubble viz pages — Gemma Scope 2 logit lens labels

## Notes

Not a research paper; infrastructure node for all SAE experiments.
