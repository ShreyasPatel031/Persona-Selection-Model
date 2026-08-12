# GradSAE: Output-Side Gradient Attribution for SAE Features

**Authors:** _(EMNLP 2025 authors)_  
**Venue:** EMNLP 2025  
**URL:** _(venue paper)_  
**Status:** complete

## Key claims

- Rank SAE latents by output-side gradient attribution — how much each feature's activation affects target logits.
- Identifies "driver" features more reliably than input activation alone.

## Concepts introduced or grounded

- [Output-side feature selection](../concepts/output-side-feature-selection.md)
- [Causal feature screening](../concepts/output-side-feature-selection.md)

## In this repo

- `scripts/causal_feature_screen.py` — Phase A: GradSAE-style attribution
- `scripts/sae_structure_synthesis.py` — compared against STA, clamp, OMP

## Notes

Phase A of causal feature screen; not yet run as full steering pipeline for traits.
