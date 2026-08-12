# Output-Side Feature Selection for SAE Steering

**Authors:** Gur-Arieh et al. (2025)  
**Venue:** EMNLP 2025 (related to Arad et al.)  
**URL:** _(venue paper)_  
**Status:** complete

## Key claims

- Rank SAE features by their effect on model outputs, not just input activation strength.
- Output-side gradients and relevance scores identify "driver" features vs "detector" features.

## Concepts introduced or grounded

- [Output-side feature selection](../concepts/output-side-feature-selection.md)
- [GradSAE](../papers/gradsae-2025.md) — gradient-based output attribution

## In this repo

- `scripts/causal_feature_screen.py` — cited as Gur-Arieh et al. motivation
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — reference list

## Notes

Often grouped with Arad et al. in checkpoint discussions; both address input/output feature mismatch.
