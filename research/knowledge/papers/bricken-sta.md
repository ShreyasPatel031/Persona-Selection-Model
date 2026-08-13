# Steering Target Atoms (STA)

**Authors:** Bricken et al.  
**Venue:** Anthropic / interpretability research  
**URL:** _(method reference; no single arXiv in repo)_  
**Status:** complete

## Key claims

- Attribute steering-relevant SAE atoms via gradient-based "target atom" scoring.
- Identifies which sparse features causally contribute to a behavioral target.
- Alternative to F-stat or OMP for feature selection before steering.

## Concepts introduced or grounded

- [Steering Target Atoms](../concepts/steering-target-atoms.md)
- [Output-side feature selection](../concepts/output-side-feature-selection.md)

## In this repo

- `app/persona/sae_common.py` — `compute_sta_attribution()`
- `scripts/sta_phase1_sweep.py`, `sta_projection_test.py`, `sta_alpha_sweep_q0.py`
- `scripts/sae_structure_synthesis.py` — STA compared in synthesis report

## Notes

STA top-K independent clamps did not recover multi-neuron traits (see per-feature clamp dead end).
