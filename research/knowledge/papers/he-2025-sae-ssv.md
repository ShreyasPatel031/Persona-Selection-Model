# SAE-SSV: Supervised Steering Vector in SAE Latent Space

**Authors:** He et al. (2025)  
**Venue:** EMNLP 2025  
**URL:** _(venue paper; implementation in repo)_  
**Status:** complete

## Key claims

- Select top-K SAE features by F-statistic on labeled contrastive activations.
- Optimize joint weight vector v in SAE latent space via **L_steer** objective (supervised direction + LM regularizer).
- Decode v_res = W_dec^T v and steer in residual stream — multi-neuron coordinated steering.
- Stage 2 classifier ranking is an alternative feature-selection path.

## Concepts introduced or grounded

- [F-stat feature ranking](../concepts/f-stat-feature-ranking.md)
- [SAE-SSV](../concepts/sae-ssv.md)
- [SAE sparse basis](../concepts/sae-sparse-basis.md)

## In this repo

- `scripts/sae_ssv_optimize.py` — `optimize_v_steer()`, full K-sweep + judge
- `scripts/test_stage2_feasibility.py`, `scripts/ssv_stage2_test.py` — Stage 2 classifier path
- `scripts/probe_steer_sweep.py` — F-stat + supervised direction variant
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — breakthrough method for all four D&D traits

## Notes

This is the only sparse SAE method in this repo that elicits Good/Evil/Lawful/Chaotic jointly.
