# Persona Vectors: Monitoring and Controlling Character Traits in Language Models

**Authors:** Chen et al. (2025)  
**Venue:** arXiv  
**URL:** https://arxiv.org/abs/2507.21509  
**Status:** complete

## Key claims

- Persona vectors v_ℓ = mean(h_pos) − mean(h_neg) over contrastive rollouts capture character traits in residual space.
- Residual add steering h += α·v_ℓ at a causally selected layer controls trait expression at generation time.
- Quality gates (data sufficiency, separability, layer sweep, steering validation) ensure reliable vectors.
- Appendix B.4 defines causal layer selection via α grid sweep, not max-norm heuristics.

## Concepts introduced or grounded

- [Contrastive activation averaging](../concepts/contrastive-activation-averaging.md)
- [Persona Vectors pipeline](../concepts/persona-vectors-pipeline.md)
- [Step B trait bundle](../concepts/step-b-trait-bundle.md)
- [Step C rollouts and judge](../concepts/step-c-rollouts-judge.md)
- [Step D vector extraction](../concepts/step-d-vector-extraction.md)
- [Causal layer selection](../concepts/causal-layer-selection.md)
- [Quality gates](../concepts/quality-gates.md)
- [Dense CAA steering](../concepts/dense-caa-steering.md)

## In this repo

- Primary replication target — `README.md`, `docs/REPLICATION_EVIL_PAPER_V0.md`
- `app/persona/quality_gates.py` — `PAPER_*` thresholds, Gates 0–3
- `app/persona/activations.py`, `rollouts.py`, `run.py` — Steps B/C/D
- [Checkpoint 001](../checkpoints/001-sae-persona-steering.md) — dense CAA baseline reproduced

## Notes

Rollout scale: 10 rollouts/question, judge filter pos > 50 / neg < 50. Steering uses raw α·v (not unit-normalized v).
