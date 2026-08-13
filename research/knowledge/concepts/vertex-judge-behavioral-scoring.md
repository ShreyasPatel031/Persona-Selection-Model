# Vertex Judge Behavioral Scoring

**Slug:** `vertex-judge-behavioral-scoring`  
**Level:** project  
**Status:** complete

## Definition

The Vertex judge scores model responses 0–100 on how well they express a target trait, using Gemini on Vertex AI. Scores drive rollout filtering (pos/neg separation), quality gates, steering validation, and all SAE K-sweep experiments.

## Prerequisites (parents)

- [Persona Vectors pipeline](persona-vectors-pipeline.md)

## Used by (children)

- [Step C rollouts and judge](step-c-rollouts-judge.md)
- [Quality gates](quality-gates.md)
- [Coherence alpha sweep](coherence-alpha-sweep.md)
- [SAE-SSV](sae-ssv.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — behavioral evaluation framework

## In this repo

- `app/persona/judge_vertex.py`
- Used in `quality_gates.py`, `sae_ssv_optimize.py`, `ssv_omp_dsweep.py`

## Notes / open questions

Judge is the behavioral ground truth for all steering success metrics in this repo.
