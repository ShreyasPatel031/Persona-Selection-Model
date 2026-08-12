# Step C: Rollouts and Judge

**Slug:** `step-c-rollouts-judge`  
**Level:** project  
**Status:** complete

## Definition

Step C runs contrastive rollouts: for each eval question, generate N responses under pos and neg system prompts, score each with a behavioral judge (0–100), and filter to kept pos (score > 50) and kept neg (score < 50). Paper scale: 10 rollouts/question, up to 1000/arm.

## Prerequisites (parents)

- [Step B trait bundle](step-b-trait-bundle.md)
- [Vertex judge behavioral scoring](vertex-judge-behavioral-scoring.md)

## Used by (children)

- [Step D vector extraction](step-d-vector-extraction.md)
- [F-stat feature ranking](f-stat-feature-ranking.md) — pos/neg labels for SAE

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — §2.2

## In this repo

- `app/persona/rollouts.py`, `judge_vertex.py`
- `app/persona/config.py` — judge filter thresholds
- Output: `persona_runs/<run_id>/rollouts/rollouts.jsonl`

## Notes / open questions

Behavioral scenario questions outperform simple eval items for stable vectors (checkpoint 001).
