# Step B: Trait Bundle

**Slug:** `step-b-trait-bundle`  
**Level:** project  
**Status:** complete

## Definition

Step B generates the **trait bundle**: contrastive prompt pairs (pos/neg system prompts), eval questions, and metadata for a target trait (e.g. Good, Evil). Chen et al. §2.1 specifies 5 contrast pairs and 20 eval questions per trait.

## Prerequisites (parents)

- [Persona Vectors pipeline](persona-vectors-pipeline.md)

## Used by (children)

- [Step C rollouts and judge](step-c-rollouts-judge.md)

## Papers

- [Chen et al. 2025](../papers/chen-2025-persona-vectors.md) — §2.1

## In this repo

- `app/persona/artifact_gen.py`
- `app/persona/schemas.py` — contrast pair schema
- Output: `persona_runs/<run_id>/artifacts/trait_bundle.json`

## Notes / open questions

D&D traits use alignment-framed system prompts (Good/Evil/Lawful/Chaotic).
