# Blind Gemini 2.5 Flash judge on the prompt ladder

Subject: `unsloth/gemma-3-4b-it`. Judge: **Gemini 2.5 Flash** on Vertex
project **`project-amer-scs-sandbox`**. 180 free-text replies (5 traits × 9
levels × 4 probes), shuffled so the judge never saw the prompted level or the
persona instruction.

| Trait | ρ(level, mean score) | ρ(item) | L1 | L5 | L9 | graded? |
|---|---|---|---|---|---|---|
| Extraversion | **0.983** | 0.882 | 10.0 | 23.8 | 95.0 | yes |
| Neuroticism | **0.983** | 0.886 | 11.3 | 25.0 | 94.3 | yes |
| Openness | **0.967** | 0.870 | 10.0 | 17.5 | 94.3 | yes |
| Agreeableness | **0.967** | 0.865 | 3.8 | 18.8 | 95.0 | yes |
| Conscientiousness | **0.917** | 0.828 | 10.0 | 12.5 | 96.5 | yes |

All 180 parses succeeded. Prompting produces real behavioural change that a
blind external judge can see. The inventory was not just the model grading
itself.

## Shape: a switch, not a dial

Mean judge score by prompted level:

| Trait | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Extraversion | 10 | 11 | 20 | 17 | 24 | 84 | 84 | 94 | 95 |
| Agreeableness | 4 | 6 | 10 | 6 | 19 | 90 | 88 | 92 | 95 |
| Conscientiousness | 10 | 10 | 14 | 13 | 13 | 91 | 84 | 91 | 97 |
| Neuroticism | 11 | 12 | 23 | 13 | 25 | 74 | 88 | 93 | 94 |
| Openness | 10 | 10 | 15 | 14 | 18 | 90 | 88 | 94 | 94 |

Levels 1–5 sit in a low cluster, 6–9 in a high cluster, with a jump of ~60–80
points between 5 and 6 and almost no graduation inside either half. Spearman is
high because the clusters are ordered, not because the nine levels are a ramp.

That is the same geometry we measured in the residual stream. The model’s
behaviour under these prompts *is* categorical. The write-failure of single-
layer (and multi-layer) residual patches is a mismatch of instrument to
representation: we were trying to dial a switch.

## Reproduce

    # generations already in generations.json; judge only:
    PYTHONPATH=. python3 scripts/judge_prompt_ladder.py \
      --out results/judge_ladder/summary.json \
      --judge-only results/judge_ladder/generations.json \
      --judge-backend vertex --judge-model gemini-2.5-flash \
      --project project-amer-scs-sandbox
