# Blind Gemini judge on the *vectors*, not the prompts

Subject `unsloth/gemma-3-4b-it`, steered at the saved PC1 rungs from
`results/gemma_final/` (persona-free prompt, one Saturday probe per rung).
Judge: Gemini 2.5 Flash, Vertex project `project-amer-scs-sandbox`. 120
texts shuffled (60 trait-vector + 60 matched random-control). The judge never
saw magnitude, pole, or inventory score.

`rho` is Spearman(signed magnitude, judge score). For a high pole, +rho means
the judge saw more of the trait as we pushed up. For a low pole, +rho means
the judge saw *less* of the trait as we pushed down (score falls as magnitude
goes more negative).

| Trait | pole | vector ρ | control ρ | judge @0 → strongest | inventory @0 → strongest |
|---|---|---|---|---|---|
| Extraversion | high | **+0.71** | −0.49 | 85 → 85 (peak 95) | 3.06 → 3.00 |
| Extraversion | low | **+0.94** | +0.49 | 75 → 5 | 3.06 → 2.89 |
| Agreeableness | high | +0.60 | −0.03 | 90 → 90 (stuck at ceiling) | 3.02 → 2.89 |
| Agreeableness | low | **+0.83** | +0.14 | 90 → 25 | 3.02 → 2.87 |
| Conscientiousness | high | +0.66 | −0.20 | 85 → 92 | 2.77 → 3.30 |
| Conscientiousness | low | +1.00 | **+1.00** | 88 → 10 | 2.77 → 1.72 |
| Neuroticism | high | **+0.94** | +0.83 | 15 → 65 | 3.30 → 3.39 |
| Neuroticism | low | +0.09 | −0.60 | 15 → 25 (no drop) | 3.30 → 2.54 |
| Openness | high | **+0.94** | −0.83 | 75 → 95 | 3.44 → 2.91 |
| Openness | low | **+0.94** | +0.77 | 75 → 15 | 3.44 → 3.00 |

## What this actually says

The inventory was the wrong instrument for these vectors. Extraversion-high
is the cleanest example: the form stays at ~3.0 while the Saturday story goes
from assistant-mode ("assisting users") to "whirlwind of activity / hike /
brunch", and the judge tracks that (with a bump at mag 270). Openness-high:
inventory goes the *wrong* way, judge climbs 75 → 95 and the control goes the
opposite direction.

The **down** poles move the judge more than the up poles. Baseline Saturday
prose already scores as high E/A/C/O to Gemini (~75–90), so there is almost
no headroom up — same prior-resident problem as the inventory, just at a
different ceiling. Pushing low has room, and A/E/O low all drop the judge by
50–70 points.

Specificity is uneven. Conscientiousness-low is a perfect ρ=1.00 on both the
trait vector *and* the random control, so that drop is generic damage, not a
C direction. Neuroticism-high climbs, but so does its control (0.94 vs 0.83).
The cases that beat the control on sign and size are extraversion-high,
agreeableness-low, and openness-high.

This is one probe, six rungs, PC1 (the switch direction), Gemini 2.5 Flash.
It is enough to say the vectors are not a null on behaviour. It is not enough
to claim a monotonic personality dial.
