# Bipolar steering: one vector, up and down, judged blind

`unsloth/gemma-3-4b-it`, PC1 ladder direction, single layer, additive residual
injection. Each pole is tested from the prior it has room to move away from:

* **up** — baseline is the low-pole prompt (level 2), steer `+v`, reference is
  the level-9 prompt
* **down** — baseline is the high-pole prompt (level 8), steer `−v`, reference
  is the level-1 prompt

Judge: Gemini 2.5 Flash on Vertex project `project-amer-scs-sandbox`, blind to
magnitude, pole and kind. 3 probes per rung, one matched-norm random control per
trait, refusing and incoherent replies dropped before any correlation.

A pole passes only with the correct dose-response sign (|ρ| ≥ 0.7), a ≥2×
margin over the random control, and ≥40% of the prompt's own gap closed.

| Trait | pole | layer | baseline → extreme | prompt reference | % of gap | ρ | control × | pass |
|---|---|---|---|---|---|---|---|---|
| Extraversion | up | 15 | 15 → **95** | 93 | **102%** | +0.94 | 2.3 | yes |
| Extraversion | down | 15 | 92 → **10** | 12 | **102%** | −1.00 | 81.7 | yes |
| Conscientiousness | up | 17 | 15 → **93** | 95 | **98%** | +1.00 | 46.9 | yes |
| Conscientiousness | down | 17 | 92 → **20** | 13 | **92%** | −0.77 | 42.9 | yes |
| Neuroticism | up | 20 | 18 → **87** | 95 | **89%** | +0.94 | 13.7 | yes |
| Neuroticism | down | 20 | 90 → 45 | 14 | 59% | −0.89 | 27.0 | yes |
| Openness | up | 19 | 15 → 71 | 94 | 71% | +0.89 | 16.8 | yes |
| Openness | down | 19 | 90 → 32 | 20 | 83% | −0.89 | 8.8 | yes |
| Agreeableness | down | 20 | 89 → 52 | 6 | 45% | −0.77 | 11.2 | yes |
| Agreeableness | up | 20 | 8 → 23 | 91 | 18% | +0.83 | **0.36** | **no** |

**Nine of ten poles pass. Four of five traits are fully bipolar.**

Extraversion reaches 102% of the prompt's gap in both directions — the vector
does more to judged behaviour than the level-9/level-1 prompt does. Trajectories
(|magnitude|: judge score):

    extraversion  up     0:15  185:22  371:39   741:66  1112:95  1483:93
    extraversion  down   0:92  185:88  371:86   741:27  1112:12  1483:10
    conscient.    up     0:15  247:22  494:22   988:48  1482:88  1976:93
    conscient.    down   0:92  247:93  494:93   988:91  1482:53  1976:20
    neuroticism   up     0:18  447:18  893:33  1787:23  2680:47  3573:87
    openness      down   0:90  382:92  764:84  1529:66  2293:76  3057:32

## What the dosing fix changed

Conscientiousness previously moved 13 → 15 (2% of the gap, ρ = −0.43) and read
as a flat null. Its grid had been built from the PC1 span at a layer chosen on
other grounds, topping out at 476. Dosing from the span of the direction being
steered at the layer being used gives a grid to 1976, and the same vector now
closes 98% of the gap with ρ = 1.00 and a 47× control margin. The vector was
never the problem; the dose was.

## The remaining failure

Agreeableness-up moves 8 → 23 and its random control moves further (margin
0.36). At layer 20 the agreeableness ladder is only weakly ordered (ρ = 0.83,
monotone fraction 0.62); layer 15 has ρ = 1.00, monotone fraction 1.00 with a
span of 819. Retesting that pole at layer 15 is the obvious next step.

Down-pole non-monotonicity at the top rung (neuroticism 45 → 72, openness
76 → 32 → and back) is where output quality starts to go; those rungs sit past
the coherence ceiling for the trait.

## Reproduce

    python3 scripts/bipolar_judge.py generate --vectors-dir DIR \
        --out generations.json --poles up,down --layer-select proven
    python3 scripts/bipolar_judge.py judge --generations generations.json \
        --out summary.json --project project-amer-scs-sandbox
