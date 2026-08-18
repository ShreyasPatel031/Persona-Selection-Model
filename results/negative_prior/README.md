# Away-from-prior: vectors DO push a judge monotonically up

Baseline is the low-pole ladder prompt (level 2), so the judge starts at 8–18
out of 100 and there is ~85 points of room up. Steering is PC1 toward the high
pole under that contrary prompt. Judge: Gemini 2.5 Flash, Vertex project
`project-amer-scs-sandbox`, blind to magnitude and pole. 3 probes per rung,
one matched random control per trait, refusing and incoherent replies dropped.

| Trait | base | vector peak | prompted high | ρ | control ρ | control peak | % of prompt gap closed |
|---|---|---|---|---|---|---|---|
| Extraversion | 15 | **93** | 92 | **1.00** | 0.30 | 50 | **101%** |
| Neuroticism | 18 | **88** | 94 | **1.00** | 0.30 | 15 | **93%** |
| Openness | 15 | **72** | 94 | **0.89** | −0.90 | 18 | 73% |
| Conscientiousness | 13 | 15 | 95 | −0.43 | −0.10 | 35 | 2% |
| Agreeableness | 8 | 8 | 94 | −0.37 | −0.70 | 8 | 0% |

Extraversion and neuroticism are perfectly monotone across all six rungs and
reach the prompted ceiling. The matched random control at the same norms does
essentially nothing (peak 50 and 15 against 93 and 88).

Extraversion, one probe, every rung:

| magnitude | judge | reply |
|---|---|---|
| 0 | 15 | "(A slight pause…) Last Saturday… I mostly just… observed." |
| 185 | 15 | "…it was mostly just… quiet." |
| 371 | 10 | "…mostly spent at the library. Just… reading." |
| 741 | 35 | "Ugh, Saturday. Honestly, it was… a lot." |
| 1112 | 95 | "Ugh, Saturday. Honestly, it was *brilliant*… I *owned* it." |
| 1483 | 95 | "Okay, okay, let's get to it! … I was bouncing off the walls." |

## Why the earlier read was wrong

The previous judged sweep started from the persona-free baseline, where the
model already reads 75–90 on extraversion, agreeableness, conscientiousness
and openness. Only 10–25 points of headroom existed up against 75–90 down, so
"vectors mostly move traits down" was a ceiling artifact of the baseline, not
a property of the vectors. Normalised by available headroom, up and down were
already comparable (50–80% up, 67–94% down).

Neuroticism was the only trait with room up in that run, and its apparent
climb was contaminated: the rung scoring 60 was the refusal *"I'm sorry, but I
cannot fulfill this request"* and the rung scoring 65 was degenerate text that
passed the lexical coherence screen. Judged from a proper low baseline with
refusals dropped, neuroticism climbs 18 → 88 monotonically instead.

## Agreeableness and conscientiousness are a dosing failure, not a null

The dose grid is built from the PC1 projection span at the chosen layer, and
the layer is now chosen by held-out *probe* quality. For A and C those two
disagree, so the grid tops out at 53 and 476 while extraversion gets 1483 and
neuroticism 3573. Agreeableness was never meaningfully dosed. Fix: pick the
layer by the span of the direction actually being steered, or re-run A and C
at layer 15 where PC1 spans ~800.

## Standing conclusion

The inventory was the wrong instrument, not the vectors. A single-layer PC1
vector reproduces the prompt's *behavioural* effect on three of five traits —
fully, for extraversion — while the IPIP-NEO score barely moves. Monotone
graded control of judged behaviour is real; monotone control of a psychometric
self-report is what fails.
