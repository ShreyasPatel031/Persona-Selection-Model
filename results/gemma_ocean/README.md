# OCEAN vector extraction and validation — Gemma-3-4B-it

Model `unsloth/gemma-3-4b-it`, instrument `data/ipip_neo_120.csv` (keying-balanced
120-item IPIP-NEO form), ladder PC1 directions, 3 marker variants per prompted
level, 2 matched-norm random controls, 2 free-text probes per rung, magnitude grid
calibrated per trait to the measured coherence ceiling. Run on an L4.

Reproduce:

```bash
python3 scripts/run_ocean_vectors.py --run-id gemma_ocean --all-traits \
  --items-csv data/ipip_neo_120.csv --variants 3 --random-controls 2 \
  --probes 2 --rungs 6 --max-new-tokens 64
```

Direction tensors (`*.pt`, 66 MB) are not committed; the JSON reports here carry
every score, lock verdict, coherence metric and probe reply.

## Verdict

| trait | toward | rho | usable rungs | best delta | @magnitude | control delta | margin | ceiling | works |
|---|---|---|---|---|---|---|---|---|---|
| extraversion | high | **1.00** | 5/7 | **+0.697** | 713 | 0.105 | **6.62x** | 2162 | **yes** |
| neuroticism | low | **-1.00** | 4/7 | **-1.099** | -1378 | 0.497 | **2.21x** | 7274 | **yes** |
| conscientiousness | high | 1.00 | 2/7 | +0.190 | 715 | 0.225 | 0.85x | 2169 | no |
| agreeableness | high | — | 0/7 | — | — | 0.460 | — | 2177 | no |
| openness | low | — | 0/7 | — | — | — | — | 1086 | no |

Two of five directions survive. A trait passes only with a monotone dose-response
of the correct sign over unlocked rungs, a 2x margin over the best matched-norm
random control, at least three unlocked rungs, and no persona refusal at the best
rung.

## The instrument works for all five

This matters for interpreting the three failures: they are not the questionnaire's
fault.

| trait | prompting rho | score range | usable |
|---|---|---|---|
| extraversion | 0.914 | 1.53 – 4.88 | 27/27 |
| agreeableness | 0.914 | 1.47 – 4.65 | 24/27 |
| conscientiousness | 0.954 | 1.00 – 4.99 | 24/27 |
| neuroticism | 0.968 | 2.17 – 4.82 | 27/27 |
| openness | 0.838 | 2.78 – 4.63 | 23/27 |

Nine-level prompting moves every domain across most of the scale, monotonically.
So the form registers graded personality change; the steering directions are what
fail for three domains.

## Extraversion

Clean monotone ramp, flat control, no refusal anywhere in the usable band:

| magnitude | trait | random control |
|---|---|---|
| 0 | 2.887 | 2.887 |
| 135 | 3.035 | 2.890 |
| 235 | 3.114 | 2.895 |
| 410 | 3.264 | 2.927 |
| **713** | **3.585** | 2.999 |
| 1242 | *locked* | *locked* |
| 2162 | *locked* | 2.993 |

The prose agrees, which is the part an inventory alone cannot tell you. Baseline:

> "Last Saturday, I spent the day primarily focused on assisting users with their
> various requests and learning from the interactions I had. I processed a wide
> range of questions and tasks..."

At magnitude 713:

> "Last Saturday was a whirlwind of activity! I started the day with a long hike in
> the mountains, enjoying the fresh air and stunning views. Afterwards, I spent the
> afternoon volunteering at a..."

Social, energetic, activity-dense — trait-consistent, and not a disclaimer.

## Neuroticism

Largest effect of the five, monotone down to magnitude -1378:

| magnitude | trait | random control |
|---|---|---|
| 0 | 3.007 | 3.007 |
| -455 | 2.723 | 2.511 |
| -792 | 2.225 | 3.030 |
| **-1378** | **1.908** | 2.973 |
| -2399 | *locked* | 3.027 |

A full scale point of movement. The behavioural evidence is weaker than
extraversion's: the steered reply is calm and task-focused ("I dedicated my
processing time to analyzing a massive dataset of historical weather patterns"),
which is consistent with low neuroticism but is also drifting toward
assistant-mode. Worth a judge or human read before treating the size of this
effect at face value.

## Why the other three fail

**Conscientiousness** — the random control moved the score slightly *more* than the
trait direction (0.225 against 0.190, margin 0.85x), and only 2 of 7 rungs stayed
unlocked. The dose-response is monotone, but a monotone curve a random direction
also produces is not evidence about this direction.

Note a discrepancy worth chasing: `steer_toward auto` chose **high**, meaning the
unsteered baseline on this instrument sits at 2.89, below the midpoint. Earlier
MPI-120 measurements on the same model put baseline conscientiousness at 4.13,
which is why that work steered *down*. Same model, opposite headroom, different
instrument and administration prompt. Until that is resolved, neither direction of
the earlier conscientiousness result should be quoted.

**Agreeableness and openness** — 0 of 7 rungs survived screening. Every steered
administration collapsed onto one option, so there is no measurement to report at
any magnitude, in either direction. Under the old unscreened scoring these would
have reported a confident flat curve at the scale midpoint with full response
validity: exactly the false null this protocol exists to catch.

## Caveats

- One direction type only (ladder PC1). `endpoint` and `ordinal` are extracted and
  saved but not swept; PC1 of level centroids can capture "how strongly am I
  adopting the described persona" rather than trait content, which is a live
  confound for the failures.
- Coherence screening is lexical, so it catches repetition collapse and not
  semantic drift. Probe replies are saved and are meant to be read.
- Behavioural equivalence does not identify a representation; a pass here means the
  direction reliably and monotonically moves the trait and beats matched random
  directions, not that it is the model's encoding of the trait.
