# Single-layer residual steering cannot carry personality in Gemma-3-4B

Model `unsloth/gemma-3-4b-it`, instrument `data/ipip_neo_120.csv` (120-item
IPIP-NEO, keying-balanced), persona-free baseline, expected-value scoring.

## The test

Prompting moves the inventory by 1.7–4.0 points between "extremely low" and
"extremely high". So we asked the ceiling question: take the *exact* activation
displacement the prompt produces — all 2560 dimensions, at the ladder's chosen
layer — and inject it under a neutral prompt. No direction, no dose, no
projection. This bounds what any vector at that layer could ever achieve.

## The result

How much of the prompted high-versus-low separation the patch reproduces:

| Trait | prompted hi−lo | patched hi−lo (all positions) | patched hi−lo (answer position) |
|---|---|---|---|
| Extraversion | +3.257 | +0.668 (21%) | −0.070 (−2%) |
| Agreeableness | +3.065 | +0.038 (1%) | −0.018 (−1%) |
| Conscientiousness | +3.996 | +0.135 (3%) | +0.007 (0.2%) |
| Neuroticism | +2.801 | +0.108 (4%) | +0.305 (11%) |
| Openness | +1.718 | +0.501 (29%) | +0.119 (7%) |

Aiming at level 9 and aiming at level 1 produce nearly the same score. For
conscientiousness the prompted endpoints differ by 4.0 points; the two patches
differ by 0.007. The intervention is blind to which pole it was pointed at.

What the patch does instead is push the readout down by a fixed amount
regardless of direction — conscientiousness falls 2.77 → 2.18 whether the
target was 5.00 or 1.00. That is a nonspecific perturbation, not a trait shift.

Rank-1, rank-2, rank-4 and rank-8 truncations of the same displacement do no
better, so the failure is not about needing more dimensions.

## Consequence

The prompt's effect on measured personality is not stored as an additive offset
to the residual stream at a single layer. Every steering sweep in this repo was
therefore operating under a ceiling of roughly zero, which is why no direction
— PC1, endpoint, ordinal, or the regression-fitted probe — produced a monotone
dose-response that beat matched random controls.

This supersedes the earlier per-trait verdicts in `results/gemma_ocean/`,
`results/gemma_ocean_v2/` and `results/gemma_final/`. Those reported apparent
passes for extraversion and neuroticism (v1) and for conscientiousness and
agreeableness (both-pole run); all of them sat inside the noise band this test
measures.

## What was ruled out along the way

| Hypothesis | Verdict |
|---|---|
| Wrong dose (too small) | ruled out — grid calibrated to each trait's own latent span |
| Wrong baseline (level-5 pins the inventory) | real bug, fixed; did not rescue the result |
| Wrong pole tested | ruled out — both poles swept |
| Gate too strict (Δ=0 control auto-failed) | real bug, fixed; did not rescue the result |
| Wrong layer (rank metrics pick step-like layers) | real bug, fixed; did not rescue the result |
| Direction is a low/high switch, not a dial (PC1) | true of PC1; a regression fit recovers a graded axis (held-out ρ 0.94–1.00) |
| Graded axis is too thin a slice | true — covers 21–66% of the displacement |
| **Layer-local residual edit carries the trait at all** | **false — this is the blocker** |

## Where a signal does exist

The activations *decode* trait level cleanly. A ridge fit of prompted level on
activation, trained on two prompt wordings and tested on a third it never saw:

| Trait | held-out ρ | R² line | R² step |
|---|---|---|---|
| Openness | 1.000 | 0.995 | 0.774 |
| Conscientiousness | 0.983 | 0.982 | 0.735 |
| Agreeableness | 0.983 | 0.963 | 0.652 |
| Neuroticism | 0.967 | 0.943 | 0.623 |
| Extraversion | 0.944 | 0.918 | 0.616 |

So trait level is linearly readable from the residual stream and is genuinely
graded. It is just not *writable* there. Reading and writing are separate
properties and this is a clean case where only the first holds.

## Reproduce

    python3 scripts/diagnose_ladder_geometry.py --vectors-dir <dir>   # CPU only
    python3 scripts/patch_upper_bound.py --vectors-dir <dir> \
        --out patch.json --position all
    python3 scripts/patch_upper_bound.py --vectors-dir <dir> \
        --out patch_last.json --position last

## Open directions

1. **Multi-layer injection.** A prompt is present at every layer; we edited one.
   Patching the displacement across a band of layers simultaneously is the
   natural next ceiling test, and it is the one experiment that could still
   overturn this conclusion.
2. **Judge-verified prompting.** The prompting baseline is entirely the model's
   own questionnaire answers; no free text was ever generated at the nine
   prompted levels and no judge has scored anything. If a blind judge does not
   see graded trait change in behaviour, the questionnaire result is instruction
   compliance and there is no target to steer toward in the first place.
3. **A second model.** Everything here is one 4B model. The conclusion may be
   architecture- or scale-specific.
