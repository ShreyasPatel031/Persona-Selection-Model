# Intensity-ladder CAA — reproducing graded inventory shifts with a vector

**Question.** Prompting (Serapio-García et al., *Nat. Mach. Intell.* 2025) and training
(BIG5-CHAT, ACL 2025) move real Big Five inventories; mean-difference activation steering
has not been shown to do the same. Can a direction derived from the *nine-level prompt
ladder* reproduce the graded IPIP movement that the prompts themselves produce?

**Why this is not trivially yes.** Nine prompt levels are nine different instructions that
stay in context while every item is answered. CAA replaces them with one direction and a
scalar: `h ← h + α·v̂`. Collapsing the ladder into a mean difference keeps only the
component that is shared across levels, and it removes the per-item instruction
conditioning. So the experiment has to measure two separate things — the *geometry* of the
ladder, and whether steering along it reproduces the *scores*.

## What the reference papers actually did

| Paper | Intervention | Inventory | Scoring protocol |
|---|---|---|---|
| Serapio-García et al. 2025 | 9 levels × 104 Goldberg adjectives + Likert qualifiers | IPIP-NEO (300), BFI (44) | one item per prompt, answer = argmax over Likert option tokens (log-prob ranking / constrained decoding) |
| Jiang et al. 2023 (MPI / P²) | Personality Prompting | MPI (IPIP-derived) | same constrained-choice scoring |
| BIG5-CHAT 2025 | SFT / DPO, high vs low | BFI, IPIP-NEO-120 | 5 repeats, temperature 0.6 |
| arXiv:2604.14463 | mean-difference residual injection, calibrated α | MPI-120 / IPIP-NEO-120 **and** synthetic SJTs | clean near-linear α curves appear on the SJTs, not the inventory |
| arXiv:2512.17639 | directions regressed on continuous IPIP scores | IPIP-50 items, forced choice | monotone on questionnaire items; open-ended weak, and a persona prompt overrides steering |

Three details are load-bearing and are reproduced here: **one item per administration**,
**answers restricted to the option tokens**, and **reverse-keyed scoring** — plus
measuring the steered model with the *same* instrument as the prompted model.

## Pipeline

```
prompt-ladder ──► centroids_<trait>.pt ──► vectors ──► ladder_vectors_<trait>.pt ──► alpha-sweep
   (baseline ρ)      (level activations)     (geometry)      (3 candidate v̂)          (steered ρ)
```

### 1. `prompt-ladder` — the baseline to beat

Administers the 50-item IPIP Big-Five markers (public domain) under nine prompted levels
of one trait, `--variants` marker rotations per level. Reports `spearman_level_vs_target_score`
(Serapio's ρ ≥ .80 is the target), `monotone_fraction_level_means`, and
`off_target_spearman` for the four unprompted traits — their Fig. 4 stability check.

Each forward pass yields both the Likert answer and the answer-position activation, so the
level-conditioned activations come for free. `--probe-contexts` additionally collects
centroids from open-ended prompts, which tests whether the ladder direction is specific to
the inventory or is a general trait direction.

```bash
PYTHONPATH=. python -m app.persona.run intensity-ladder -- \
  prompt-ladder --run-id ext_ladder --trait extraversion --variants 3 --probe-contexts
```

### 2. `vectors` — what direction is the ladder pointing in

Per layer, from the nine level centroids:

- `consecutive_step_cosine_mean/min` — do successive rungs move the same way, or does the
  ladder turn corners? Low values mean no single α can traverse it.
- `pc1_variance_ratio` — how much of the level-to-level variance is one dimension.
- `cos_endpoint_pc1` — does the classic CAA contrast (`h₉ − h₁`) agree with the ladder's
  principal axis? If it does not, plain CAA is steering off-ladder.
- `spearman_level_vs_pc1_projection`, `monotone_fraction_pc1_projection` — are the rungs
  *ordered* along the direction, not merely spread along it.
- `step_norm_cv` — spacing regularity; uneven spacing means α is not a linear dial even
  when the direction is right.

Three candidate directions are saved per layer: `v_endpoint` (mean-difference / CAA),
`v_pc1` (ladder principal axis), `v_ordinal` (minimum-norm least-squares fit of level onto
activations, the activation-space analogue of regressing on trait scores). `best_layer`
maximises monotonicity × PC1 share × |ρ|.

```bash
PYTHONPATH=. python -m app.persona.run intensity-ladder -- \
  vectors --run-id ext_ladder --trait extraversion
```

### 3. `alpha-sweep` — same instrument, vector instead of prompt

Re-administers the inventory under the **neutral level-5 prompt** while injecting `α·v̂` at
all positions of the chosen layer, sweeping α. α is in calibrated units by default
(`--alpha-units relative` scales by the mean activation norm at that layer, per
arXiv:2604.14463) so values transfer across layers and models.

Reports `spearman_alpha_vs_target_score`, `monotone_fraction_target_score`,
`target_score_range`, `response_validity` per α (steering that breaks option-token
compliance is a failure, not a null result), and `mean_off_target_abs_delta` for
cross-trait leakage.

```bash
PYTHONPATH=. python -m app.persona.run intensity-ladder -- \
  alpha-sweep --run-id ext_ladder --trait extraversion --direction pc1 \
  --alphas 0,0.25,0.5,0.75,1,1.25,1.5,2
```

Run it for all three directions; `--direction endpoint` is the control that says whether
the ladder geometry bought anything over the two-arm contrast you already have.

## Reading the outcome

The two measurements are independent, which is what makes the result informative either way:

| Ladder geometry | α sweep on IPIP | Interpretation |
|---|---|---|
| 1D and ordered | monotone, ρ high | Graded CAA works; extend to all five traits, multiple seeds and models, then add held-out behaviour |
| 1D and ordered | flat or non-monotone | Direction is right but a constant residual offset cannot supply per-item instruction conditioning — the prompting result does not transfer by construction |
| not 1D, or unordered | — | "Intensity" is not one direction; a single α cannot traverse the ladder, and a mean-difference vector is the wrong parameterisation |
| any | scores move but validity drops | Steering is degrading option-token compliance rather than shifting the trait |

The third row is the outcome the surrounding literature makes most likely: in
arXiv:2604.14463 mean-difference vectors are strongest on open-ended SJTs and markedly
weaker on the inventory, and in arXiv:2512.17639 questionnaire-fitted directions still lose
to prompt context. The value of running it is that geometry and scores are separated, so the
failure (if it fails) is diagnosed rather than guessed.

## Cost

Forwards are single-token prefills, no generation. `prompt-ladder` with 9 levels ×
3 variants × 50 items ≈ 1,350 forwards; each α costs 50. Sweeping three directions across
8 α values ≈ 1,200 more. Feasible on one T4 for Gemma-3-4B-it.

## Caveats before scaling

- Instrument reuse: the prompts use Goldberg markers and the inventory uses IPIP items, so
  induction and measurement are not the identical text — but they are lexically close, the
  validity concern BIG5-CHAT raises about prompt-induced personality.
- `v_ordinal` is fitted with a min-norm solution on few points; treat it as exploratory
  unless refit on more administrations (raise `--variants`).
- Single-model, single-seed results say little. The interesting claim needs all five traits,
  multiple seeds, at least two model sizes, and then held-out behaviour that the inventory
  score predicts across α.
