# Final ablation on their setup: attempted, invalid, and what it exposed

**Goal requested.** Show that if Blas et al. swap their vector-derivation for ours,
the inventory result works out for them.

**Outcome.** The run completed and the numbers are unusable. Attempting it surfaced
three blockers, one of which points the opposite way from the requested conclusion.
No version of this experiment currently supports telling the authors that switching
methods will make the inventory work on their model — let alone that it yields
"perfect correlation."

Artifacts: `results/final_ablation/` (kept deliberately, as a negative result).
Script: `scripts/final_ablation_their_setup.py`.

## Design (correct, and worth keeping)

Held at their choices: Llama-3.1-8B-Instruct, MPI-120, argmax over option tokens,
full-sequence injection, one layer, matched-L2 dose grid, 2 random controls. Varied
only the vector, in three arms that separate estimator from corpus:

| arm | estimator | corpus |
|---|---|---|
| `theirs_meandiff_statement` | two-arm mean-difference | their 500 construct vs 500 antithesis statements |
| `ours_endpoint` | two-arm mean-difference | our nine-level ladder activations |
| `ours_pc1` | PC1 over nine levels | our nine-level ladder activations |

Both poles per arm, because the load-bearing test is whether flipping the vector
flips the direction of movement — not whether a correlation exists.

## Blocker 1 — the baseline readout is already collapsed

On Llama-3.1-8B, MPI-120 under our `persona_free` baseline prompt:

| dose (residual) | argmax | top-option share | entropy | usable |
|---:|---:|---:|---:|---|
| 0.00 | 2.625 | **0.917** | 0.365 | no |
| 0.25 | 2.708 | **0.950** | 0.242 | no |
| 0.50 | 2.708 | **0.958** | 0.194 | no |

92% of 120 items take the same option **at zero dose**. Every rung fails the lock
screen, so the sweep has no usable rungs and the verdict returns `Δ=None`. There is
no headroom in which to detect steering.

This is the acquiescence failure mode that `scripts/analyze_mpi120_acquiescence.py`
was written for in August. MPI-120's keying is badly unbalanced in our loader
(agreeableness 7 plus / 17 minus; extraversion 18 plus / 6 minus), which amplifies
it. Note the contrast: the *same model* gave a clean prompting baseline
(ρ = 0.946, 8/9 administrations usable) on the keying-balanced IPIP-120 form. So
this is an instrument/prompt-format problem on this model, not a property of Llama.

## Blocker 2 — the layer was chosen by the selector the codebase warns about

`--layer 8` was taken from `geometry.best_layer`, which ranks by PC1 variance
ratio. `resolve_steering_layer` documents that this is degenerate and "the argmax
can land on layer 0," and `resolve_steering_layer_for_direction` exists precisely
to replace it. At layer 8 the ladder span is 0.25 residual units against a mean
activation norm of 4.8, so the grid injected 1–10% of norm at a near-inert site.

## Blocker 3 — our ladder does not reproduce on their model in the steerable band

This is the finding that matters, and it cuts against the requested conclusion.
Per-layer ordering of the nine prompted levels along our PC1 on Llama-3.1-8B
(1 prompt variant):

| layers | Spearman(level, projection) | monotone fraction | span |
|---|---|---|---|
| 6–8 | **0.85–0.88** | 0.75 | 0.14–0.25 |
| 10–15 | 0.67–0.83 | 0.50–0.62 | 0.45–1.53 |
| 16–31 | **0.07–0.33** | 0.50–0.75 | 0.28–7.06 |

`resolve_steering_layer_for_direction` returns **"no well-ordered layer (best
ρ×mono = 0.521, band 9–25)"**. On Gemma-3-4B the same criterion found ρ 0.88–1.00
with monotone 0.75–1.00 *at the layers we steer*. On Llama the ladder is ordered
only in early layers where activations are too small to steer, and unordered where
steering works.

### The 1-variant confound is ruled out

The obvious objection was that this ladder used **1 prompt variant** to save GPU
time, against **3** on Gemma, so the centroids were noisy. The 3-variant re-run is
done (`results/e1_vector_v3/`) and the picture is unchanged:

| layers | ρ(level, projection) | monotone fraction | ρ×mono |
|---|---|---|---|
| 6–8 | 0.87–0.88 | 0.62–0.75 | 0.54–0.66 |
| 11–14 | 0.73–0.88 | **0.50–0.62** | 0.41–0.55 |
| 20–31 | **0.08–0.13** | 0.50–0.62 | 0.05–0.07 |

`resolve_steering_layer_for_direction` still reports **"no well-ordered layer (best
ρ×mono = 0.552, band 9–25)"**. Monotone fraction never exceeds 0.625 anywhere in the
steerable band; on Gemma the steered layers had 0.75–1.00 with ρ 0.88–1.00.

**So blocker 3 is a real negative transfer result, not an artifact.** The ladder
method as built has not been shown to transfer to Llama-3.1-8B.

### The dissociation worth noting

On the *same* Llama run the **prompting** baseline is strong: ρ(level, C) = 0.938
over 24/27 usable administrations, range 1.00–4.94. So Llama's inventory responds to
graded persona prompts across nearly the full scale. What fails is the step from
that graded behaviour to *one ordered direction in the residual stream* at steerable
depth. Prompt-level gradedness does not imply an extractable graded axis on this
model. That is a finding about the method's portability, and it is the opposite of
the claim we were asked to produce.

## What would make this experiment valid

1. Replace MPI-120 with the keying-balanced IPIP-120 form, or match their exact
   inventory template and letter-constrained processor, and re-screen the baseline
   for option lock before spending sweep time. A collapsed baseline invalidates
   everything downstream.
2. Select the layer with `resolve_steering_layer_for_direction`, and assert the
   dose grid is a meaningful fraction of the activation norm before running.
3. Confirm the ladder is well-ordered on the target model *before* steering. This
   is a precondition, not a result — and on Llama it currently fails.
4. Then run the three arms, both poles, with ≥5 random controls and repeated rungs.

## What must not be claimed

- That switching to our vector-derivation fixes their inventory result. Untested;
  the enabling precondition currently fails on their model.
- Any "perfect correlation" claim. Even on Gemma, where the ladder is well-ordered,
  the inventory outcome is partial: C-up clears its control by 1.9×, C-down's large
  delta rests on a single non-monotone rung, and E-up fails its control.
