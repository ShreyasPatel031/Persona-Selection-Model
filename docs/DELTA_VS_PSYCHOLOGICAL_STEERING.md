# What we changed relative to Blas et al. 2026

Our pipeline started from **Blas, Jia & Ferrara, "Psychological Steering of Large
Language Models"** (arXiv:2604.14463). That paper is the closest prior work: same
intervention family (additive residual injection), same instrument family
(MPI-120 ≡ IPIP-NEO-120), calibrated dose units, no LLM judge on the inventory,
and **the same model** — their `G4` is `gemma-3-4b-it`, which is our default in
`app/persona/activations.py`. So the comparison is direct rather than analogical.

They reported two negative results that we do not reproduce:

1. **Nothing on the inventory.** §5.2: *"Importantly, none of these observations
   applied to the inventory responses; we observed no salient patterns beyond
   occasional co-occurring peaks in µ_sum MPI-120 scores."* They therefore
   *"narrowed our analyses to SJT responses"* — synthetic situational-judgment
   text scored first by a style classifier and then by GPT-5.1.
2. **No Big Two structure.** §5.3: only 46.15% of cross-trait sign patterns match
   Digman α/β; *"No LLM satisfied all Big Two correlations."* Crucially this
   covariance is computed **on the SJT scores, not on the inventory**, because by
   then the inventory had been dropped.

Their leakage is large: λ = 0.4–0.7, i.e. non-target traits move 40–70% as much
as the target per unit of dose.

---

## The seven differences

Ordered by how much evidence we have that the change is what mattered.

### 1. Readout: expected value over option tokens, not argmax

**Theirs.** Greedy decoding, *"inventory completions constrained to allow only
valid responses ("A", "B", "C", "D", or "E") and limited to 1 new token"*. That is
an argmax over the option tokens. Serapio-García et al. used the same family of
constrained readout and got clean prompting curves, so this is a reasonable
choice — but an argmax is a step function, and it cannot register movement in the
option distribution that has not yet crossed a decision boundary.

**Ours.** `score_traits_ev` takes the expected value over the option-token
softmax. `target_argmax` is recorded alongside `target_ev` at every rung, so the
two readouts can be compared on *identical activations*.

**Evidence.** `scripts/readout_argmax_vs_ev.py` recomputes every sweep we have
under both readouts (`results/readout_argmax_vs_ev.json`):

| sweep | ρ argmax | ρ EV | distinct argmax values | rungs |
|---|---:|---:|---:|---:|
| N up (in-span) | **+0.37** | **+0.98** | **2** | 9 |
| A down (in-span) | −0.61 | −0.85 | 6 | 9 |
| E up (in-span) | +0.67 | +0.75 | 7 | 9 |
| A up | +0.67 | +0.90 | 4 | 5 |
| A down | −0.32 | −0.60 | 5 | 6 |
| C down | −0.72 | −0.90 | 4 | 5 |
| O up | +0.87 | +1.00 | 2 | 3 |
| C up | +1.00 | +1.00 | 4 | 4 |
| N down | −0.90 | −0.90 | 5 | 5 |
| O down | −1.00 | −1.00 | 4 | 4 |
| E down | −0.50 | −0.50 | 3 | 3 |
| E up (ceiling-dosed) | −0.05 | −0.40 | 3 | 5 |
| N up (ceiling-dosed) | −0.40 | −0.20 | 4 | 4 |

Expected value gives a larger correctly-signed |ρ| in 7 of 13 sweeps, ties in 4,
and the two remaining rows are the ceiling-dosed wrong-sign cases where no readout
rescues a bad dose grid. The readout never hurts, and where the effect is small it
is the difference between seeing it and not.

N-up is the clean case. Under argmax the 120-item score takes **two distinct
values across nine doses** and correlates at +0.37; under expected value the same
forward passes give +0.98. A two-valued staircase is precisely what *"no salient
patterns"* looks like. The effect here is genuinely small in absolute terms
(0.08 EV points), which is why readout resolution decides whether it is visible
at all.

### 2. Dose unit: the ladder's own span, not the fluency ceiling

**Theirs.** α is calibrated — in "centroid units", the distance from the
between-centroid midpoint to a centroid of the *statement corpus*. Sweeps are
*unbounded*, in integer steps, early-stopped on fluency decay, and α\* is picked
as the coefficient of maximum steering effect. So the grid ends where the model
starts breaking.

**Ours.** `direction_span_magnitude` projects (level-9 centroid − level-1
centroid) onto the direction actually being injected, at the layer actually being
used. That is the residual-stream size of a full-scale personality change,
measured on the ladder. The grid is laid out in fractions of that span
(`span_multiples_grid`); the coherence ceiling from `calibrate_magnitude_ceiling`
is only a soft upper bound, never the dose scale.

**Evidence.** We made their mistake first. The `gemma_final` sweeps ran to
coherence ceilings 3–4× past each trait's span and produced two wrong-sign
correlations. Re-dosing inside the span (`results/e1_inspan/`) fixed both:

| pole | old ρ (ceiling-dosed) | new ρ (span-dosed) | usable rungs |
|---|---:|---:|---:|
| N up | −0.20 (wrong sign) | **+0.98** | 4 → 9 |
| E up | −0.40 (wrong sign) | **+0.75** | 5 → 9 |
| A down | +0.60 | **+0.85** (sign-corrected) | 6 → 9 |

Note what this is *not*: `results/gemma_final/inspan_reanalysis.json` merely
deletes the out-of-span rungs from the old grid, and that leaves N-up at −0.20.
Truncation alone does not fix it. What fixes it is **sampling densely inside the
span** — eight doses between 0.05× and 0.55× span, giving 9 usable rungs instead
of 4. Their integer-step-to-the-ceiling grid puts almost no rungs in the region
where the trait actually moves.

### 3. Vector source: a graded prompt ladder read at the answer position

This is the dataset change.

**Theirs.** 35,000 generated texts per condition from Llama-3.1-8B, deduplicated
to 500 first-person statements expressing the construct and 500 its antithesis.
Activations are taken from two prefill templates (`"Answer with Yes or No…"` and
`"Tell me about yourself." → statement`). Vectors are a **two-arm** contrast:
mean difference of the two centroids (MDS/MDB), or a logistic probe normal.

**Ours.** Nine-level prompt ladder (Goldberg markers × Likert qualifiers, the
Serapio-García shaping method), and the activations are collected **from the
forward passes that answer the inventory items**, at the answer position, with
`context_mode: "inventory"`. The direction is **PC1 across the nine level
centroids**, not a difference of two.

Two consequences:

- A two-arm contrast yields a *ray*. It has a direction but no notion of
  intensity ordering, so there is nothing to check and nothing to calibrate
  against. Nine rungs give an *axis* whose ordering is measurable:
  `spearman_level_vs_pc1_projection`, `monotone_fraction_pc1_projection`,
  `consecutive_step_cosine_mean`. Both the layer choice and the dose unit are
  derived from that ordering, so neither is available without the ladder.
- Their activations come from a "tell me about yourself" statement distribution
  and are then injected while the model answers Likert items. Ours are extracted
  at exactly the position they will be injected for. Their own preliminary
  analysis hints at this mismatch: probes on the statement activations `h^s`
  reached perfect test accuracy in **0.60%** of cases versus every case for the
  yes/no activations `h^b`, which is why they restricted probe vectors to `h^b` —
  yet MDS (built from `h^s`) is the vector they carried forward.

### 4. Layer choice: best-ordered ladder, not strongest effect

**Theirs.** Sweep all layers, keep the layer with the largest steering extremum
on the SJT score. Peaks land mid-network.

**Ours.** `resolve_steering_layer_for_direction` ranks layers by how *well
ordered* the ladder is along the injected direction (Spearman × monotone
fraction), tie-breaking on span.

**Evidence.** Picking by effect size or span actively misleads. Agreeableness
span-first selects layer 20 (ρ 0.83, monotone 0.62, span 2197), where steering up
moved a judge *less than a random vector did*. Ordering-first selects layer 15
(ρ 1.00, monotone 1.00, span 819), which closes 103% of the prompt's gap in both
directions. The commit that changed this is `f66fd0a`.

### 5. Lock screening: a collapsed readout is missing data, not a null

**Theirs.** A sweep step counts as valid if the SJT text is fluent, the score
moved in the intended direction, and responses did not repeat verbatim for three
consecutive steps. Degenerate *inventory* readouts survive all three checks.

**Ours.** `option_lock` screens the raw option histogram at every administration
and locked rungs are excluded from the correlation.

**Why this specifically hides an inventory effect.** With keying-balanced items,
an acquiescence-corrected score is `(plus_mean + minus_mean)/2`. If the model
answers option *v* to everything, plus-keyed items score *v* and minus-keyed
score *6 − v*, so the trait averages to **exactly the scale midpoint whichever
option was locked onto** — while `response_validity` reports 1.0. "The model
stopped discriminating between items" and "steering did nothing" produce the same
number, and the first dominates at high dose. Since their grid runs to the
fluency ceiling, their inventory scores are averaging in exactly these rungs.

The same trap bites the baseline: on Gemma-3-4B the level-5 "neutral" ladder
prompt pins 97–100% of O/A/C items to option 3, so a sweep anchored there starts
from a degenerate readout. We use `persona_free_system_prompt` instead.

### 6. Instrument: purpose-built keying balance

**Theirs.** MPI-120, items manually rephrased in the second person.

**Ours.** `data/ipip_neo_120.csv`, built by `scripts/fetch_ipip_items.py` from the
public IPIP-NEO facet pool with **2 plus- and 2 minus-keyed items per facet**.
Balance is what makes acquiescence correction well defined and makes a lock show
up as an unambiguous midpoint pin rather than a plausible-looking shift.

### 7. Controls: matched-norm random directions at identical doses

**Theirs.** Methods are compared against each other and against P². No random
direction at matched norm.

**Ours.** `random_control_directions` at the same grid, and `verdict.works`
requires `control_margin_ratio ≥ 2.0`. N-up in-span clears it by 12.9×, A-down by
3.0×. Without this, a large perturbation moving the score proves only that a
large perturbation moves the score.

---

## What this predicts about their Big Two null

Their covariance analysis sweeps α ∈ {0, α\*/9, …, α\*} where α\* is the
**maximum-effect** coefficient found by an unbounded, fluency-limited sweep. On
our runs that region is the degradation regime, where every direction — including
matched-norm random ones — drives the same generic drift (C up, N down, O down
hard). Cross-trait correlations measured there are dominated by that shared
attractor, which has no reason to respect α/β structure. λ = 0.4–0.7 is
consistent with being in it.

Restricting to in-span doses, the shared drift largely disappears on our runs and
the residual covariance becomes trait-specific. So the concrete, falsifiable
claim is: **their Big Two null is a dosing artifact, and recomputing the same
correlations over 0 → 1× ladder span rather than 0 → α\* should raise the α/β
match rate above 46%.** Their repository is public
(`github.com/leonardo-blas/psychological-steering`), so this is checkable without
re-deriving anything.

---

## Honest state of our evidence

- 4 of 10 pole-directions have been re-dosed in-span so far (N-up, E-up, E-down,
  A-down). The other six carry `gemma_final` ceiling-dosed numbers.
- **E-down still fails**: only 2 usable rungs in the window, the rest option-lock.
  It needs a denser low-dose grid.
- The opposite-prior IPIP sweep, which is what gives the effects real headroom
  instead of 0.08–0.33 EV points, is incomplete.
- Single model, single seed. Their 14-model sweep is broader than ours, and the
  argmax/EV contrast above is currently demonstrated on one model.
- We make no factor-structure claim, so Dorner et al.'s CFA objection and their
  "one model wearing personas is not a population" objection are sidestepped
  rather than answered.
