# What actually differs from Blas et al. 2026

Our pipeline started from **Blas, Jia & Ferrara, "Psychological Steering of Large
Language Models"** (arXiv:2604.14463). It is the closest prior work: same
intervention family (additive residual injection), same instrument family
(MPI-120 ≡ IPIP-NEO-120), calibrated dose units, no LLM judge on the inventory,
and **the same model** — their `G4` is `gemma-3-4b-it`, our default in
`app/persona/activations.py`. The comparison is direct rather than analogical.

They report two negative results we do not reproduce:

1. **Nothing on the inventory.** §5.2: *"none of these observations applied to the
   inventory responses; we observed no salient patterns beyond occasional
   co-occurring peaks in µ_sum MPI-120 scores."* They *"narrowed our analyses to
   SJT responses"* — synthetic situational-judgment text scored by a style
   classifier and then GPT-5.1.
2. **No Big Two structure.** §5.3: 46.15% of cross-trait sign patterns match
   Digman α/β; *"No LLM satisfied all Big Two correlations."* Computed **on the SJT
   scores, not the inventory**, because the inventory had already been dropped.

This document has been revised twice as the evidence was checked. Two candidate
differences did not survive; they are recorded below rather than deleted, because
knowing what *isn't* the difference is what makes the remaining list credible.

---

## Not the difference: the vector construction

This was previously listed as the biggest difference — "the dataset change". It is
essentially cosmetic, and `scripts/endpoint_vs_pc1_geometry.py` shows why.

They build a **two-arm mean-difference** vector: centroid of 500 construct
statements minus centroid of 500 antithesis statements. We build **PC1 across nine
graded prompt-ladder centroids**. At the layers we actually steer, those two
directions are nearly the same vector:

| trait | layer | cos(endpoint, PC1) | ρ level~PC1 | ρ level~endpoint |
|---|---:|---:|---:|---:|
| extraversion | 15 | 0.993 | 0.983 | 0.983 |
| agreeableness | 15 | 0.993 | 1.000 | 1.000 |
| conscientiousness | 15 | 0.996 | 0.883 | 0.883 |
| neuroticism | 20 | 0.994 | 0.967 | 0.967 |
| openness | 15 | 0.988 | 1.000 | 1.000 |

Mean cosine **0.993** — about 7° apart in 2560 dimensions. And the ladder is
ordered *exactly* as well along the two-arm contrast as along PC1 (identical
Spearman to three decimals). Nine graded levels do not buy a different direction
than two opposed piles.

Worse: **we never ran the head-to-head.** Every sweep in `results/` uses
`direction: "pc1"`. The codebase supports `--direction endpoint` precisely as this
control and `docs/INTENSITY_LADDER_CAA.md` describes it as such, but it was never
executed. So the claim "our vector is better" has no experimental support and, on
the geometry, probably has no room to be true.

What the ladder *does* buy is not the direction but two **measurements** that a
two-arm contrast cannot produce, because you need at least three points to check
an ordering: the layer-selection criterion and the dose unit. Those are real and
are differences 2 and 3 below.

## Not the primary difference: the readout

Also previously overstated. They score an item by greedy-decoding one constrained
option token — an argmax, which is what a human respondent does. We report expected
value over the option-token distribution.

Recomputing every sweep both ways (`scripts/readout_argmax_vs_ev.py`): **the sign
tally is identical, 11/13 either way.** Expected value is more sensitive — it gives
a larger correctly-signed |ρ| in 7 of 13 and ties in 4 — but it does not change the
pattern, so it cannot explain their null.

Where it matters is N-up alone: ρ +0.37 argmax vs +0.98 EV, on a score movement of
0.042 argmax points across 24 items, i.e. **about one item changing its answer**.
And N-up fails the random-control comparison under both readouts. So this is a
sensitivity difference on our weakest pole, not a load-bearing one.

---

## The differences that survive

### 1. Injection scope: they perturb ~3 positions on the inventory, we perturb ~65

The strongest difference, readout-independent, and **verified in their code** rather
than inferred from the paper. Repo: `github.com/leonardo-blas/psychological-steering`.

**Their injection span** (`replication/injection_utils.py`, `inject()`). On the
prefill pass they inject from the start of the *assistant turn* to the end of the
sequence:

```python
for b_local in range(B_local):
    s = starts[b_local]              # index where the assistant turn begins
    for t in range(s, T_local):
        if (t - s) % stride_val == 0:
            hidden[b_local, t] = hidden[b_local, t] + v_local
```

and on each decode step (`T_local == 1`) they inject that one new position
(lines 283–290). `starts` is the token length of the chat template *without*
`add_generation_prompt` (lines 200–209), so **the system prompt and the item text
are never injected** — only the generation prompt, any assistant prefix, and
generated tokens.

**The two tasks then get wildly different exposure**
(`replication/psychometric_utils.py`):

| | assistant_prefix | max_new_tokens | injected positions |
|---|---|---:|---:|
| `run_inventory` (lines 225–226) | `""` | **1** | **~3–4** |
| `run_sjts` (lines 259, 284) | `"I would"` | **64** | **~69** |

For Gemma-3 the generation prompt `<start_of_turn>model\n` is 3 tokens, and the
inventory adds no prefix and one answer token.

**Ours.** `_Steering.hook` does `h.add_(delta)` on the full `(batch, seq, d)`
tensor — every position, system prompt and item text included. Measured on our own
120-item form, one administration is **55–60 content tokens** plus template, so we
perturb on the order of **65 positions** where they perturb 3–4.

**Why this explains their result, in their own words.** Their sweep found
`s = 1` > `s = 2` > `s ∈ {3,4}` and they conclude *"injecting into more completion
activations yields stronger steering."* Applied to their own two tasks that predicts
the inventory is the weakest possible case. And note the coincidence: their **SJT**
exposure (~69 positions) is about the same as our **inventory** exposure (~65). The
task where they got clean linear dose-response is the task with roughly our
injection coverage; the task where they got "no salient patterns" is the one with
~17× less. Their inventory null looks like a dose-exposure artifact of answer length,
not a fact about inventories.

**Also confirmed while reading their code:**

- The inventory readout is argmax as described: `do_sample=False` plus a logits
  processor restricting to valid option ids, one token (`psychometric_utils.py`
  lines 211, 231–232).
- The cross-trait grid is `alpha * i / 9` for `i` in 0..9, i.e. 10 points from 0 to
  α\* where α\* is picked by `pick_extrema` as the maximum-effect coefficient
  (`11_cross_trait_sweeps.py` lines 31–37, 39–51). This confirms the Big Two
  covariance is measured over 0 → max-effect dose.
- They *do* compute inventory extrema in the cross-trait sweep and create an
  `inventory_responses.db` alongside `sjts_responses.db` (lines 60–61, 183–184), so
  the inventory cross-trait numbers were generated even though the paper reports
  covariance on the SJTs.
- **What is released:** code, `data/{inventories,sjts,heads}.db`, classifiers, and
  vectors for **Llama-3.1-8B-Instruct only**. No sweep result databases and no
  Gemma vectors, so checking their Gemma inventory numbers means re-running the
  pipeline, not just reading their artifacts.

### 2. Dose grid: fractions of the ladder span, not up to the fluency ceiling

**Theirs.** α swept unbounded in integer steps, early-stopped on fluency decay,
with α\* taken as the maximum-effect coefficient. The grid ends where the model
starts breaking.

**Ours.** `direction_span_magnitude` projects (level-9 − level-1 centroid) onto the
injected direction at the injected layer; the grid is fractions of that
(`span_multiples_grid`). The coherence ceiling is only a soft upper bound.

**Evidence, under argmax so it does not depend on our readout.** Three poles have
both a ceiling-dosed and an in-span run:

| pole | grid | ρ argmax | trait span | ctrl span | ratio | |
|---|---|---:|---:|---:|---:|---|
| E-up | ceiling | −0.05 | 0.167 | 0.125 | 1.33 | nothing |
| E-up | in-span | **+0.67** | 0.625 | 0.250 | **2.50** | supported |
| N-up | ceiling | −0.40 | 0.208 | 0.375 | 0.56 | nothing |
| N-up | in-span | +0.37 | 0.042 | 0.167 | 0.25 | still nothing |
| A-down | ceiling | −0.32 | 0.250 | 0.208 | 1.20 | nothing |
| A-down | in-span | −0.61 | 0.333 | 0.292 | 1.14 | still nothing |

One clean conversion (E-up: no effect → supported effect), one sign flip without
power (N-up), one no-change (A-down). Suggestive, not settled — three poles, one
decisive.

Note also that **truncating** a ceiling-dosed grid is not enough:
`results/gemma_final/inspan_reanalysis.json` just drops out-of-span rungs and
leaves N-up at −0.20. What changed E-up was *sampling densely inside* the span,
9 usable rungs instead of 5.

### 3. Layer selection by ladder ordering, not by effect size

**Theirs.** Sweep all layers, keep the one with the largest steering extremum.

**Ours.** `resolve_steering_layer_for_direction` ranks layers by how well *ordered*
the ladder is along the injected direction (Spearman × monotone fraction),
tie-breaking on span. This genuinely requires ≥3 ladder levels, so it is the one
place the nine-level design pays off.

**Evidence.** Agreeableness picked by span selects layer 20 (ρ 0.83, monotone 0.62,
span 2197), where A-up moved the judge *less than a matched random vector*
(control margin **0.36**). Picked by ordering it selects layer 15 (ρ 1.00, monotone
1.00, span 819): margin **7.2**, closing 103% of the prompt gap in both directions.
Commit `f66fd0a`. This is judge-side evidence, not inventory-side.

### 4. A collapse screen on the inventory response distribution

**Theirs.** A sweep step is valid if the SJT text is fluent, the score moved in the
intended direction, and responses did not repeat verbatim three steps running.
A degenerate *inventory* readout passes all three.

**Ours.** `option_lock` screens the raw option histogram; locked rungs are treated
as missing data. This matters specifically because with balanced keying, a model
answering one option to everything scores **exactly the scale midpoint** while
`response_validity` reports 1.0 — "stopped discriminating" and "steering did
nothing" produce the same number.

**Evidence.** Tightening the screen to `top option < 75% and ≥ 4 of 5 options used`
turns 11/13 sign-correct into **9/9**: both wrong signs were degraded readouts, not
inverted traits. The cost is that 5 of 14 sweeps stop being measurable.

### 5. Instrument keying balance (untested)

Theirs: MPI-120, items manually rephrased second-person. Ours:
`data/ipip_neo_120.csv`, 2 plus- and 2 minus-keyed per facet, which is what makes
acquiescence correction well defined and midpoint pinning diagnosable. No
experiment isolates this.

### 6. Baseline prior and headroom (untested against theirs, but it bites us too)

They steer from a no-persona baseline and note the ceiling problem themselves. We
do the same on the inventory sweeps — and it is why **N-up and O-up have no argmax
effect at all**: Gemma's baseline already sits at 3.30 and 3.44 of 5, so there is
nothing to push up into. Our judge results avoid this by starting from an *opposing*
prompt prior. The equivalent inventory run is incomplete, and it is the single
experiment most likely to change the result.

---

## Where this leaves it

Honest ranking of the differences by evidential support:

| difference | evidence | status |
|---|---|---|
| Injection scope (~3 vs ~65 positions) | verified in their `inject()` + task configs; their own stride result predicts the null | **strongest; code-confirmed** |
| Dose grid in span units | E-up: nothing → supported, under argmax | 1 of 3 poles decisive |
| Layer choice by ordering | A judge margin 0.36 → 7.2 | demonstrated, judge-side only |
| Collapse screen | 11/13 → 9/9 sign-correct | demonstrated, within our data |
| Keying balance | none | untested |
| Baseline headroom | explains our own N-up/O-up failures | untested vs theirs |
| ~~Vector construction~~ | cos 0.993 with two-arm contrast | **not a difference** |
| ~~Readout~~ | identical 11/13 sign tally | **not the primary difference** |

Full audit of our own numbers, including what should not be claimed, in
`docs/AUDIT_BEFORE_CONTACTING_AUTHORS.md`. The short version: five of ten
pole-directions show an argmax dose-response that clears a matched-norm random
control, conscientiousness works in both directions, and the methodological
critique of their design is factual — but there is one control draw per sweep,
no repeated rungs, and no head-to-head against the two-arm vector.
