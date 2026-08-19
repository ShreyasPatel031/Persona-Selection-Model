# Pre-publication audit of the inventory claims

Written before contacting any paper authors. Everything here is recomputed from
committed artifacts by `scripts/audit_inventory_claims.py`,
`scripts/readout_argmax_vs_ev.py` and `scripts/audit_readout_and_noise.py`.

**Verdict: the judge results are strong, the inventory results are not yet strong
enough to lead with. Do not contact authors yet.** The reasons are below, with the
specific runs that would fix each one.

---

## What the two pillars actually show

### Judge side (`results/bipolar`, `results/bipolar_afix`) — strong

10 of 10 pole-directions sign-correct, with large effects measured from an
*opposing* prompt prior, so there is real headroom:

| trait | pole | ρ | baseline → extreme | % of prompt gap | control margin |
|---|---|---:|---|---:|---:|
| extraversion | down | −1.00 | 91.7 → 10.0 | 102% | 81.7 |
| extraversion | up | +0.94 | 15.0 → 95.0 | 102% | 2.3 |
| conscientiousness | up | +1.00 | 15.0 → 93.3 | 98% | 46.9 |
| conscientiousness | down | −0.77 | 91.7 → 20.0 | 92% | 42.9 |
| neuroticism | up | +0.94 | 18.3 → 86.7 | 89% | 13.7 |
| openness | down | −0.89 | 90.0 → 31.7 | 83% | 8.8 |
| openness | up | +0.89 | 15.0 → 71.0 | 71% | 16.8 |
| neuroticism | down | −0.89 | 90.0 → 45.0 | 59% | 27.0 |
| agreeableness | down | −0.94 | 91.0 → 5.0 | 104% | 32.2 (L15 refit) |
| agreeableness | up | +0.89 | 6.0 → 95.0 | 103% | 7.2 (L15 refit) |

Caveats that are real but not fatal: **3 generations per rung**, a single judge
model (`gemini-2.5-flash`) with no human validation and no inter-rater
reliability, and agreeableness needed the layer-selection fix before it beat
controls at all (at layer 20, A-up moved the judge *less than a random vector*;
margin 0.36).

The honest problem with this pillar is not its quality — it is that **judge-scored
steering results are not novel.** The literature is full of them. Our novelty
claim rests entirely on the inventory, which is our weaker pillar.

### Inventory side — weaker than previously reported

Sign-correctness across the 14 sweeps, as a function of how strictly a rung must
behave to count as a measurement:

| usability rule | sign-correct | measurable |
|---|---:|---:|
| as shipped (top option < 90%, entropy ≥ 0.30) | 11/13 | 13/14 |
| top option < 80% | 9/11 | 11/14 |
| **top option < 75% and ≥ 4 of 5 options used** | **9/9** | **9/14** |
| all 5 options used and top option < 75% | 3/4 | 4/14 |

Two things follow, and they point in opposite directions.

**Good news, and it is the real finding.** Both wrong signs (E-up and N-up on the
original ceiling-dosed grids) **flip to correct** once rungs with a partly
collapsed response distribution are excluded. Under the `top < 0.75 and ≥ 4
options` rule every measurable dose-response is sign-correct. The wrong signs were
never trait effects; they were readout degradation being scored as though it were
personality.

**Bad news.** That rule leaves only 9 of 14 measurable, and the five it drops are
mostly the *large* effects: C-down (Δ 1.06), N-down (Δ 0.42), E-down, O-up. Our
biggest inventory movements happen precisely where the respondent is falling
apart. Under the strictest rule only **C-up (+1.00, n=4) and N-up (+1.00, n=3)**
survive as clean measurements.

---

## The four things a reviewer will attack

### 1. Only one random control per sweep

`n_random_controls = 1` in every run. Every specificity claim rests on a **single
draw** of a matched-norm random direction. "The trait vector beats random" is
currently "the trait vector beats one particular random vector."

*Fix:* re-run with 5–10 control draws and report the control delta distribution,
not a single max.

### 2. No rung is ever repeated, so there are no error bars

`n_repeated_doses = 0` everywhere. Every point on every curve is one
administration of 120 items at temperature-free constrained scoring. The scoring
is deterministic given the prompt, so repeats require prompt variants (the
`--variants` marker rotation the prompt ladder already uses) — but we never ran
them on the sweeps. A monotone 9-point trend with no variance estimate is not
something to put a ρ on in public.

*Fix:* 3 marker variants per rung, report mean and spread.

### 3. Our headline N-up effect is invisible to the standard readout

This is the one to be most careful about, and the earlier framing in
`DELTA_VS_PSYCHOLOGICAL_STEERING.md` overstated the general case.

The correction: **argmax and expected value give the same 11/13 sign-correct
tally.** The readout does *not* explain the overall pattern. What it changes is
whether one specific small effect is visible:

| | ρ argmax | ρ EV | score movement |
|---|---:|---:|---|
| N-up in-span | +0.37 | +0.98 | argmax Δ 0.042, EV Δ 0.083 |

24 neuroticism items × 0.042 ≈ **one item changes its committed answer** across
the entire dose range. A human respondent scored the same way would look
unchanged. The expected-value readout is measuring a shift in the option
*distribution* that has not become an answer.

Worth knowing what else is happening at that rung: the top-option fraction climbs
0.642 → 0.792 as dose rises, and the raw histogram moves from
`{1:1, 2:77, 3:38, 4:3, 5:1}` to `{1:5, 2:95, 3:17, 4:1, 5:2}` — roughly 21 items
sliding from "neither" to "moderately inaccurate". That is an **acquiescence
shift**, which balanced keying is designed to cancel, and the reported 0.083 is
the residual after that cancellation. So the trait signal is small relative to a
much larger nuisance movement, and it is riding on a distribution that is
progressively collapsing.

It is not nothing: the matched-norm control moves N *down* by 0.103 over the same
doses while the trait direction moves it *up* by 0.083, which is a genuine
directional separation. But the defensible claim is "the option distribution shifts
monotonically with dose", not "the inventory score moves".

*Fix:* the opposite-prior design. From an opposing prior there is headroom, and an
effect large enough to survive argmax. That run is still incomplete.

### 4. `works=True` is not comparable across result sets

Four sweeps report `works=True`. That number is misleading in both directions.
Most `gemma_final` sweeps report `control_margin_ratio: null` → `beats_controls:
false` → `works: false` **because of a stale code path**: when the control never
moved in the steered direction, `max_control == 0`, and the older code returned
`None` instead of the "infinite margin" sentinel that
`app/persona/intensity_ladder.py` uses now. So C-up (ρ +1.00, Δ 0.53, healthiest
rung in the whole set) is recorded as a failure.

*Fix:* re-derive verdicts from the curves with current code rather than trusting
the stored `works` field. Do not quote `works` counts until then.

---

## Additional specifics that need cleaning up

- **E-up in-span is not monotone.** EV goes 3.069 → 3.016 → 2.974 → 2.974 → 3.025
  → 3.127 → 3.281 → 3.396 → 3.331: down first, then up, then down. `ρ = +0.75`
  over |magnitude| hides a U shape; `monotone_fraction = 0.50`. Its best rung uses
  only 3 of 5 options (`{1:32, 2:70, 5:18}` — no 3s or 4s at all), which is a
  distributionally strange respondent, and under the strict rule its sign flips.
- **E-down never worked**, on either grid. 2 usable rungs in-span, 3 on the
  ceiling grid, and it fails every stricter rule.
- **C-down's Δ 1.06 is at a broken rung.** Best rung histogram
  `{1:47, 2:10, 3:21, 5:42}` — bimodal at the two extremes. It passes the lock
  screen (top fraction 0.392) but no coherent respondent answers that way.
- **N-down margin is 1.42 and O-down is 1.14**, both below our own 2.0 threshold.
- **Monotone fractions are mostly 0.5–0.88**, not 1.0. "Monotonic dose-response"
  should be stated as "monotone in rank" (Spearman), not "monotone".
- **Single model, single seed.** Blas et al. cover 14 models. Any comparison we
  draw is one model against their fourteen.

---

## What is safe to say right now

These do not depend on our effect sizes:

1. Blas et al. administered MPI-120 at every rung of their sweep, argmax-scored
   it, found no pattern, and dropped it in favour of judge-scored SJTs. Their
   Big Two conclusion is therefore computed on judge/classifier text, not on
   inventory scores. **This is a factual description of their paper.**
2. Their dose grid runs to the fluency ceiling, and they have no screen for a
   collapsed inventory response distribution. **Also factual, from their §5.1.**
3. In our data, excluding rungs whose response distribution has partly collapsed
   changes wrong-sign dose-responses into correct-sign ones (11/13 → 9/9). This is
   a within-our-data result on one model, stated as such.
4. Our judge-side bidirectional control from opposing priors is clean, and is
   consistent with — not a challenge to — the existing judge-based literature.

## What is not safe to say yet

- "We show monotonic dose-response on a real psychometric inventory." Not with one
  control draw, no repeats, and the biggest effects sitting on degraded rungs.
- "N is steerable where the literature says it is not." The N-up inventory effect
  is one item in 24 on the standard readout.
- Anything about Big Two structure from the inventory. We have not computed
  cross-trait covariance under the strict usability rule, and at overdose it is
  dominated by a shared degradation drift.

## Minimum work before contacting anyone

1. **Finish the opposite-prior IPIP sweep** — this is the one that gives effects
   enough headroom to survive an argmax readout. Currently incomplete.
2. **5–10 random controls per sweep**, reporting the control distribution.
3. **3 prompt-marker variants per rung** for error bars.
4. **Adopt `top < 0.75 and ≥ 4 options` as the primary usability screen**, with
   the shipped screen as a reported sensitivity analysis.
5. **Re-dose the remaining six poles in-span**, and fix E-down or drop it
   explicitly.
6. **Second model** (a non-Gemma family) for at least the two clean traits.
7. Recompute all verdicts with current code.

Only after 1–4 does the inventory pillar carry the novelty claim.
