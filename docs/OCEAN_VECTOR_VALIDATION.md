# Deciding whether an OCEAN vector actually works

This is the protocol for answering one question: does a Big Five steering
direction do something real, or does it only look like it does? It exists because
an earlier round of OCEAN work concluded "the vectors do not work" on evidence
that could not have supported that conclusion.

Entry point:

```bash
python3 scripts/run_ocean_vectors.py --run-id ocean_v1 --trait conscientiousness
python3 scripts/run_ocean_vectors.py --run-id ocean_v1 --all-traits
```

Plumbing check with no GPU and no credentials:

```bash
python3 scripts/smoke_ocean_pipeline.py
```

---

## The three deliverables

A direction is only interesting if all three hold at once.

| Deliverable | How it is measured | Where it lands |
|---|---|---|
| Monotonic dose-response | Spearman rho between `abs(magnitude)` and the inventory score, computed over unlocked rungs only | `trait_curve.spearman_absalpha_vs_target_ev` |
| Correlation against a reference | Prompting baseline rho(level, score) on the *same instrument*, plus rho between magnitude and free-text trait markers | `prompting_rho`, `trait_curve.marker_spearman` |
| Real behavioural change | Free-text replies at every rung, each with a coherence verdict and a high-minus-low marker rate | `trait_curve.rows[].probes` |

`verdict.works` requires the direction to beat matched-norm random controls, keep
at least three unlocked rungs, and produce a non-zero monotone trend.

---

## Four ways the old protocol produced a false null

Each of these is now a control rather than an assumption.

### 1. Magnitude was uncalibrated

Steering strength copied from a paper does not transfer between models. Gemma-3-4B
residual norms are on the order of `1e4`, so a literature `alpha` of about 2 on a
unit-norm direction is a no-op.

The measured evidence: the known-good `good` alignment vector on this model has
`norm(v) = 1227` at layer 15, and its own alpha sweep scores

| alpha | `norm(alpha * v)` | judged trait score |
|---|---|---|
| 0.5 | 654 | 1 / 100 |
| 1.0 | 1227 | 32 / 100 |
| 1.5 | 1841 | 95 / 100 |
| 2.0 | 2455 | 97 / 100 |
| 3.0 | 3683 | 97 / 100 |

Nothing happens below roughly 1800. An OCEAN sweep that tops out at an effective
magnitude of 800 is testing the region where the *known-working* vector also does
nothing, so its null says nothing about the direction.

Here, magnitude is always `norm(alpha * v)` on a unit-norm direction, and
`--alpha-units relative` expresses `alpha` in units of the layer's mean activation
norm so the grid is model-independent.

### 2. Steering pushed into a saturated ceiling

An RLHF-tuned model is already conscientious and agreeable. Measured on this
model, baseline conscientiousness is 4.13 on a 5-point scale: 0.87 of headroom
upward, 3.13 downward. Testing only the upward direction asks the direction to do
the one thing the prior leaves no room for.

`--steer-toward auto` reads the unsteered baseline and steers toward whichever
pole it is furthest from. Steering conscientiousness *down* from the prior moved
the score by more than a full scale point where steering up moved it by 0.28.

### 3. A collapsed readout scored as a clean null

This is the important one, because it is silent.

An acquiescence-corrected trait score is `(plus_keyed_mean + minus_keyed_mean) / 2`.
If the model answers the same option `v` to every item, plus-keyed items score `v`
and minus-keyed items score `6 - v`, so a keying-balanced trait averages to
**exactly the scale midpoint, whichever option was locked onto**. Meanwhile
`response_validity` reports 1.0, because every answer was a parseable option.

So "the model stopped discriminating between items" and "steering had no effect"
produce the same number, and the first is far more common at high magnitude.

`option_lock()` screens every administration on the raw option distribution and
reports a `locked` verdict; locked rungs are treated as missing data, not as
measurements. `scripts/smoke_ocean_pipeline.py` demonstrates the failure directly:
a randomly initialised 2-layer model answers option 1 to all 24 items at every
prompted level, which the unscreened path reports as `target_score = 3.0` with
`response_validity = 1.0` at all three levels — a flat, confident curve from a
model with no personality at all.

The prompting baseline is screened the same way, since a prompted level can lock
too, and a spurious midpoint there flattens the very curve the steering is
compared against.

Expected-value scoring (`score_traits_ev`) is reported alongside argmax scoring,
because the option distribution can still be moving after the argmax has pinned.

### 4. Behaviour was never measured

An inventory records what the model says about itself under a forced choice. That
is worth measuring, but it is not behaviour, and it is exactly the measurement
that collapses first.

Every rung now also generates free text, screened two ways:

- `coherence_metrics()` is a judge-free lexical screen for the repetition collapse
  that additive steering produces past its ceiling. It has to be judge-free
  because it runs at rungs that are broken on purpose.
- `marker_score()` gives a net high-minus-low trait marker rate per 100 words, as
  supporting evidence for the direction of a shift.

The largest magnitude still producing prose is reported as
`coherence_ceiling_magnitude`. That ceiling turns out to be a useful vector
quality signal in its own right: on the earlier runs a matched-norm random
direction degenerated into character soup at magnitude 1850, while a
dialogue-derived conscientiousness direction was still writing clean sentences at
5500. A direction that buys headroom before breaking is doing something
structured.

---

## A worked run, and why it fails

Qwen2.5-1.5B-Instruct, conscientiousness, PC1 direction at layer 16, the committed
120-item form, one ladder variant, one random control, on CPU.

**The instrument works.** The prompting baseline moves the score across most of the
scale and does it monotonically, with no locked administrations:

```
level means  [2.50, 2.58, 3.00, 3.13, 3.33, 4.42, 3.46, 4.63, 4.58]
rho = 0.967   range = [2.50, 4.63]   usable = 9/9
```

**Calibration finds the ceiling by itself.** Doubling upward, coherence holds to
magnitude 32.4 and collapses at 64.9 (type-token ratio 0.93 to 0.53), so the grid
is placed below 32.4. The layer's mean activation norm here is about 50; on
Gemma-3-4B it is around 2.7e4. The same relative alpha grid cannot serve both,
which is the whole argument for calibrating rather than choosing.

**The dose-response looks convincing.**

| magnitude | trait direction | random control |
|---|---|---|
| 0 | 3.29 | 3.29 |
| -2.0 | 2.94 | 3.32 |
| -4.1 | 2.77 *(locked, excluded)* | 3.34 |
| -8.1 | **2.83** | 3.32 |
| -16.2 | 2.85 | 3.14 |
| -32.4 | 3.07 *(locked, excluded)* | 3.00 |

`rho = -0.80` with the correct sign, 4 of 6 rungs usable, best delta `-0.46`
against a control delta of `0.29`. Under a "beats the control" rule this passes.

**It should not pass, and the probes are what show it.** The free text at the best
rung is:

> "I'm sorry, but as an artificial intelligence language model, I don't have
> personal experiences like humans do. However, I can tell you about common
> activities people might engage in on Saturdays..."

The model has stopped answering as a self. The inventory moved because the persona
collapsed into a disclaimer, not because conscientiousness fell. Trait markers
never fire across the sweep, consistent with prose that is not about the trait at
all. The margin over the control is also only 1.58x.

So the verdict requires more than beating a control:

- `control_margin_ratio >= 2.0` (`MIN_CONTROL_MARGIN`) — a large perturbation moves
  the score in some direction whatever it is, so winning by a nose means nothing.
- `dose_response_sign_correct` — steering toward "low" must push the score down.
- `refused_at_best_rung` false — `refusal_score()` detects collapse into
  disclaimer or refusal, which produces a real, monotone, control-beating curve
  for entirely the wrong reason.
- at least three unlocked rungs.

Under those rules this run reports `works = False`, which is the correct answer.

### Limits this run exposed

The coherence screen is lexical, so it accepts text that is grammatical but
semantically broken. At magnitude -32.4 it passed prose reading *"I'm sorry for the
success is too it"* with a type-token ratio of 0.93. It catches repetition
collapse, which is the dominant failure, and it does not catch semantic drift.
Refusal detection covers one specific and common case; the saved text is still the
primary evidence and is meant to be read.

---

## Instrument

`data/ipip_neo_120.csv` is a keying-balanced 120-item form derived from the
public-domain IPIP-NEO facet pool (`scripts/fetch_ipip_items.py` fetches all 300
items across 30 facets and selects 4 per facet, 2 plus- and 2 minus-keyed).

Balanced keying is deliberate. It is what makes acquiescence correction well
defined and makes midpoint pinning unambiguous. The built-in `IPIP_50` is
lopsided — neuroticism is 8 plus-keyed to 2 minus-keyed — so a lock there biases
the score toward the locked option instead of pinning it, which is harder to
diagnose. Prefer the 120-item form via `--items-csv data/ipip_neo_120.csv`.

Provenance and licensing are recorded in `data/ipip_neo_120.provenance.json`. The
form is IPIP-derived; it is not a reproduction of the published IPIP-NEO-120 item
selection.

---

## Reading the output

```
trait             toward steer rho  usable best delta    @mag ctrl delta marker rho  ceiling  works
conscientiousness low         -0.8     4/6    -0.4642 -8.1089     0.2941       None -32.4358  False
```

- `usable 4/6` — two rungs were option-locked and excluded. If this is `0/6`, the
  run measured nothing regardless of what the scores say.
- `best delta` vs `ctrl delta` — the trait direction has to beat matched-norm
  random directions on the same grid. Without this column a large delta only
  shows that a large perturbation changes behaviour.
- `ceiling` — largest magnitude still producing prose. A `best delta` obtained
  above the ceiling is an artefact of a broken model, not a trait shift.
- `marker rho` of `None` means the trait markers never fired, which is a hint that
  the prose is not about the trait; read the saved text before believing a score.
- `works` — every check passed: margin over controls, correct dose-response sign,
  enough unlocked rungs, and no refusal at the best rung.

The prompting baseline is printed underneath as the reference: prompting can move
these scores across most of the scale, so a direction whose dose-response is real
but much smaller is a real direction with a smaller effect, which is a different
claim from a direction that does not work.

---

## What this does not establish

Behavioural equivalence does not identify a representation. Many directions
produce the same behaviour, so a passing verdict here supports "this direction
reliably and monotonically moves the trait, and does so more than a matched random
direction" — not "this is the model's representation of the trait". See
`research/checkpoints/002-interpretability-causation-steering-conflict.md` for the
non-identifiability argument, and note that a contrastive direction on a
prior-resident trait encodes the transition away from the prior rather than the
trait's content.
