# The case for publishing this

Every number below points at a committed artifact in `results/`. Claims we cannot
support are in the last section, because a reviewer will find them anyway.

## The claim

A published null — that activation steering along personality directions produces
"no salient patterns" on psychometric inventories (Blas et al., arXiv:2604.14463,
§5) — is substantially a measurement artifact. We identify five measurement
faults, each independently testable, and show that repairing them turns the null
into a graded, dose-ordered inventory response that survives a dose-matched
random-direction control on 5 of 10 pole-directions.

The contribution is not "steering works after all." It is that a specific class
of null result in LLM psychometrics is not evidence about models, and that the
faults producing it are cheap to check and mostly arithmetic.

## The load-bearing observation

**On a keying-balanced inventory, midpoint responding scores exactly the scale
midpoint for every domain.** The 120-item IPIP-NEO form is 12 positive and 12
reverse-keyed items per domain, so an item answered "3" contributes 3.0 whether
or not it is reverse-scored. Therefore:

    refusal / hedging  ==  "average personality"  ==  "steering did nothing"

are arithmetically identical, at full response validity. `tests/test_ocean_validation.py`
asserts this for every option value. Any null on such an instrument is
uninterpretable until the option distribution is screened, and none of the work
we are responding to screens it.

This single identity is what makes the rest of the paper necessary, and it is
checkable by anyone in an afternoon.

## The five faults, with the ablation for each

### 1. The readout cannot see the movement

Inventory items scored by argmax over the option tokens — greedy-decoding one
constrained option, which is what the null we are addressing does — cannot
resolve a shift in the option distribution that has not crossed a decision
boundary. On **identical forward passes**, neuroticism-up gives ρ(dose, score)
= **0.37 by argmax and 0.98 by expected value** over the option-token softmax,
because the argmax takes only 2 distinct values across nine doses. Openness-high:
0.87 vs 1.00, again 2 distinct argmax values.

`results/readout_argmax_vs_ev.json` (14 sweeps, both readouts on the same
activations). We also report the argmax-only result so the claim does not rest on
our scoring choice: 5 of 10 poles clear 2× a random control under argmax alone
(`results/argmax_dose_response.json`).

### 2. The baseline is degenerate

The obvious "neutral" baseline — the midpoint rung of a prompted intensity
ladder, "I am neither organized nor disorganized…" — instructs neutrality, and a
forced-choice inventory answers that with the neutral option on essentially every
item: 97–100% of openness, agreeableness and conscientiousness items on
Gemma-3-4B. It is the only rung in a nine-level ladder that locks. A sweep based
there has no variance left to move, and reports a null with full validity.

### 3. The prior prompt was never verified

This is the fault we expect to be cited. Steering-from-an-opposite-prior designs
assume the low-pole prompt produces low-pole behaviour. On Gemma-3-4B it does not,
for one domain, and the failure is invisible because of the identity above.

Asked to describe itself as "very unimaginative, very uncreative, very incurious",
the model answers the neutral option on the openness items it will not endorse
("I do not like art", "I tend to vote for conservative political candidates"), and
scores **3.05 against a level-9 reference of 4.75** — an absent prior reading as an
average one. Extraversion has no such problem (prior 1.52) because the model will
call itself shy. The asymmetry is specific: the high pole of openness is adopted
fine.

We measured three persona framings × 5 domains × 2 poles
(`results/prior_prompt_calibration/summary.json`, 30 rows). Four of five domains
establish both priors under the original wording once six markers are used
(conscientiousness 1.21/5.00, extraversion 1.63/4.87, agreeableness 1.74/4.61,
neuroticism 1.70/4.85). **Openness fails under all three framings**; role-play
gets it to 2.58 against a 4.25 reference and no further.

Two consequences, and the second is the one nobody has stated:

- an opposite-prior sweep on that domain has no prior to move away from;
- **a steering vector fitted to level-conditioned activations inherits the
  non-compliance.** If the low-openness levels were never adopted, the level
  centroids differ across the ladder by hedging rather than by openness, and the
  fitted direction is a hedging axis wearing a trait's name. That is a concrete
  mechanism for an openness vector that moves agreeableness and conscientiousness
  on the inventory while leaving openness flat, which is what we observed.

We ship the diagnostic (per-domain option histogram at every level, not the
inventory-wide one) and the fix (`scripts/calibrate_prior_prompts.py`, then
`scripts/rebuild_ladder_vectors.py`, which refuses to build a trait whose priors
do not hold).

### 4. "Answer decisively" backfires

The natural repair for midpoint hedging — instructing the model to use the middle
option only when genuinely neutral — makes it worse, consistently: **100% of
neuroticism items, 92% of openness items and 75% of extraversion items** land on
the middle option. Naming the middle option appears to prime it. Anyone
administering inventories to LLMs is likely to try this instruction; it is worth
one line in a paper to stop them.

### 5. Steering layers are readout-specific, and the control was not dose-matched

Layers validated against an LLM-judge free-text readout do not transfer to a
forced-choice inventory. Moving openness from L19 to L15 turns Δ +0.005 into
**+0.725**; moving neuroticism from L20 to L15 turns Δ 0.0 with ρ = −0.88 (wrong
direction) into **+0.525**. The published null we are addressing, and our own
earlier write-up, both reuse layers across readouts.

Separately, the standard pass criterion compares the trait direction's best
movement anywhere on the dose grid against the random control's largest movement
anywhere on the grid — the trait at its best against noise at its worst. A random
direction is a control for what a perturbation of a given *size* does, so it must
be read at the same size. At α=790, conscientiousness-up has moved the inventory
+1.49 EV while the matched random direction has moved +0.26; at the ceiling rung
both have moved ≈+1.75, and the ceiling rung is what the flag reported. Re-scoring
dose-matched, with the band defined by the control's behaviour and never by the
trait's, gives 5/10 rather than 2/10 (`results/dose_matched_control.json`,
`scripts/dose_matched_control.py`).

## The positive result after repair

Gemma-3-4B, 120-item IPIP-NEO, expected-value readout, rebuilt vectors, calibrated
priors, inventory-validated layer (`results/opposite_prior_ipip/summary_v3_calibrated_prompts.json`,
scored in `results/dose_matched_control.json`):

| pole | Δ EV | ρ over band | dose-matched margin | % of prompting gap |
|---|---|---|---|---|
| openness-down | −1.07 | 1.00 | 17.8× | 75% |
| openness-up | +0.51 | 0.88 | 8.1× | 31% |
| conscientiousness-down | −0.43 | 1.00 | 4.6× | 11% |
| neuroticism-down | −0.69 | 1.00 | 3.4× | 24% |
| conscientiousness-up | +0.60 | 1.00 | 2.9× | 16% |

5 of 10 poles. The progression is the argument: openness-up goes from ρ = −0.50
(old vectors, uncalibrated prior) to ρ = +0.40 (rebuilt vectors, uncalibrated
prior) to ρ = +0.88 with an 8.1× margin (rebuilt vectors **and** calibrated
prior). Each repair is separately attributable.

On a free-text LLM-judge readout, **10 of 10 pole-directions are sign-correct**
(`results/bipolar/summary.json`), which is why the inventory-versus-judge gap is
a readout story rather than a steering story.

## Three secondary results that stand on their own

**Big Two.** Blas et al. report that 46.15% of cross-trait sign patterns match the
Digman metatraits, computed on SJT/classifier scores. The inventory version of the
same test on our sweeps gives **12/16 = 75%** (`results/big_two_covariance.json`).
Same theory, different instrument, different answer.

**The vector difference is the corpus, not the estimator.** On Llama-3.1-8B, the
shipped mean-difference vectors are near-orthogonal to our ladder PC1: max
|cos| = **0.067** across all 32 layers, mean 0.022. But a two-arm mean-difference
contrast built on *our* ladder data aligns with our PC1 at cos **0.90–0.99**. So
"PCA versus mean-difference" is cosmetic; what the vector is estimated from is
not. (`results/e1_vector/vector_geometry.json`,
`results/endpoint_vs_pc1_geometry.json`)

**Injection scope.** Matching a narrow injection span (2 tokens versus 79) destroys
bipolar control. Under full-sequence injection the vector's sign controls the sign
of movement (C-up +0.54, C-down −1.42, opposed). Under assistant-span injection
both poles drift the same way (+0.33, 0.00), so the residual span correlation is
not trait signal — a dose-response can be reported for something that has no
directional control at all.
(`results/injection_scope_ablation/bipolar_and_collapse_check.json`)

## Why this has not been done

The faults are all *below* the level anyone reports. Papers report the score, the
correlation, and sometimes a random-direction control; they do not report the
option histogram, whether the persona prompt was adopted, whether the layer was
validated on this readout, or at which dose the control was read. Each fault is
individually mundane and jointly they manufacture a null. Finding them required
running the pipeline against itself — the openness vector's failure is what
exposed the openness prompt's failure, which is what exposed the identity that
made both invisible.

## Form and venue

A short empirical paper: *"Null results in LLM personality steering are
measurement artifacts."* Sections map onto the five faults, one ablation each, on
one model and one instrument, with the repaired result as the demonstration rather
than the headline. Natural fit for an interpretability or evaluation workshop
(BlackboxNLP, an ICLR/NeurIPS workshop) or a findings track. The
screening-and-calibration scripts are the reusable artifact.

## What we cannot claim, and what a reviewer will demand

- **One model.** The positive result is Gemma-3-4B only. The Llama-3.1-8B port is
  unresolved: the prompted ladder is well-ordered only at L6–8, which is not in
  the steerable band, so ρ×monotonicity tops out around 0.55 even with three
  marker variants (`docs/FINAL_ABLATION_ATTEMPT.md`).
- **The decisive head-to-head has not run.** Their vector versus ours, same model,
  same instrument, same dose grid, same readout. Until that exists we can say the
  vectors are near-orthogonal and that our pipeline produces a dose-response; we
  cannot say ours would fix their result on their model.
- **5 of 10, not 10 of 10.** Agreeableness fails both poles dose-matched (margins
  1.4 and 1.8), extraversion-up is non-monotone over the quiet band (ρ = 0.09),
  neuroticism-up has no quiet band worth reading.
- **Openness's pass starts from 2.58, not a genuine low pole.** No framing we
  tested establishes a low-openness prior, so that row is "steering from a
  partially-lowered prior", and we should say so in the caption.
- **Down-poles are confounded by construction.** From a high prior, degradation
  and the intended effect both push the score toward the middle. The dose-matched
  band mitigates this; it does not remove it.
- **No repeated rungs.** The only noise estimate is the matched random direction,
  with one control direction per sweep. Repeats per rung are the obvious ask.
- **The E2 dose-matched injection-scope ablation was never run**, so the scope
  result is confounded by dose (79 positions versus 2 at the same α).
