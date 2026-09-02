# Reliable personality control on a published Big Five inventory

**Result.** A single steering vector per trait, injected at one layer of
Gemma-3-4B, moves that trait's score on the real IPIP-NEO-120 monotonically with
the size of the injection. Median dose-response correlation **ρ = 0.97**, with
**ρ ≥ 0.85 on 9 of 10 trait poles**. Turning the dial further moves the trait
further, in the direction you chose, on a test built for humans.

Everything below is computed from committed artifacts:
`results/final_cycle/FINAL_TABLE.json` (canonical numbers),
`phase4_sweeps_specificity.json` (per-rung raw), `phase1_reliability.json`.

---

## The headline table

| trait | ρ up | ρ down | Δ up | Δ down | bipolar range | α |
|---|---|---|---|---|---|---|
| openness | 0.95 | 1.00 | +0.79 | −0.90 | 1.69 | 0.90 |
| conscientiousness | 0.98 | 1.00 | +2.02 | −1.65 | **3.67** | 0.94 |
| extraversion | 0.88 | 0.95 | +1.80 | −1.77 | **3.57** | 0.78 |
| agreeableness | 1.00 | 0.85 | +1.01 | −1.40 | 2.42 | 0.87 |
| neuroticism | 0.60 | 1.00 | +1.14 | −0.59 | 1.74 | 0.94 |

Mean bipolar range **2.62 of 4 available Likert points** (the 1–5 scale has 4
points of travel). Conscientiousness and extraversion cover ~90% of it.

---

## What each number means

**Instrument.** MPI-120 (`data/mpi_120.csv`) — the published Johnson IPIP-NEO-120
reworded to second person, the same 120 items 619,150 humans have taken and the
same form used by Jiang et al. and Blas et al. Not a subset, not homemade: every
rung administers all 120 items.

**Score (EV).** Each item is presented alone and answered by the probability
distribution over the five Likert option tokens; the item's value is
`Σ p(option) × option`, reverse-keyed where the item is negatively keyed, then
averaged per domain. Expected value rather than argmax, so the signal stays
graded after the top choice saturates. Range 1–5, midpoint 3.

**Steering.** One vector `v` per trait, extracted as PC1 across the nine
level-conditioned activation centroids of the prompt ladder, taken at
**layer 15** and unit-normalised. At each rung the residual stream at layer 15
receives `+α·v̂`. Nothing else changes.

**Dose (α).** Expressed in units of the trait's own **latent span** — the
projection of (level-9 centroid − level-1 centroid) onto `v̂` at layer 15. A dose
of 1.0 span is the activation distance prompting itself traverses from one pole
to the other, so the grid is comparable across traits instead of being an
arbitrary norm. Eight rungs from 0.15 to 1.30 span.

**ρ (dose-response correlation).** Spearman correlation between `|α|` and the
target domain score across usable rungs, signed so that positive means "more
dose, more trait in the intended direction." This is the reliability claim: it
says the dial is ordered, not that it is strong.

**Δ (total delta).** Target-domain score at the best rung minus the score at
α = 0, in Likert points. The size of the effect.

**Bipolar range.** `|Δ up| + |Δ down|` — the full interval of the human scale the
one vector can reach by sign alone.

**Baselines (opposite-prior design).** Steering up starts from the level-2
(low-trait) prompt and steering down from the level-8 (high-trait) prompt, so each
direction has room to move. A sweep started at the model's own midpoint has ~1
point of headroom and cannot show a real effect even if one exists.

**α (Cronbach's alpha).** Internal consistency of the 120-item instrument on this
model, across four paraphrased administrations. 0.78–0.94, i.e. the test behaves
like a test here — a precondition for any score movement meaning anything.
All five domains cleared the reliability gate: none locked, all sign-stable.

**Collapse screening.** Every administration is checked for option lock (one
option covering ≥90% of items, or answer entropy < 0.30 nats). A locked rung is
treated as missing data, not as a measured score, so a degenerate readout cannot
masquerade as a midpoint.

---

## Why the movement is trait-specific, not a global slide

Two quantities per rung, both free because all 120 items are scored every time:

- **off-target** — mean |Δ| across the other four domains.
- **drift** — mean *signed* Δ across all five domains. Large drift means every
  domain slid the same way, which is degradation wearing a trait-shaped costume.

Compared against a **matched-norm random direction at the same dose**, restricted
to doses where the random direction has not yet slid the whole readout
(`|drift| < 0.25`):

| pole | dose | trait Δ | random Δ | trait off | random off | winner |
|---|---|---|---|---|---|---|
| O-up | 612 | **+0.78** | −0.60 | 0.61 | 0.30 | trait |
| O-down | −346 | **−0.39** | −0.05 | 0.20 | 0.20 | trait |
| C-up | 919 | **+2.02** | +0.47 | 0.63 | 0.64 | trait |
| E-up | 1051 | **+1.80** | +1.27 | 0.86 | 0.86 | trait |
| E-down | −387 | **−0.58** | −0.17 | 0.39 | 0.25 | trait |
| A-up | 955 | **+1.01** | −0.66 | 0.36 | 0.44 | trait |
| N-up | 638 | **+1.14** | +0.81 | 1.18 | 0.31 | trait |
| N-down | −315 | **−0.35** | +0.05 | 0.39 | 0.29 | trait |
| C-down | −387 | −0.22 | −0.29 | 0.37 | 0.22 | random |
| A-down | −593 | −0.54 | −1.17 | 0.11 | 0.39 | random |

**8 of 10 poles: the trait vector moves the target more than a random direction
of identical norm at identical dose.** The dose match matters — comparing each
direction's *best* rung instead flatters random, because random's best is always
its largest dose, where it is destroying the model rather than steering it.

A-down is the instructive exception: random moves the target further (−1.17 vs
−0.54) with **3.5× the off-target spread** (0.39 vs 0.11). It is moving
everything; the vector is moving one thing.

---

## The Big Two: recovering the structure they reported missing

Blas, Jia & Ferrara (*Psychological Steering…*, arXiv:2604.14463, Sec 5.3) tested
whether steering reproduces the **Big Two metatraits** — the higher-order factors
the Big Five collapse into in human data (Digman 1997; DeYoung 2002):

- **α / Stability** = C+, A+, **N−**
- **β / Plasticity** = E+, O+

The test: if the traits sit on real metatraits, steering one trait should drag its
same-metatrait partners in the *predicted* direction. They found **46.15%** of
sign patterns matched — indistinguishable from the 50% coin flip — and that no
model satisfied all Big Two correlations. Crucially, they computed this on
**SJT/classifier-scored text**, having already dropped the inventory as
patternless.

Run on inventory scores, it comes back (`results/final_cycle/big_two_covariance.json`,
`scripts/big_two_final_cycle.py` — pure reanalysis, no GPU):

| pole | metatrait | on-target Δ | shared drift | drift/signal | matched | drift-proof |
|---|---|---|---|---|---|---|
| A-down | α | −1.40 | −0.37 | 0.27 | 2/2 | 1/1 |
| A-up | α | +1.01 | +0.42 | 0.41 | 2/2 | 1/1 |
| C-down | α | −1.65 | −0.55 | 0.33 | 2/2 | 1/1 |
| C-up | α | +2.02 | +0.44 | 0.22 | 2/2 | 1/1 |
| E-down | β | −1.77 | −0.86 | 0.49 | 1/1 | — |
| E-up | β | +1.80 | +0.73 | 0.41 | 1/1 | — |
| N-down | α | −0.59 | +0.33 | 0.55 | 2/2 | — |
| N-up | α | +1.14 | −0.71 | 0.62 | 2/2 | — |
| O-down | β | −0.90 | −0.39 | 0.43 | 1/1 | — |
| O-up | β | +0.79 | +0.55 | 0.70 | 1/1 | — |

**16/16 predicted pairs matched (100%, exact binomial p = 3×10⁻⁵ against chance)**
versus their 46.15%. Restricted to the 7 sweeps where drift is small relative to
signal: **11/11, p = 0.001**.

**Why this is not just shared drift.** The obvious objection: a big injection
degrades the respondent, all five domains slide the same way, and the sign pattern
imitates α structure while carrying no trait information. Shared drift here is
real (0.22–0.70 of signal), so the objection has to be answered rather than waved
off.

α answers it, because **N loads negatively**. When C or A is steered up, α predicts
N moves *down* — opposite to the direction everything else is sliding. A global
slide cannot produce that. Those **drift-discordant** pairs:

| sweep | shared drift | N predicted | N actual | |
|---|---|---|---|---|
| C-up | **+0.44** | − | **−1.17** | ✓ |
| C-down | **−0.55** | + | **+1.62** | ✓ |
| A-up | +0.42 | − | −0.18 | ✓ |
| A-down | −0.37 | + | +0.37 | ✓ |

**4/4.** Steering conscientiousness up pushed neuroticism down 1.17 points while
the five-domain average moved *up* 0.44. That is the metatrait, not degradation.

The β poles (E, O) carry no discordant pairs — β's loadings are both positive, so
its predictions always point the same way as drift and cannot discriminate. The β
rows are reported as matches but should be read as uninformative on the confound;
the α evidence is what carries this result.

**Definitions used above.** *Shared drift* = mean signed Δ across all five
domains. *drift/signal* = |shared drift| / |on-target Δ|. *Drift-proof* = predicted
pairs whose predicted sign opposes the sign of shared drift. Deltas are taken at
the rung where the target moved furthest in the intended direction, against the
α = 0 rung of the same sweep, option-lock screen applied to both.

---

## Where this sits in the literature (verified Aug 31, 2026)

"Real test" here means: the model itself answers the published Likert items and
is scored psychometrically (keying, domain means) — no judge model anywhere in
the loop. Under that definition, the steering literature looks like this:

- **Blas, Jia & Ferrara (arXiv:2604.14463)** — the only steering paper that
  administered a published inventory directly (MPI-120, one constrained option
  token, greedy). Their own words: *"none of these observations applied to the
  inventory responses; we observed no salient patterns."* Their linearity and
  Big Two claims are computed on judge/classifier-scored SJT text, and the Big
  Two result is a **departure** from the metatrait structure. `gemma-3-4b-it`
  is in their 14-model list — the same model used here.
- **SAS personality sliders (arXiv:2603.03326)** — GPT-4 judge scores generated
  responses to BFI items. Not a psychometric administration.
- **PERSONA (arXiv:2602.15669)** — BFI-44 rewritten into scenarios by GPT-4o,
  scored by GPT-4.1-mini. Not a psychometric administration.
- **Behavioural Asymmetry (OpenReview X89cn3iy8b)** — gpt-oss-20b and
  gpt-5.4-nano judges on 30 generations per condition. Also finds Neuroticism
  hardest to steer, which corroborates our weakest pole.
- **Mechanistic Personality Analysis / SAE interventions (arXiv:2606.28770)** —
  embedding similarity, LLM classification, human raters. No inventory.
- **PAS (Zhu et al., arXiv:2408.11779)** — the nearest prior work and the one to
  cite carefully. Real IPIP-NEO items, answered directly by the model. But the
  question is different: per-subject attention-head interventions optimized to
  minimize |model answer − one human's answer| on held-out items (profile
  mimicry). No dose-response on domain scores, no reliability or collapse
  screening, no metatrait analysis.

The two claims that appear to be novel: (1) **dose-response control of Big Five
domain scores on a directly administered published inventory** (ρ up to 1.00,
median 0.97), and (2) **recovery of Big Two covariance under steering**, where
the one paper that tested it reported a departure. Both were obtained on a model
where the null was originally reported, by changing the readout (EV over the
option-token distribution instead of one-token argmax) and the baseline design
(opposite-prior instead of midpoint), with dose calibrated in latent-span units.

---

## Honest limits

- **Magnitude, not parity.** Prompting on this instrument spans 1.76–3.68 points
  (`results/final_cycle/ladder/prompt_ladder_*.json`). Steering recovers roughly
  40–70% of that gap. The claim is a reliable, specific, ordered dial — not that
  activation steering matches a prompt.
- **N-up is weak.** ρ = 0.60 and off-target (1.18) exceeds on-target (1.14) at
  the matched dose: at that dose neuroticism-up is closer to a global shift than
  to trait control. It is reported, not hidden.
- **No error bars.** Every rung is a single administration of the 120 items. The
  ρ values are ordering statistics over eight rungs; the Δ values are point
  estimates. Repeats at the identified doses are the next compute worth spending.
- **The Big Two result rests on 16 pairs, 4 of them confound-proof.** 100% of a
  small n. It is a sign test at one rung per sweep, not an estimated correlation
  matrix, and it does not establish the *magnitude* of the metatrait loadings.
- **One model, one layer.** Gemma-3-4B, layer 15. Multilayer injection with
  constant per-layer α was tested previously and did not work
  (`results/patch_multi/`); it is closed, not open.
- **Extraction protocol frozen.** PC1 at layer 15, chosen before these results
  were seen. No method search was run against this table.

---

## Reproducing

```bash
python3 scripts/final_cycle_run.py \
  --out-dir results/final_cycle \
  --items-csv data/mpi_120.csv \
  --rungs 8 --random-controls 1 --phases 1,2,3,4
```

Phase 1 gates reliability, phase 2 runs the prompting ladder and collects
level-conditioned activations, phase 3 extracts PC1 at layer 15, phase 4 sweeps
both poles with the random control. ~21 min on one L4.
