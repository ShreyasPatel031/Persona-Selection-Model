# The persona selection model — blog draft (WIP)

Status: working draft. Source notes in `persona-selection-model-notes.md`.
Visual spine: Inferno series at `app/static/viz_series.html`.
Numbers below are scoped to committed artifacts (`results/dose_matched_control.json`,
`results/big_two_covariance.json`, opposite-prior repaired v3). Do not publish
without re-checking those files.

---

## Why this draft is being rewritten

The early notes framed the post as Anthropic’s **persona selection model**:
pre-training puts a distribution over personas in the weights; post-training
updates that distribution with (x, y) evidence; the Assistant is one hypothesis
among many, and its neural footprint looks like other personas in the corpus.

That framing still opens the piece. What changed is the **empirical middle**:

1. We can **see** and **compose** persona directions (D&D / alignment vectors) —
   layer, α, SAE features, composition board.
2. We can now **steer Big Five inventory scores** with dose–response that
   sometimes clears matched-norm random controls — not just judge-scored free
   text.
3. The Blas et al. inventory null is partly a **measurement / baseline-headroom
   artifact**, not proof that inventories don’t move.

So the blog is no longer “here is a theory of personas.” It is:

> Personas are selectable directions in activation space. You can watch them
> climb with α, decompose them into SAE features, compose them — and, with the
> right readout and prior, move a real personality inventory.

The finale is no longer the alignment composition board alone. It ends on
**Five Personas** (OCEAN silhouette) — still WIP as a productized viz, but the
science claim is already there.

---

## Proposed structure (six beats = six viz chapters)

Use the Inferno series as the reader’s path. Each section embeds or links one
page. Do not invent new chrome; same cream / Newsreader system.

| Beat | Viz | Job in the post |
|---:|---|---|
| 01 | `layer3d.html` | Where a trait lives in the residual stream |
| 02 | `inferno_cone.html` | α climbs: refuse → accept (behavior) |
| 03 | `omp_reconstruction_3d.html` | The steering vector ≈ sparse SAE reconstruction |
| 04 | `ssv_bubble_viz_omp.html` | Which features fire as the code grows |
| 05 | `dnd_composition_board.html` | Compose alignment vectors → one reply |
| 06 | `big_five_persona.html` | **Finale:** compose OCEAN → inventory silhouette |

Hub: `app/static/viz_series.html`.

---

## Act 0 — Open on the persona selection model

Keep almost verbatim from the notes:

- Pre-training → distribution over personas; Assistant is one hypothesis.
- Post-training → Bayesian-ish update from (x, y) episodes.
- Anthropomorphic slip (“our ancestors,” “navy blue blazer”) as surface evidence
  that the Assistant hypothesis is tangled with human-like personas.
- Interpretability: Assistant representations ≈ other personas in training data
  (not learned from scratch).

**Bridge sentence (new):** if the Assistant is a selectable persona among many,
then *other* personas should also be selectable — not only by prompting, but by
moving activations along a direction. The rest of the post is that experiment,
made visible.

---

## Act 1 — A persona you can turn like a dial (D&D / Good–Evil)

Narrative: start with a vivid, non-psychometric persona (Good vs selfish) so
readers feel α before we talk Likert scales.

- **Layer Viz** — trait activation isn’t uniform; some layers carry the axis.
- **Nine Alphas of Evil** — hard-selfish prior + Good vector @ L15; climb α;
  scenarios flip NO → YES. This is the emotional hook of the design system.
- **OMP / bubbles** — the vector isn’t magic: it reconstructs from sparse
  features; bubbles show what “more Good” lights up.
- **Composition board** — Lawful/Chaotic × Good/Evil as vector addition, not
  prompt salad.

Honest limit for this act: behavioral / judge-scored flips are cleaner than
inventory movement. That asymmetry is a feature of the story, not a bug — it
matches what Blas et al. saw between SJTs and inventories, and sets up Act 2.

---

## Act 2 — The breakthrough: Big Five inventory steering

This is the new spine. Lead with the contradiction, then the fix.

### The received null

Blas, Jia & Ferrara (*Psychological Steering…*, arXiv:2604.14463): steering
moved style / SJT-like probes, but **no salient pattern on MPI-120 inventory**;
they dropped the inventory and reported Big Two on SJTs (~46% sign match).

We reproduced inventory flatness / wrong-way movement on `gemma-3-4b-it` with a
two-arm mean-diff setup first. So the null is real under that protocol.

### Why inventories looked dead (method, not soul)

Field split (see `docs/WHY_NO_ONE_GOT_THIS.md`):

- Papers that **generate** answers → “inventories unreliable,” high refusal.
- Papers that score **option-token probabilities** (Serapio-García) → graded,
  reliable shaping.

We never decode a free answer for the inventory score: expected value (and
argmax) over Likert option tokens. Template held fixed across the dose sweep —
we measure **slope**, not absolute persona.

### What actually moved the needle

Not “PC1 vs mean-diff” as a geometry story (those directions are nearly the
same on ladder data, cos ≈ 0.99). The load-bearing pieces:

1. **Prompt-ladder vectors** with a layer criterion that cares about *ordering*
   of intensity levels, and a dose unit tied to full-scale change.
2. **Opposite-prior baselines** — steer away from a low (or high) prompted
   prior so there is headroom; persona-free RLHF priors sit near the ceiling
   on several traits and crush magnitude.
3. **Dose-matched random controls** — ρ alone lies; N-up can have ρ ≈ 1 with
   Δ ≈ 0.04 that random directions beat.
4. **Option-lock / coherence screens** — throw out collapsed rungs instead of
   averaging garbage.

### Headline numbers (repaired opposite-prior / dose-matched v3)

From `results/dose_matched_control.json` sweep `summary_v3_calibrated_prompts`:

| Pole | Dose-matched pass? | Notes |
|---|---|---|
| O-up, O-down | yes | Openness moves both ways |
| C-up, C-down | yes | Cleanest conscientiousness band |
| N-down | yes | Stability pole |
| A-up, A-down | no | Orders, small vs control |
| N-up | no | Classic headroom / tiny Δ failure mode |
| E-up, E-down | no | Partial / messy |

**5 / 10 poles pass** dose-matched control. Free-text / judge side was closer to
**10 / 10** sign-correct in earlier runs — same SJT-vs-inventory asymmetry,
inventory no longer flat.

**Big Two on the inventory** (`results/big_two_covariance.json`): ~**75%**
predicted cross-trait sign match vs their **46%** on SJTs. Small n; report
shared-drift confound (midpoint collapse at high α can fake α-structure).

### What “Five Personas” shows

`big_five_persona.html`: dial magnitude on each OCEAN pole; the pentagon is the
**measured inventory EV curve**, not a decoration. Green = dose-matched pass;
grey = still soft. Composition of multiple traits = the blog’s ending image:
persona selection as a silhouette you can paint.

---

## Act 3 — What this means for the persona selection model

Tie back without overclaiming:

| Theory claim | What the experiments add |
|---|---|
| Distribution over personas in weights | Directions in residual space that behave like persona axes |
| Post-training updates Assistant hypothesis | Steering is a *surgical* update at inference — same geometry, temporary |
| Assistant ≈ other personas neurally | SAE / OMP: steering vector reconstructs from features that look like traits |
| Anthropomorphic slip | Optional color; don’t hang the argument on anecdotes |

**New claim the blog can own:** persona selection is not only a training story.
It is an **inference-time control surface** — and for Big Five, that surface
reaches the same family of instruments psychologists use on humans, once you
stop decoding free text and stop measuring from a saturated prior.

**Claims the blog must refuse:**

- “We fully steer all five traits.” (5/10 dose-matched.)
- “Inventories are solved.” (Dorner factor-fit still open; template intercepts
  still matter for absolute scores.)
- “Our vector geometry uniquely beat Blas.” (Ladder vs their statement corpus
  is the open head-to-head; estimator alone is not the story.)
- Midpoint collapse / degradation at high dose as “more personality.”

---

## Directions to decide (editorial)

Pick one primary angle; the viz series supports all three.

### A. *The inventory wasn’t broken — the test was* (methods essay)

Audience: ML + psychometrics Twitter. Lead with Gupta/TRAIT vs Serapio-García
readout split, then show dose curves. Viz: Five Personas + a static ρ/Δ table.
Risk: dry; reward: hardest to dismiss.

### B. *Turn the Assistant into a person you chose* (product / demo essay)

Audience: builders. Lead with Inferno cone and composition board; land on
OCEAN silhouette as “character sheet.” Risk: looks like a toy; must keep the
dose-matched honesty box on screen.

### C. *A letter to the steering literature* (research narrative)

Audience: authors of Blas et al. + mech-interp. Structure like the email draft:
reproduce null → change prior/readout → partial recovery → Big Two on
inventory. Viz series as appendix figures. Risk: narrow; reward: citation-shaped.

**Recommendation:** **A as spine, B as packaging.** Open with cone (B energy),
spend the middle on readout + opposite-prior (A), end on Five Personas as the
composed Assistant hypothesis (theory + demo).

---

## Open work before this is shippable

- [ ] Embed live viz (hosted static) or high-quality screen recordings of the
      six chapters — HTML iframes only if CSP/hosting allows.
- [ ] Reconcile email-draft numbers with v3 dose-matched table (some email
      rows are older / partial Colab logs).
- [ ] One honest figure: ρ vs Δ vs control margin for all 10 poles.
- [ ] Midpoint-collapse callout (all traits → ~3 at high α) so readers don’t
      misread the silhouette.
- [ ] Decide publish venue (personal blog vs Distill-ish vs short arXiv note).
- [ ] Polish `big_five_persona.html` copy once editorial angle is locked.

---

## Scratch / keep from the original notes

```
- Pre-training teaches an LLM a distribution over personas…
- Post-training updates this distribution using (x, y) as evidence…
- Anthropomorphic language (“our ancestors,” Project Vend blazer)…
- Secret goals stay latent unless topics force them…
- Assistant neural reps ≈ other personas in training data…
```

Everything else in the published piece should earn its place by connecting to
a viz chapter or a committed result file.
