# Draft email to the authors of arXiv:2604.14463

Status: draft for human review. Every number below is traceable to a committed
artifact; see the provenance table at the bottom before sending.

---

**Subject:** Paper reproduction: recovering inventory dose-response and Big Two covariance with a nine-level prompt-ladder vector

Dear Dr. Blas and colleagues,

I've spent the last week working through *Psychological Steering of Large Language
Models*, and I wanted to share a reproduction attempt plus one methodological change
that partially moved the inventory result. I'd value your read on whether it holds
up, and I have two questions about your setup at the end.

**What I set out to reproduce.** Two negative results in particular: that steering
produced no salient pattern on the inventory scores, and that you did not recover
Big Two covariance structure.

**I reproduced the first one, on a different model.** I started from your approach
directly: two-arm mean-difference vectors built from construct/antithesis statement
prefills, MPI-120, one item per administration, answers restricted to the option
tokens with greedy decoding. On `gemma-3-4b-it` this did not move the inventory in
the intended direction. Up-steering moved the target dimension the *wrong* way for
three of five traits (O −0.46, C −0.42, A −1.13 Likert points), and after a per-trait
(layer, α) retune it was still wrong for the same three (−0.08, −0.42, −0.50).
Off-target movement frequently exceeded on-target movement — steering openness up
moved extraversion by −1.71 and conscientiousness by −1.25. So on our setup the
inventory null reproduces, and if anything is stronger than what you report. That
early run had no matched-norm random controls and one dose per trait, so I'd treat it
as directional rather than precise.

**Why I kept going.** Prompting (Serapio-García et al., *Nat. Mach. Intell.* 2025)
and fine-tuning (BIG5-CHAT) both move real inventories in a graded way. Reproducing
Serapio-García's nine-level prompting ladder on the same Gemma model, I get
Spearman ρ between prompted intensity level and observed trait score of 0.84–0.97
across the five domains. So the instrument is movable on this model; that pointed me
at the vector rather than the questionnaire.

**The change.** Instead of contrasting two piles of generated statements, I derived
the direction from the *nine-level prompting ladder itself*: administer the inventory
under nine graded persona instructions (Goldberg adjective markers × Likert
qualifiers, level 5 neutral), record the residual stream at the answer position for
each level, and take PC1 across the nine level centroids. Two things this buys that a
two-arm contrast cannot, because you need at least three points to check an ordering:
a layer-selection criterion (rank layers by how well-*ordered* the nine levels are
along the direction, not by separation) and a dose unit (the projected distance from
level 1 to level 9, so α is expressed in units of one full-scale personality change).

**What that produced on `gemma-3-4b-it`:**

- *Free-text behaviour*, scored by a blind Gemini-2.5-Flash judge with opposing
  priors: **10 of 10 pole-directions sign-correct**, closing 18–102% of the gap that
  the explicit persona prompt achieves.
- *Inventory ordering*: **8 of 10 pole-directions order near-perfectly with dose** —
  |ρ| ≥ 0.85 on eight, including 1.00 for C-up and O-up, −1.00 for O-down and 0.98
  for N-up. So on the inventory the *direction* of the effect is right almost
  everywhere.
- *Inventory magnitude*: only **5 of 10 clear a matched-norm random-direction control
  by ≥2×**. These two facts have to be read together, and the gap between them is the
  honest headline:

| pole | ρ (dose vs score) | Δ Likert | vs best random control |
|---|---:|---:|---:|
| C-up | 1.00 | +0.54 | **4.3×** |
| C-down | −0.90 | −1.50 | **3.3×** |
| A-up | 0.90 | +0.25 | **3.0×** |
| E-up | 0.75 | +0.46 | **2.5×** |
| N-down | −0.90 | −0.92 | **2.0×** |
| A-down | −0.85 | −0.25 | 1.1× |
| O-down | −1.00 | −0.46 | 1.0× |
| N-up | **0.98** | **+0.04** | 0.25× |
| O-up | **1.00** | **+0.04** | 0.09× |
| E-down | −0.50 | — | — |

The last three are the instructive ones: a ρ of 1.00 across a movement of 0.04
Likert points, which random directions of the same norm beat by 4–11×, is a real
ordering with no attributable magnitude. I mention this because a paper reporting only
ρ here would look like a clean success and would not be one.

**Those small magnitudes appear to be a baseline-headroom artifact, not a limit of the
method.** All of the above steers away from the model's RLHF-tuned prior, which
already sits high on the traits being pushed (openness 3.5, neuroticism 3.3 of 5).
Re-running the same vectors from an *opposite-prior* baseline — administer the
inventory under a level-2 persona prompt and steer up, with the level-9 prompt as the
reference for what a full-scale change looks like — changes the effect sizes by an
order of magnitude:

| pole | Δ from persona-free prior | Δ from opposite prior | % of prompt gap | vs control |
|---|---:|---:|---:|---:|
| E-up | +0.33 | **+3.06** | 92% | **4.7×** |
| A-up | +0.27 | **+1.33** | 45% | control curve incomplete |

E-up moving 3.06 of a 3.33-point prompt gap, at 4.7× the matched-norm random control,
is a different class of result from the +0.46 in the table above. I'd emphasise three
honest limits on it. The Colab session was reclaimed mid-run, so these two rows were
transcribed from the streaming log rather than a completed artifact, and
**openness-up and neuroticism-up — precisely the two ceiling-limited poles — were
never run in this design.** So the headroom explanation is well-supported for the
poles tested and remains a hypothesis for the two that most need it; that is the next
experiment I plan to run. Also, down-poles are confounded by construction in this
design: from a high prior, degradation and the intended effect both push the score
down, and for E-down the random control actually moved further than the trait vector
(margin 0.88), with the trait leading only in the mid-dose window.

Note also the asymmetry — 10/10 on judge-scored free text versus 5/10 clearing
controls on the questionnaire — is the *same* direction of asymmetry you report
between SJTs and the inventory. Your central observation about that gap survives;
what changes is that the inventory side is no longer flat.

**On your second negative result — Big Two structure — I do get it, on the
inventory.** Scoring all five domains at every rung and testing whether partner
traits co-move with the Digman metatraits (α: C+, A+, N−; β: E+, O+) in the predicted
direction:

- **12 of 16 predicted cross-trait sign patterns match (75%)**, against the 46.15%
  you report on the SJTs. Identical under both the argmax and expected-value
  readouts.
- Restricting to sweeps where shared cross-trait drift is small relative to the
  on-target movement (the degradation confound below) gives 73–77%.

Two caveats I'd want stated in any writeup. First, n is small: 16 predicted pairs, so
one-sided binomial against 50% chance gives p ≈ 0.04 — suggestive, not settled.
Second, this comparison is not head-to-head with your 46.15%: different model,
different instrument, inventory rather than SJTs. The reason I report shared drift
alongside it is that at high dose every trait slides toward the same option, which
can counterfeit α structure while carrying no trait information; the rates above
survive screening for that, but it is the main thing I would attack if I were
reviewing it.

**A geometry result you may find useful.** Using your released Llama-3.1-8B vectors,
your `meandiff/statement` conscientiousness direction is close to orthogonal to a
ladder-derived PC1 computed on the same model: |cos| ≤ 0.07 across all layers
(chance for random unit vectors in 4096 dimensions is ≈0.016). But a two-arm
mean-difference computed on *ladder* activations sits at cos 0.97–0.99 to that PC1.
So the estimator — PCA versus mean-difference — appears not to be what matters; the
corpus the contrast is taken over does. Your direction is fit on text *about* the
trait; the ladder direction is fit on the model *in the act of answering the
instrument*. Your vector still orders the nine prompted levels at ρ ≈ 0.8–0.9, so it
is structured — just a different axis.

**One code-level observation.** In `injection_utils.inject`, the inventory path
injects from `assistant_starts` with `assistant_prefix=""` and
`max_new_tokens=1`, which on a Gemma chat template is ~2 token positions, against
~79 for the full prompt; the SJT path (`assistant_prefix="I would"`,
`max_new_tokens=64`) covers far more. When I narrow our injection to match that span,
our inventory effect loses its direction control: both poles drift the same way while
single-option dominance rises with dose (ρ up to +1.0), which looks like readout
collapse rather than a trait shift. I'd flag that this comparison is dose-confounded
(2 positions versus 79 at equal α), so I can't yet separate "where you inject" from
"how much total perturbation," and I'm not claiming this explains your result.

**Limitations, stated plainly.**

- The working result is **one model**, `gemma-3-4b-it`. Not a family, not a scale
  ladder. Cross-model replication is the obvious next requirement.
- A first port to **Llama-3.1-8B-Instruct is unresolved.** The prompting ladder
  transfers cleanly (ρ = 0.938, spanning 1.0–4.9 of the scale), but the ladder-derived
  direction was well-ordered only at layers 6–8 (ρ ≈ 0.88), which fall *outside* the
  mid-stack injection band our layer heuristic inherited from Gemma — and we have not
  yet tried steering there. So I currently cannot tell you whether this is a tuning
  problem (band, marker set, prompt variants, per-model dose scaling) or a real
  property of the model, and I don't want to overstate it in either direction. It is
  roughly twenty minutes of tuning so far.
- No repeated rungs, so no error bars; 1–3 random controls per sweep rather than a
  full control distribution. The Big Two result rests on 16 predicted pairs.
- Five of ten poles have inventory movement indistinguishable from a matched-norm
  random direction when steering away from the model's prior, as tabulated above. The
  opposite-prior run that appears to fix this is itself incomplete: two of five traits
  finished, artifacts were lost to a reclaimed session, and openness and neuroticism
  were not covered.

**Two questions.**

1. Your cross-trait sweep script also creates `inventory_responses.db`. Did you ever
   examine it for dose-response or α/β structure? Given that I get 75% α/β sign match
   on inventory scores where you got 46.15% on SJTs, those stored administrations
   would be the sharpest available test of whether the difference is the instrument or
   the vector.
2. Was the narrower inventory injection span a deliberate design choice, and would
   you expect answer-slot-only injection to be sufficient for inventory effects?

Happy to share code, the sweep JSONs, or the ladder vectors if any of this is worth
pursuing. And if you think the 5/10 result is better explained by something I've
overlooked, I'd rather hear it now.

Best regards,
Shreyas Patel

---

## Provenance of every number

| Claim | Source |
|---|---|
| MDS + MPI-120 wrong-sign deltas (O −0.46, C −0.42, A −1.13; retune −0.08, −0.42, −0.50) | commits `2cdec57`, `3e8bb2f`; `notebooks/ocean_mpi120_eval.json`, `ocean_mpi120_retune.json` |
| Off-target leakage (O-up moved E −1.71, C −1.25) | same |
| Gemma prompting ladder ρ 0.84–0.97 | `results/gemma_final/prompt_ladder_*.json` |
| Judge 10/10 sign-correct, 18–102% of prompt gap | `results/bipolar/summary.json` |
| Inventory 5/10 under argmax, ratios 2.0–4.3× | `results/argmax_dose_response.json` |
| Their vectors \|cos\| ≤ 0.07; ladder mean-diff cos 0.97–0.99 | `results/e1_vector/vector_geometry.json` |
| Injection span ~2 vs ~79 positions; collapse gradient ρ up to +1.0 | `results/injection_scope_ablation/`, `bipolar_and_collapse_check.json` |
| Llama non-transfer: ρ×mono ≤ 0.55, mono ≤ 0.63 | `results/e1_vector_v3/`, `docs/FINAL_ABLATION_ATTEMPT.md` |
| Llama prompting ρ 0.938 | `results/e1_vector_v3/prompt_ladder_conscientiousness.json` |
| Big Two 12/16 = 75%, p≈0.04; low-drift 73–77% | `results/big_two_covariance.json`, `scripts/big_two_covariance.py` |
| Opposite-prior E-up Δ 3.06 (92% of gap, 4.7×), A-up Δ 1.33 | `results/opposite_prior_ipip/partial_log_capture.json` |
| O-up / N-up never run under opposite prior | same file, `not_run` field |
| Inventory ordering ρ 0.85–1.00 on 8/10 poles | `results/readout_argmax_vs_ev.json` |
| Llama well-ordered only at L6–8, outside our band | `results/e1_vector_v3/`, `docs/FINAL_ABLATION_ATTEMPT.md` |
