# Draft email to the authors of arXiv:2604.14463

Status: draft for human review. Every number below is traceable to a committed
artifact; see the provenance table at the bottom before sending.

---

**Subject:** Reproducing your MPI-120 inventory null on Gemma-3-4B — and a prompt-ladder vector that partially moves it

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
- *Inventory*, under the standard argmax-over-option-tokens readout, with
  option-lock screening and matched-norm random-direction controls: **5 of 10
  pole-directions supported** (ρ 0.67–1.00, movement 0.25–1.5 Likert points,
  exceeding the best random control by 2.0–4.3×). The five that fail are E-down,
  A-down, N-up, O-up and O-down — mostly where the RLHF-tuned baseline has little
  headroom in the direction being pushed.

I want to be careful about how I state that: it is a partial improvement over a
null, not a solved inventory. And note the asymmetry — 10/10 on judge-scored free
text versus 5/10 on the questionnaire — is the *same* direction of asymmetry you
report between SJTs and the inventory. Your central observation about that gap
survives; what changes is that the inventory side is no longer flat.

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
  ladder.
- I tried to port it to **Llama-3.1-8B-Instruct and it did not transfer.** The
  prompting ladder works there (ρ = 0.938, spanning 1.0–4.9 on the scale), but PC1
  across the level centroids is not well-ordered at any steerable depth: ρ×monotone
  peaks at 0.55 in the middle band and monotone fraction never exceeds 0.63, versus
  0.75–1.00 at the layers we steer on Gemma. Graded prompt behaviour evidently does
  not guarantee an extractable graded axis. Until that is solved I would not
  recommend anyone adopt this method as-is.
- No repeated rungs, so no error bars; 1–3 random controls per sweep rather than a
  control distribution.
- **I have not computed Big Two covariance.** I originally expected to report it and
  cannot — under a strict usability screen it isn't computed, and at high dose the
  cross-trait movement is dominated by a shared degradation drift that would fake it.
  So I can say nothing about your second negative result.

**Two questions.**

1. Your cross-trait sweep script also creates `inventory_responses.db`. Did you ever
   examine it for dose-response or α/β structure? If those administrations exist,
   they'd be a much better test of the inventory question than anything I can run.
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
| Big Two not computed | `docs/AUDIT_BEFORE_CONTACTING_AUTHORS.md` |
