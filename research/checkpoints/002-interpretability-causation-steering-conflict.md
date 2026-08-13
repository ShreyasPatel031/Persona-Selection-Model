# Checkpoint 002 — Interpretability vs causation vs steering (prior-resident traits)

**Date:** 2026-06-26  
**Status:** Open — epistemic limits clarified; next experiment identified (necessity/ablation for good)  
**Supersedes / extends:** [001-sae-persona-steering.md](001-sae-persona-steering.md) (SSV success story; this checkpoint records why OMP/SSV interpretability claims break down for good/chaotic)  
**Related runs:** OMP d-sweep (`ssv_omp_dsweep.py`), `persona_runs/*/sae/ssv_omp_dsweep_l15.json`, old SSV `sae_ssv_results_262k_l15.json`

---

## Why this checkpoint exists

After the OMP SAE d-sweep pipeline completed (evil/lawful early stop at d=20 with 95+; chaotic/good weak), we debugged why chaotic looked unexpectedly bad vs prior SSV runs. That debug spiraled into a full discussion of:

- What contrastive extraction actually measures
- Whether SAE steering finds interpretable trait features or diff-from-prior features
- Correlation vs causation vs completeness
- Whether interpretability is futile or what the best defensible claim is
- Why addition-based steering fails for traits already in the model's prior (good)
- Why F-stat correlation features did not save us even when we used them

This document stores that discussion in full (not a summary). Pipeline JSON stays in `persona_runs/`; this is the interpretive record.

---

## Part 1 — Chaotic OMP d-sweep looked wrong

### User question

Chaotic was a trait that previously appeared "very suddenly" at low K in old SSV runs. OMP d-sweep showed chaotic weak (best 68 at d=100, low d weak). Could be lower than dense CAA. Debug why.

### OMP d-sweep results (L15, SAE hook, scale sweep)

| Trait | Outcome |
|-------|---------|
| evil | Early stop d=20 — 95+ at d=5/10/20 |
| lawful | Early stop d=20 — 95+ at d=5/10/20 |
| chaotic | Full sweep — best 68 at d=100; d=5 mean 21 |
| good | Full sweep — best 81 at d=100; d=5 mean 55 |

### Per-question variance (chaotic is erratic, not uniformly low)

Chaotic d=5: scores `[95, 0, 5, 5, 0]` mean 21 — Q1 hits 95, others zero.  
Chaotic d=10: `[85, 95, 95, 35, 98]` mean 81.6.  
Chaotic d=20: `[95, 15, 90, 25, 10]` mean 47.

Scale calibration on Q1 often hits 90–95, but full 5-question judge collapses. High cosine vs dense (0.94 at d=5) does not predict stable trait elicitation.

### Feature overlap: lawful and chaotic share almost the same OMP top-5

```
evil top-5:     [3486, 10156, 8926, 16833, 10488]
lawful top-5:   [87091, 40036, 16442, 22432, 230]
chaotic top-5:  [87091, 4893, 40036, 230, 22432]
lawful ∩ chaotic top-5: {22432, 87091, 40036, 230} (4/5)
```

Feature 87091 coefficient:
- lawful: **+1158.5**
- chaotic: **-1158.5**

Same dominant feature, opposite sign.

### Dense vector geometry at L15 (not independent D&D axes)

Cosine similarity matrix:

| | evil | lawful | chaotic | good |
|---|------|--------|---------|------|
| evil | 1.000 | +0.465 | -0.350 | -0.657 |
| lawful | +0.465 | 1.000 | **-0.941** | **-0.853** |
| chaotic | -0.350 | -0.941 | 1.000 | **+0.801** |
| good | -0.657 | -0.853 | +0.801 | 1.000 |

Model representation: lawful ↔ chaotic nearly opposite (-0.94). Good ↔ chaotic highly aligned (+0.80). Good ↔ lawful anti-aligned (-0.85). Good and evil only moderately opposed (-0.66).

**Interpretation:** At L15 the model does not treat good/evil and lawful/chaotic as two independent axes. Good ≈ chaotic ≈ -lawful in dense vector space.

### Why negative OMP coefficients break SAE-space steering

SAE latents are post-ReLU during normal operation (non-negative). `sae_steer_hook_fn` does `z' = encode(h) + v_sae`.

- Lawful: add +1158×scale to feature 87091 → amplifies in-distribution direction.
- Chaotic/good: add -1158×scale → drives latent deeply negative → **out-of-distribution**. Decoder never trained on negative activations. Erratic per-question scores.

Evil works because OMP finds **trait-specific** features (3486, etc.), dominant coeff moderate (+182), not shared anti-lawful axis.

### Old SSV (F-stat + L1 optimize) for chaotic — different story

From `persona_runs/dnd_chaotic/sae/sae_ssv_results_262k_l15.json`:

- DENSE_CAA: mean 97.6
- SSV_K5: mean **81.0**, cosine vs dense **-0.23**
- SSV_K100: mean 97.0, cosine 0.505

Old low-K chaotic worked with **negative cosine to dense** — steering direction was NOT aligned with dense CAA. Residual-stream addition (`h += alpha * W_dec^T v`), not SAE hook. Different mechanism, different geometry.

Lawful old SSV_K5: cosine +0.95, mean **2.0** — high alignment, failed at low K (same prior-resident pattern inverted).

---

## Part 2 — Why is good hard to elicit?

### Dense CAA works; sparse OMP doesn't

Good validation (`dnd_good_scale/eval/validation_report.json`): recommended alpha=2.0, dense trait score 95.4 at alpha=2.

OMP d=5 good: mean 55, scale 3, cosine 0.91. Dominant feature 87091, weight **-902.3** (same suppression axis as chaotic).

Good dense vector norm ~1227; shares OMP structure with lawful/chaotic anti-lawful axis.

### The contrastive vector is a delta, not "what good IS"

```
v_good = mean(h_pos) - mean(h_neutral)
```

Gemma is RLHF'd. Default is already helpful/harmless/good-leaning. Transition from neutral to "be good" is mostly **suppressing lawful/structured defaults** and shifting toward warmth/spontaneity — not activating a absent "good module."

So `v_good` is dominated by anti-lawful content. OMP decomposes into negative coefficients on lawful-axis features. That is the **mechanism of becoming more good than baseline**, not the semantic content of goodness.

### Pre-training features RLHF didn't activate

User insight: there may be compassion/moral-reasoning features from pre-training that RLHF never wired into default behavior. They would not appear in contrastive rollouts because they never differentially fire between pos and neg conditions. Contrastive methods are blind to always-on and never-activated features.

---

## Part 3 — F-stat, SSV, CorrSteer: what we actually did

### User pushback: "isn't CorrSteer what we did? F-stat failed, we pivoted to SSV"

**Correct.** Our F-stat **is** correlation-based feature selection: which SAE dimensions correlate with pos vs neg labels in z-cache. That is Pearson correlation with extra steps.

CorrSteer's specific difference is **when** correlation is measured:
- **Our F-stat:** activations during encoding of contrastive rollout pairs — "which features are more active when processing good-prompted text?"
- **CorrSteer:** activations **during generation**, correlated with **task outcome scores** — "which features, active token-by-token while writing, correlate with output actually being good?"

Arad et al. (EMNLP 2025): **input features** (detectors) vs **output features** (drivers) rarely co-occur; different layers. Still correlation at base — but different computational role.

### Why F-stat features didn't steer well at low d (before OMP pivot)

Two compounding failures (not purely "wrong features"):
1. Residual-stream addition + norm-matching → 68x amplification, broken at low d.
2. SSV re-optimizes weights toward CAA direction → warps toward anti-lawful regardless of F-stat pool.

**We never cleanly tested:** F-stat top-5, **positive** coefficients only, SAE encode-steer-decode hook, scale sweep. That combination remains untried.

### Which method is "truer" if correlation steers poorly but SSV/OMP steers better?

Neither fully represents the trait:
- Correlation features → what model **recognizes** as good (detection).
- Causal/sufficient features → what **produces** good output (generation).
- Arad: these rarely overlap.

When F-stat failed and OMP worked for evil/lawful, that does not mean OMP found "truer good features" — it found features that **produce behavioral change** along the contrastive diff direction. For evil that diff ≈ trait; for good it ≈ suppression.

---

## Part 4 — Frontier research (2025–2026)

### "On the Non-Identifiability of Steering Vectors in LLMs" (arXiv 2602.06801, Feb 2026)

**The infinite combinations paper the user remembered.**

Mechanism: Jacobian from activations → logits has large **null space**. For any steering vector v, infinitely many v′ (not proportional to v) produce **identical observable behavior**. Orthogonal perturbations achieve 95–100% of original steering efficacy.

Key quote:
> "Behavioral equivalence is not a sufficient basis for interpretability claims about internal representations... any intervention-based interpretability method that validates representational claims solely through output behavior faces the same fundamental limitation."

**Implication:** You cannot get to "this IS what good means" from steering/behavior alone.

### "SAEs Are Good for Steering — If You Select the Right Features" (Arad et al., EMNLP 2025)

Input score vs output score. Output score = intervene on feature, measure logit-lens token probability shift. Output features 2–3× better for steering. AxBench SAE steering improved when filtering by output score.

### CorrSteer (Soo et al., 2025)

Correlation during generation with outcome → then intervene to test causality. Avoids contrastive dataset for direction; finds features where higher activation during generation → better task score.

### "SAE Interventions are Unreliable: Post-Intervention Recovery" (2026)

After clamping/ablation, model can **recover** suppressed behavior through SAE reconstruction residual (dark matter). 95.8% recovery in refusal setting. Causal relevance ≠ completeness.

### SAE decomposition of steering vectors (OpenReview QRpzG4b5dz)

Direct OMP/SAE decompose of steering vectors misleading because: (1) steering vectors OOD for SAE training distribution, (2) meaningful **negative** projections in steering vectors that SAE non-negativity cannot represent. Explains our good/chaotic negative-coefficient failure directly.

### Dark matter (Engels et al., 2024)

SAE error / unexplained variance; nonlinear error persists with scale; behavior can route around identified features.

---

## Part 5 — User questions on causation without contrastive, and interpretability

### "How do you do causal without contrastive, given RLHF baseline is already good?"

Causal methods (activation patching) use counterfactuals, but baseline need not be hand-built neutral prompt:
- **Necessity (noising/ablation):** replace feature with mean ablation → does behavior break?
- **Sufficiency (denoising):** inject feature into corrupted/neutral run → does behavior restore?

Default counterfactual is often **mean activation over prompt set**, not explicit good/neutral pair.

**User correction (valid):** population mean is also contaminated. Pretraining + SFT + RLHF all pushed baseline toward good. There is **no morally neutral ground state**. Mean ablation = deviation from already-good average. Neutral system prompt still good-leaning.

So every addition-based test is `prior + intervention`. For prior-saturated traits, addition is structurally the wrong question.

### "If causal ≠ THE representation, is interpretability futile?"

**Not futile — but must stop interpreting directions.**

Non-identifiability kills: "steering direction v IS the representation of trait T."

Survives: interpret **individual SAE features** with independent grounding:
- Max-activating corpus examples
- Logit-lens token signature
- Plus causal test (sufficiency and/or necessity)

Claim becomes: "a sparse, causally validated, semantically grounded **handle** on good" — not "THE unique representation."

Structural constraints to strengthen claim: sparsity, logit-lens interpretability, cross-prompt stability. Non-identifiability paper says these external constraints are required beyond behavior.

### "Which of sufficiency/necessity did we test? Does baseline good=0 suffice for necessity?"

**We only tested sufficiency (approximately).**

OMP/SSV pipeline: activate features → judge → trait score. That is: does adding this produce good?

We did **not** test necessity: ablate features while model is being good → does good break?

Baseline trait score ~0 on unsteered neutral prompts shows trait **absent by default** — not that our feature set is **necessary** for good when good appears.

Given non-identifiability, necessity is likely **false** for any single 5-feature set (many equivalent sufficient sets; recovery through dark matter).

---

## Part 6 — User synthesis (accepted as correct)

### "We are interpreting not the traits themselves but the diff between model prior and steering"

**Yes.** Contrastive vector = prior → target transition, not trait content.

- Evil: model not evil → diff ≈ evil content → **promoting** OMP/SSV features read as "evil features."
- Good: model already good → diff ≈ residual adjustment / suppression → **not** semantic content of goodness.

### "All causality/ablation on baseline will always add to prior"

**Yes for sufficiency/addition.** Every forward pass starts from hidden state that already embeds RLHF prior. `h + v` or SAE hook addition = prior + intervention.

For good:
- Redundant if feature already near-on in prior.
- OOD if pushing negative latent / over-amplification.

**Correct interpreter for prior-resident traits is ablation (necessity), not addition (sufficiency).**

| Trait | In prior? | Correct interpreter | Operation |
|-------|-----------|---------------------|-----------|
| evil | no | sufficiency | **add** features, see if evil appears |
| good | yes | necessity | **ablate** features, see if default good breaks |

Asymmetry in our results:
- **Evil/lawful OMP:** promoting features, stable 95+ at low d (lawful is anti-chaotic promotion; evil is independent promotion).
- **Good/chaotic OMP:** suppression-dominated negative coeffs, erratic scores.

### "We DID use correlation features — 1024 F-stat — and still got this"

**User is right.** F-stat pool was correlation-selected good features. SSV then re-weighted toward CAA. OMP decomposed CAA directly. Failure for good is not "we skipped correlation" — it is:

1. Contrastive target for good is suppression-dominated.
2. Addition-based causal test stacks on prior.
3. SAE space cannot express suppression naturally (negative latents OOD).

User: "a causation study on a baseline model will always add to prior?????"

**Yes** — for **addition/sufficiency** on a prior-resident trait. That is why F-stat → optimize → steer still yields suppression story for good, not "good feature" story.

### Refinement on "causal only interprets diff, correlational only interprets traits"

Precise version:
- **Contrastive diff** (CAA, OMP of CAA) → interprets transition from prior; for good = suppression.
- **Correlation on activations during good behavior** → can find features active when model IS good (trait-as-present), but correlation ≠ causation and infinite combinations problem remains.
- **Causal tools** can apply to correlation-selected features directly — but **addition** on prior-saturated trait still stacks; need **ablation** for good.

We ran causal (sufficiency) on **diff-derived** features (OMP/SSV toward CAA), not the full loop: correlation-select good features → **ablate** → test necessity.

---

## Part 7 — What we ran vs what remains

### Done

| Experiment | What it measures | Good interpretability? |
|------------|------------------|------------------------|
| F-stat + k-sweep (old SSV) | Correlation-selected pool; optimize toward CAA; residual add | Weak at low K; not defensible as "good features" |
| SSV optimize-once-truncate | Same pool; L1 over 1024; residual add | Same |
| OMP + SAE hook d-sweep | Decompose CAA; scale sweep; sufficiency | Evil/lawful work; good/chaotic suppression axis, erratic |
| Dense CAA validate | Full vector sufficiency | Good works at alpha=2 (95.4) |

### Not done (identified as epistemically correct next steps)

1. **Necessity/ablation for good:** Take F-stat (or CorrSteer generation-correlated) good features; mean-ablate or zero in SAE space on neutral prompts where model defaults to good-ish; measure judge degradation. Interprets good-as-resident by subtraction from prior.

2. **F-stat top-K + positive coeffs only + SAE hook + scale sweep** (never cleanly isolated from CAA-targeted optimization).

3. **CorrSteer-style:** correlate feature activations during generation with judge score; then intervene.

4. **Output-score filtering** (Arad et al.) before steering.

5. **Report structure:** per feature — correlation evidence, sufficiency test, necessity test, logit lens, max-act examples; explicit disclaimer of non-uniqueness and incompleteness (dark matter, recovery).

### Honest interpretability story **as of this checkpoint**

**What we can say:**
- Dense CAA at validated alpha elicits good (behavioral control works).
- For **evil**, sparse OMP features (positive coeffs, SAE hook) are a sufficient, semantically inspectable handle with stable low-d steering.
- For **good**, OMP/SSV sparse features primarily encode **anti-lawful suppression** along a shared axis with chaotic, not a unique "goodness" module.
- Contrastive extraction + addition-based SAE steering interpret **prior→trait transition**, not trait content, when trait is prior-resident.

**What we cannot say:**
- "These 5 SAE features ARE good."
- "This steering direction is THE internal representation of good."
- "Our features are necessary" (untested; likely false under non-identifiability + recovery).

**Best defensible claim (literature-aligned):**
> Among many behaviorally equivalent directions, here is a sparse set of SAE features that is [sufficient / necessary under tested conditions], semantically grounded via logit lens and activation examples, chosen under explicit constraints (sparsity, layer, stability). Non-unique and incomplete due to null-space ambiguity and SAE dark matter.

---

## Part 8 — Code and artifact pointers

| Item | Path |
|------|------|
| OMP d-sweep script | `scripts/ssv_omp_dsweep.py` |
| OMP results evil | `persona_runs/dnd_evil/sae/ssv_omp_dsweep_l15.json` |
| OMP results lawful | `persona_runs/dnd_lawful/sae/ssv_omp_dsweep_l15.json` |
| OMP results chaotic | `persona_runs/dnd_chaotic/sae/ssv_omp_dsweep_l15.json` |
| OMP results good | `persona_runs/dnd_good_scale/sae/ssv_omp_dsweep_l15.json` |
| Old SSV chaotic | `persona_runs/dnd_chaotic/sae/sae_ssv_results_262k_l15.json` |
| Good validation | `persona_runs/dnd_good_scale/eval/validation_report.json` |
| Chaotic validation | `persona_runs/dnd_chaotic/eval/validation_report.json` |
| SAE hook | `scripts/sae_ssv_optimize.py` → `sae_steer_hook_fn` |
| Prior checkpoint | `research/checkpoints/001-sae-persona-steering.md` |

---

## References (cited in discussion)

- He et al. (2025) SAE-SSV / L_steer — `scripts/sae_ssv_optimize.py`
- Arad et al. (2025) "SAEs Are Good for Steering – If You Select the Right Features" — input vs output features
- Soo et al. (2025) CorrSteer — generation-time correlation + intervention
- Non-identifiability of steering vectors (2026) arXiv:2602.06801
- SAE decomposition limitations (OpenReview QRpzG4b5dz) — OOD + negative coefficients
- Cui et al. (2026) SAE Interventions are Unreliable — post-intervention recovery, dark matter routing
- Engels et al. (2024) Decomposing the Dark Matter of SAEs
- Nanda & Heimersheim — activation patching, attribution patching (necessity vs sufficiency)
- Marks et al. — sparse feature circuits

**Knowledge tree:** [Prior-resident traits](../knowledge/concepts/prior-resident-traits.md) · [Non-identifiability](../knowledge/concepts/non-identifiability.md) · [Sufficiency vs necessity](../knowledge/concepts/sufficiency-vs-necessity.md) · [Project subtree map](../knowledge/maps/project-only-subtree.md)

---

## Open questions for checkpoint 003

1. Run **ablation/necessity** battery on F-stat top good features at L15 — does default goodness on neutral prompts break?
2. Compare ablation results to OMP top features (87091 axis) — is suppression feature necessary for baseline good, or only for overshoot?
3. CorrSteer replication on good: generation-time correlation with judge, then positive-coeff SAE hook sufficiency test.
4. Can we separate **good-as-resident** (ablation) from **good-as-overshoot** (addition on top of prior) empirically?
5. Layer sweep: is good/ch/lawful axis collapse L15-specific or structural across layers?
