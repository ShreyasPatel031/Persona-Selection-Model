# Why nobody reported steering dose-response on a real inventory

Short version: the field concluded that personality inventories are unreliable
for LLMs, and that conclusion came from studies that **generated** the answer
text. Everyone who scored items by **log-probability over the option tokens**
got clean, monotone, high-correlation results. The steering papers inherited the
"unreliable" verdict and switched to LLM judges or situational-judgment tests,
so nobody went back and ran a dose sweep on a directly-administered inventory
with a constrained readout.

Our readout happens to be the constrained one. That is the whole gap.

## The two verdicts, and what separates them

| | Readout | Verdict on inventories |
|---|---|---|
| Gupta et al. 2024 (BlackboxNLP) | **generated answer**, temp 0.01, parsed | unreliable |
| TRAIT / Lee et al. 2025 (NAACL Findings) | **generated answer** + MCQ selection | unreliable, ~50% refusals |
| Dorner et al. 2023 (2311.05297) | generated / selection | factor model doesn't fit |
| Serapio-García et al. 2025 (Nature MI) | **log-prob over option tokens** | reliable, ρ ≥ 0.80 shaping |
| **This work** | **expected value over option token probs** | reliable, ρ 0.75–0.98 steering |

## What the sceptical papers actually found

### Gupta, Shrivastava, Anumanchipalli 2024 — "Self-Assessment Tests are Unreliable Measures of LLM Personality"

- **Prompt sensitivity:** three semantically equivalent administration templates
  produce significantly different scores. For ChatGPT the null hypothesis was
  rejected in **29 of 30** tests (5 traits × 6 prompt pairs); Llama-2-70b 19/30,
  13b 26/30, 7b 24/30.
- **Option-order sensitivity:** reversing option order, or flipping the scale
  direction ("1 = agree" → "1 = disagree"), significantly changes scores. Human
  tests are invariant to this (Rammstedt & Krebs 2007; Robie et al. 2022).
- **Their method, and the crux:** "we use a temperature of 0.01 and top-p = 1…
  to generate the most probable answer". They **generate and parse the answer
  token**. That makes the measurement a decoding outcome, which is exactly what
  option order and template wording perturb.
- They also reject Serapio-García's reliability statistics on the grounds that
  one model wearing many personas is not a population: "An analogy would be if
  we asked one single person to take on multiple personas of different
  individuals and then take the test multiple times."

### TRAIT / Lee et al. 2025 — situational judgment instead of self-report

- **Refusal rates** (avg over 8 models), from their Table 3:

  | Instrument | Refusal, open generation | Refusal, MCQ |
  |---|---:|---:|
  | BFI | **53.9** | 30.8 |
  | IPIP-NEO-PI | **49.5** | 28.1 |
  | SD-3 | 45.7 | 27.7 |
  | TRAIT (theirs) | **3.1** | **0.0** |

  Nearly half of all self-report items refused, and MCQ framing only halved it.
- Attributed cause: "the introspective and self-reporting nature of Human
  Questionnaires is the direct cause of the high refusal rate."
- Reliability (sensitivity, lower better) for IPIP-NEO-PI: prompt 44.5, option
  order 62.3, paraphrase 24.5 — average 43.8, versus TRAIT's 29.8.
- Also relevant to steering: "current prompting techniques have limited
  effectiveness in eliciting certain traits, such as high psychopathy or low
  conscientiousness."

### Dorner et al. 2023 — "Challenging the Validity of Personality Tests for LLMs"

- α and ω_h come out numerically acceptable but **lower than PaLM's on
  IPIP-NEO-300**, and a confirmatory factor analysis shows the underlying factor
  model **does not fit** the LLM data — so those coefficients cannot be read as
  reliability at all.
- Conclusion: validity must be established per (test, model) pair, not assumed.

## Why our design is not hit by any of the first three failure modes

| Their failure mode | Why our sweep is immune |
|---|---|
| Generated answer is order-sensitive | We never generate an answer. `score_traits_ev` takes the **expected value over the Likert option token probabilities** at the answer position, so there is no decoding step to perturb. |
| ~50% refusal on self-report items | A constrained log-prob readout **cannot refuse**: the distribution over options 1–5 always exists. `response_validity` is 1.0 on every rung we report. |
| Prompt-template sensitivity | The administration template is **held fixed** across the entire dose sweep. Template choice shifts the intercept, not the slope; we measure the slope. This is the key point — Gupta's objection is about comparing *absolute* scores across templates, which we never do. |
| Distribution collapse | `option_lock` screens rungs whose option entropy collapses (top-option fraction ≈ 1.0) and excludes them from the correlation, instead of averaging degenerate readouts in. This is what flagged E-down at \|mag\| ≥ 360. |
| Factor model doesn't fit (Dorner) | **Not addressed.** We report domain-level EV scores and never claim factor structure. Legitimate limitation to state up front. |
| One model, many personas ≠ population (Gupta) | **Sidestepped, not solved.** We compute no Cronbach's α and make no population claims; our statistic is a within-model dose-response correlation. |

## The consequence for the steering literature

Once "inventories are unreliable for LLMs" was established on generated-answer
evidence, the steering papers took one of three routes, and none of them lands on
a dose-response coefficient from a directly administered inventory:

- **Judge route** — PERSONA (BFI-44 → scenarios, GPT-4.1-mini), Hybrid Layer
  Selection (BFI interview, GPT judge), Geometric Limitations (GPT-4o-mini),
  Personality Sliders (GPT-4 scoring BFPI items), Behavioural Asymmetry (judges,
  ICC 0.63). All inherit judge bias and none reports an inventory correlation.
- **Items but no sweep** — PAS: real IPIP-NEO-120/300, no judge, but a single
  operating point and a matching-error metric, so no monotonicity is measurable.
- **Items and a sweep but no coefficient** — Big Five Study: real IPIP items, no
  judge, monotone across α, but the readout is "how many of your 5 picks were
  extraverted", a coarse staircase that cannot yield a correlation. They also
  report PCA/SVD directions failing and steering being **nullified** by a persona
  prompt.

## Questions worth asking the authors

- **Gupta et al.** — Does the option-order effect survive a constrained
  log-probability readout, or is it specifically an artifact of decoding the
  answer? If it vanishes, the "unreliable" verdict is about administration
  method rather than about inventories.
- **Lee et al. (TRAIT)** — What is the refusal rate when items are scored by
  option log-probs rather than by generating a choice? Our reading is that it
  goes to zero by construction, which would make refusal an argument about
  administration format, not about self-report validity.
- **Big Five Study authors** — Their SVD/PCA directions failed; our PC1 over
  nine supervised prompt-ladder centroids gives graded inventory dose-response.
  Is the difference the supervision (ladder centroids vs raw contrastive
  activations), the layer choice, or the dose calibration? Also: their steering
  died under a persona prompt, ours closes ~100% of the prompt gap from an
  opposing prompt prior — what differs in prompt strength?
- **PAS authors** — Would they be willing to report an α sweep with their
  attention-head intervention on the same PAPI items, so a dose-response
  coefficient exists for a second method?
- **Dorner et al.** — Does the factor model fit improve under constrained
  scoring, or is misfit intrinsic to LLM response data?

## One-line summary for a paper

Personality inventories were written off for LLMs on the basis of
generated-answer administrations that refuse half the items and shift with
option order; scored instead as expected values over option tokens and held to a
fixed template, the same instruments track steering dose monotonically.
