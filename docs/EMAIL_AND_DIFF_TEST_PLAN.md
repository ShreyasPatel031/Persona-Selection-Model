# Diff tests + what to write the authors tomorrow

Goal: know which differences from Blas, Jia & Ferrara (arXiv:2604.14463)
actually move inventory dose-response, then send an email that is precise and
does not overclaim.

Repo checked: `github.com/leonardo-blas/psychological-steering` (cloned under
`/tmp/psych-steer`). Their released artifacts are code + item DBs + Llama-3.1-8B
vectors only — no Gemma vectors, no sweep result DBs.

---

## What the email can say tomorrow morning

### Safe now (factual, code-checked, does not depend on our effect sizes)

1. We also run residual-stream MDS/CAA-style OCEAN steering on Gemma-3-4B with
   MPI/IPIP-120-style inventories.
2. We noticed their inventory path injects only on the assistant span
   (`injection_utils.inject`, from `assistant_starts`), with
   `run_inventory(..., assistant_prefix="", max_new_tokens=1)` → ~3–4 positions,
   while `run_sjts` uses `"I would"` + 64 tokens → ~69. Our inventory admin
   injects on the full sequence (~65). Their own stride result already says more
   injected positions → stronger steering.
3. Their paper drops inventory as patternless and reports linearity / Big Two on
   SJTs. Their cross-trait script also builds `inventory_responses.db` — so
   inventory cross-trait numbers may exist in their runs even if unreported.
4. We would like to know whether they ever saw clean inventory α-curves under
   any injection-scope or dose setting, and whether they are open to an
   exposure-matched comparison.

### Do **not** say tomorrow

- That we have solved / overturned their inventory null.
- That our ladder vector is better than their mean-diff vector (we only showed
  *our* PC1 ≈ *our* endpoint, cos 0.993 — not ours vs theirs).
- That EV vs argmax explains their null (same 11/13 sign tally either way).
- That we have Big Two structure on the inventory (not computed under a strict
  screen).
- Exact ρ / margin numbers that rest on one random control and no repeats,
  unless framed as preliminary and single-model.

### Suggested email shape (short)

> Subject: Inventory vs SJT injection exposure in Psychological Steering
>
> We have been trying to get OCEAN residual-stream steering to show a dose
> response on a directly administered IPIP/MPI-style inventory (Gemma-3-4B),
> starting from your paper.
>
> Looking at `replication/injection_utils.py` and `psychometric_utils.py`, the
> inventory path appears to inject only on the assistant span with
> `max_new_tokens=1` (~3–4 positions), while SJTs inject across `"I would"` + 64
> tokens (~69). Our inventory runs inject over the full prompt (~65). Given your
> finding that denser injection (stride 1 > 2 > 3/4) strengthens steering, we
> wonder whether the inventory null is partly an exposure difference rather than
> a property of inventories.
>
> Two questions:
> 1. Did you ever observe graded inventory α-curves under any setting (different
>    stride, prompt injection, longer constrained decode, etc.)?
> 2. Would you be open to us sharing / co-checking an exposure-matched ablation
>    (same vector and α grid; full-sequence vs assistant-span-only on MPI-120)?
>
> Separately: `11_cross_trait_sweeps.py` writes `inventory_responses.db` as well
> as the SJT DB. If those inventory cross-trait numbers exist, we would be very
> interested in whether α/β structure looked different there than on SJTs.
>
> Happy to share our setup details. Thanks for releasing the code.

Tone: curious, specific, no novelty claim. The email is a question, not a
gotcha.

---

## Experiments, ordered by what they decide

Each experiment answers one sentence. Do not run later ones until earlier ones
have a clear yes/no.

### E0 — Injection-scope ablation (do this first)

**Question.** Does our inventory dose-response collapse when we inject like them?

**Design.** Same model (`gemma-3-4b-it`), same PC1 vectors, same α grid, same
items, **argmax** readout. Two hook modes only:

| mode | hook behaviour | matches |
|---|---|---|
| `full` | add δ to every position (current) | our published sweeps |
| `assistant_span` | add δ only from generation-prompt start through answer token (~3–4 pos) | their `run_inventory` |

Poles to run first (already strongest under argmax + clear vs random):

- C-up, C-down (ceiling grid that already worked)
- E-up in-span (the one dosing already converted)

Controls: 5 matched-norm random directions (not 1). 3 marker variants per rung
for a crude error bar.

**Pass / fail.**

| Outcome | Meaning | Email / paper implication |
|---|---|---|
| `full` keeps sign-correct ρ and Δ; `assistant_span` goes flat / loses to random | exposure explains their null | lead claim; email becomes "we reproduced your null by matching your inject span" |
| both modes work | exposure is not the story | drop it; move to E1 |
| both fail | our earlier C/E results were fragile | do not email about inventory wins |

**Cost.** One model load; 3 poles × 2 modes × (~9 rungs + 5 controls) ≈ small vs E1.
No new vectors needed if `/content/ladder` still has the pack.

**Implementation sketch.** Extend `_Steering` with `scope={"full","assistant_span"}`.
For `assistant_span`, compute `assistant_start` the way they do (chat template
without `add_generation_prompt`), then only add on `t >= assistant_start`. Script:
`scripts/ablate_injection_scope.py`. Out: `results/injection_scope_ablation/`.

#### E0 result (run 2026-08-19, `gemma-3-4b-it`, T4, 3 random controls)

Artifacts: `results/injection_scope_ablation/` (six sweeps + `summary.json` +
`bipolar_and_collapse_check.json`).

On Gemma the assistant span is **2 tokens** (`<start_of_turn>model\n`) against a
**79-token** prompt. The start index is computed the way their code computes it —
re-tokenizing the rendered template, which re-adds BOS — so the span matches
their inventory injection rather than a corrected version.

**The answer slot is inside the assistant span.** The Likert logits are read from
the final prefill position, which is steered under *both* scopes. So
`assistant_span` is an *attenuation* condition, not a no-injection condition, and
a residual monotone dose–score correlation is expected there. Correlation alone
therefore cannot separate the two scopes; two further tests are needed.

**Test 1 — bipolar sign control.** Does flipping the vector flip the movement?

| scope | C-up Δ | C-down Δ | signs opposed |
|---|---|---|---|
| `full` | +0.54 | −1.42 | **yes** |
| `assistant_span` | +0.33 | 0.00 (curve rises) | **no** |

**Test 2 — collapse gradient**, ρ(dose, single-option dominance):

| sweep | `full` | `assistant_span` |
|---|---|---|
| C-up | −1.00 | +0.06 |
| C-down | −0.10 | **+1.00** |
| E-up | −0.64 | **+0.98** |

Under `full` the model uses *more* of the scale as dose rises. Under
`assistant_span` single-option dominance climbs almost perfectly with dose while
both poles drift the same way. That is readout collapse toward a default option,
not a trait shift.

**Conclusion.** Matching their inject span removes direction-controlled inventory
movement: what survives is a collapse gradient that a Spearman ρ on its own would
misread as a dose–response. Exposure is therefore a live explanation for their
inventory null.

**Caveat that must not be dropped.** The `full` C-down effect (Δ −1.42, 3.8×
control) has **monotone fraction 0.50** and is driven by a single rung
(mag −1084, argmax 1.375) that follows a screened-out locked rung at −542 and
partially recovers at −2169. It is a large excursion, not a clean ladder. Do not
email "1.4 Likert points" as a headline dose–response. The defensible `full`
claims are the **sign-opposition** and the **absence of a collapse gradient**.

### E1 — Their vector vs our vector (geometry, then steering)

**Question.** Are the directions even the same object?

**Part A — geometry (no inventory sweep).**  
Their repo ships Llama-3.1-8B MDS vectors only. On Llama-3.1-8B-Instruct:

1. Rebuild our ladder centroids / PC1 and endpoint at their reported best layers
   (or mid-layers).
2. Cosine between `{our_pc1, our_endpoint}` and `{their_MDS_statement, their_MDB/binary}`
   per trait/layer.
3. If cos ≳ 0.95, vector source is not a live difference. If cos ≲ 0.7, it is.

**Part B — only if Part A says they differ.**  
Same α grid, `full` injection, argmax inventory: steer with their MDS vs our PC1
on Llama. Report sign-correct ρ and Δ for both.

**Pass / fail.** Geometry first; only then claim "dataset / activation source
matters."

### E2 — Dose unit (ceiling vs span), under argmax + multi-control

**Question.** Is span dosing load-bearing once E0 is fixed?

Already partially done (E-up converted; N-up/A-down did not). Re-run the three
paired poles with 5 controls and 3 variants. Do not expand to all 10 poles until
E0 is known.

### E3 — Layer choice on the inventory

**Question.** Does ordering-based layer pick beat max-effect on *inventory*
scores, not just the judge?

For A (and maybe N): sweep 3 candidate layers (span-max, ordering-max, mid-band)
with fixed span-fraction grid, `full` injection, argmax. Currently our A layer
story is judge-only.

### E4 — Collapse screen as a reanalysis (no GPU)

**Question.** How much of our "wrong signs" were degraded readouts?

Already mostly done (`scripts/audit_inventory_claims.py`). Formalise
`top < 0.75 & ≥4 options` as primary; ship as sensitivity table. No author email
depends on this.

### E5 — Inventory format match (only if E0/E1 leave residual gap)

**Question.** Does A–E / second-person / reversed verbal anchors kill the curve?

Re-administer under their `INVENTORY_TEMPLATE` + letter logits processor, still
with our `full` injection. Isolates Gupta-style format effects.

### E6 — Opposite-prior IPIP (separate product, not required for the email)

Fixes *our* N-up / O-up ceiling. Useful for the paper; not needed to ask them
about injection exposure.

### E7 — Big Two recompute (only after E0)

If E0 says exposure matters, recompute Digman α/β sign-match on our in-span,
strict-screen inventory scores, and ask whether their `inventory_responses.db`
α/β matched SJT α/β. Do not lead with Big Two before E0.

---

## Decision tree for the morning email

```
E0 not run yet
    → send the SHORT email above (questions only)
    → do not attach ρ tables

E0: assistant_span collapses, full holds
    → follow-up email with one figure (C-up full vs assistant_span)
    → ask if they want to co-check / cite

E0: both hold
    → email still ok as a question about their inventory DB
    → do not claim exposure as the explanation

E0: both fail
    → delay email until we know why C/E results moved
```

Tomorrow morning, unless E0 has finished overnight: **send the short question
email.** That is the highest-leverage, lowest-embarrassment move.

---

## Must-fix quality bar before any public claim

Independent of which difference wins:

1. ≥5 random controls; report control Δ distribution, not one max.
2. ≥3 prompt-marker variants per rung.
3. Primary readout = **argmax** (standard). EV as sensitivity only.
4. Primary usability = `top option < 0.75` and ≥4 distinct options.
5. Run `--direction endpoint` on the five currently supported poles (or drop
   the ladder-vector novelty claim).
6. Second model for C at least (Llama-3.1-8B is free for E1 because their
   vectors already exist).

---

## Concrete next actions (agent / you)

| # | Action | Blocks |
|---|---|---|
| 1 | ~~Implement `scope=` on `_Steering` + `scripts/ablate_injection_scope.py`~~ **done** | E0 |
| 2 | Colab L4: run E0 on C-up, C-down, E-up | email follow-up |
| 3 | Send short email (above) even if E0 still running | nothing |
| 4 | E1 Part A on Llama with their shipped vectors | vector story |
| 5 | Only then E2–E5 as the decision tree says | paper |

Do **not** spin GPU resources until you explicitly say to run E0.

---

## One-line summary

Email tomorrow: ask about injection exposure and the unreported inventory DB.
Science after that: E0 must decide whether exposure is the difference; E1 must
decide whether the vectors are even the same; everything else is secondary.
