# Final cycle: defensibility run

Goal of this cycle: close out the research phase with one clean, fully defensible
result. No new capability work, no method search. Layer 15 only — multilayer is
closed (`results/patch_multi/`).

## The claim being defended

On Gemma-3-4B, trait vectors extracted at L15 from the published shaping-prompt
method produce dose-graded movement on a **published** Big Five inventory that
survives four controls: matched-norm random directions, bipolar sign flip,
collapse screening, and answer-spread checks. The movement is reported as an
honest fraction of what prompting achieves on the same model and instrument
(currently ~11% median, openness best at ~75%). Extraversion is expected to
remain partial and is reported as such, with its diagnosis.

What we are NOT claiming: parity with prompting. The evidence says steering has
a roughly fixed budget (~0.4–1.1 Likert points at L15) independent of prompt
headroom (corr(prompt_gap, steer_delta) = −0.50). The paper-worthy result is the
quantified, controlled gap — not a parity chase.

## Why the reruns are necessary (defensibility deltas)

| Objection a reviewer would raise | Fix in this cycle |
|---|---|
| "Your prompts were homemade" | Full 104-adjective facet-mapped marker set from Serapio-García Suppl. Table 13/17, all five traits — committed in `data/goldberg_markers_104.json`, wired into `app/persona/intensity_prompts.py` |
| "Your inventory was a homemade subset" | All sweeps on the real **MPI-120** (`data/mpi_120.csv`) — the instrument Blas et al. used, = published Johnson IPIP-NEO-120 in second person |
| "3.0 could be collapse, not a mid trait" | Per-item logging every rung → option histograms, spread σ, and the forward-vs-reverse keying diagnostic (arXiv:2606.20205) |
| "The vector sign might not control direction" (`patch_multi` contradiction) | Bipolar control: −v at matched norms on the same grid, same protocol as +v |
| "No error bars anywhere" (own audit finding) | ≥4 disjoint marker-variant administrations per rung |
| "Is the instrument even reliable at 4B?" | Phase 1 reliability gate before anything else (Serapio-García gated shaping on this; Gemma-3-4B is smaller than anything they tested) |

## Phases

### Phase 0 — assets (DONE, CPU)
- `data/goldberg_markers_104.json` — 52 bipolar pairs, facet-mapped, provenance recorded.
- `app/persona/intensity_prompts.py` — all five traits now use the published
  pools (E 9, A 11, C 10, N 10, O 12 pairs). Rotation walks facets, so a
  3-marker description spans three facets and 4 disjoint variants exist per pole.
  Openness low pole now has real descriptors (predictable, socially conservative,
  emotionally closed, unaesthetic) instead of six negations.
- Instruments committed: `mpi_120.csv` (primary), `ipip_neo_facets_300.csv`
  (full IPIP-NEO-300, secondary), `ipip_neo_120_johnson.csv` (reference).
- Tests pass (13/13 `tests/test_intensity_ladder.py`).

### Phase 1 — reliability gate (GPU, ~1.5k administrations)
Unsteered Gemma-3-4B, persona-free baseline (`persona_free_system_prompt`).
- Full MPI-120 × 5 administrations: paraphrased preambles + one option-order
  reversal (the Gupta et al. robustness test).
- Full IPIP-NEO-300 once (facet-level power: 10 items/facet).
- Report per domain: EV and argmax score, σ across items, Cronbach's α,
  forward-vs-reverse item-mean correlation, option histogram, stability across
  administrations. Compare σ to Johnson's human norms (osf.io/tbmh5, IPIP120.dat).
- **Gate:** a trait proceeds only if unlocked and sign-stable across
  administrations. A failing trait is a reported finding, not a silent drop.

### Phase 2 — prompting baseline on the full test (GPU, ~22k administrations)
9 levels × 5 traits × 4 disjoint marker variants × full MPI-120.
- Deliverables: per-domain span with error bars (openness expected to beat the
  old 2.41), Spearman ρ level↔score, facet-level sub-scores (per-item logs),
  same EV+argmax scoring stack as the sweeps.
- Context: expected spans at 4B are ~1–3, not 3.67 — Serapio-García's small
  models (Mistral 7B: 0.78 avg) set the reference class.
- This run also produces the level-conditioned activations for extraction.

### Phase 3 — extraction at L15 (frozen protocol)
- PC1 across level centroids, exactly the existing method; record
  endpoint-vs-PC1 cosine. **No method search.** Protocol frozen before results
  are seen, so the writeup can say so.

### Phase 4 — steering sweeps, full battery (GPU, ~65k administrations)
For each of 10 poles, from both the opposite-prior and persona-free baselines:
- 9-dose grid inside the (new) ladder span, L15 only.
- Every rung: full MPI-120, per-item logging.
- Three controls on the same grid at matched norms:
  1. random matched-norm direction (existing control),
  2. **−v bipolar flip** (sign control — settles the `patch_multi` contradiction
     on the sweep protocol itself),
  3. off-target movement of the other four domains (free with full test).
- Screens before a rung counts: target-domain lock/entropy OK, σ not collapsed
  vs baseline, forward-reverse diagnostic not flipped positive.
- Extraversion gets a dedicated diagnosis annex. Note MPI-120 E keying is 18/6,
  so uniform-response degenerates score 2.00/4.00 (visible), unlike the old
  12/12 form where every degenerate scored 3.00.

### Phase 5 — analysis and writeup artifacts (CPU)
- Pre-registered pass/fail per pole; recovery % against the new prompt gap.
- Comparison rows: Blas et al. inventory null (same instrument, same readout
  choice shown both ways) and Serapio-García small-model Δs (same method,
  bigger models).
- Every number in the blog traced to a committed JSON artifact.

## Budget

All runs are single-token-readout administrations of a 4B model:
Phase 1 ≈ 1.5k, Phase 2 ≈ 22k, Phase 4 ≈ 65k → <100k forward passes total,
a few GPU-hours on the usual A100/T4 setup.

## Non-goals (explicit)

- No multilayer, no layers other than 15.
- No new extraction methods, no new instruments beyond the three committed.
- No parity chase with prompting; the deliverable is the controlled, quantified
  gap and which poles clear all four controls.
