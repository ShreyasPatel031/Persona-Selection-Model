# Fixing the last IPIP wrinkles: E-up, N-up (and E-down / A-down strength)

## Where we stand

Behavioral control (Gemini judge on steered free text) is **10/10 poles**:
sign-correct, monotone (|ρ| 0.77–1.00), large margins over same-norm random
controls (`results/bipolar`, `results/bipolar_afix`). The remaining gap is the
IPIP-NEO inventory, where the `gemma_final` sweeps gave:

| Pole | Judge ρ | IPIP ρ (full grid) | Problem |
|---|---|---|---|
| E-up | +0.94 | **−0.40** | wrong sign |
| N-up | +0.94 | **−0.20** | wrong sign |
| E-down | −1.00 | −0.50 | moderate |
| A-down | −0.94 | −0.60 | moderate |

## The key diagnostic (already run, no GPU): dose mismatch

The judge runs dosed by **trait ladder span** (E 741, A 819, C 988, N 1787,
O 1529). The IPIP sweeps dosed up to **coherence ceilings** — 3–4× further
(E to 2162, N to 7274). Past the span, any vector (including the random
control) drags answers toward the degradation attractor (C↑ N↓ O↓), which is
exactly what manufactured the wrong signs.

`scripts/inspan_reanalysis.py` recomputes IPIP ρ using only rungs inside the
behavioral span (`results/gemma_final/inspan_reanalysis.json`):

| Pole | full-grid ρ | in-span ρ | reading |
|---|---|---|---|
| A-up | +0.09 | **+1.00** | fixed by dosing alone |
| C-up | +0.54 | **+1.00** | fixed |
| C-down | +0.94 | **+1.00** | fixed |
| N-down | +0.77 | **+1.00** | fixed |
| O-down | +1.00 | **+1.00** | already fine |
| N-up | −0.20 | −0.20 (but 3.30→3.35→3.39 monotone rise through 909, collapse starts at ~1818 ≈ span edge) | needs a grid **inside 0–0.5× span** |
| E-up | −0.60 | +0.20 (rungs 135/270 are dead zone, jump at 540) | needs **denser 300–750** grid |
| E-down | +0.03 | +0.50 (only 3 rungs) | needs more in-span rungs |
| A-down | +0.60 | +0.30 (effect appeared at 1088 > L15 span 819) | needs span check / more rungs |

Conclusion: the wrong signs are mostly **overdose artifacts**, not missing
signal. The experiments below confirm and harden this.

## Experiments, in order, with branch logic

### E1. Re-dosed IPIP sweeps (one GPU session; highest expected value)

Re-run `run_validated_sweep` for the four poles with grids anchored to the
trait span, not the coherence ceiling:

- **N-up (L20)**: 8 rungs, 0.05–0.55× span (≈90–980). The existing data show a
  monotone rise precisely in this window.
- **E-up (L15)**: 8 rungs, 0.4–1.0× span (≈300–750), skipping the dead zone.
- **E-down (L15)**: same window, negative.
- **A-down (L15)**: 8 rungs up to 1.3× span (the observed effect sat at 1088;
  L20's span for A is 2197, so L15's 819 may underestimate the usable range).
- 2–3 repeats per rung (different seeds), average EV, keep the same-norm random
  control at the **same doses**.

Pass gate (pre-register it): sign correct, ρ ≥ +0.8 over ≥4 usable rungs,
beats the control's move on the target trait by ≥2× (or report the margin).

- **If all four pass** → done. Run E8 (unified re-run) and write it up. The
  paper story becomes: "one dosing rule (trait latent span) yields sign-correct
  monotone control on both instruments."
- **If N-up passes but E-up doesn't** → E is dose-dead on inventory at L15;
  go to E2 (direction) and E3 (layer).
- **If N-up still fails even at mid-dose** → go to E4 (guardrail hypothesis).

### E2. Steer E with `v_probe` instead of PC1 (half a session)

PC1 is a low/high switch; the ridge probe `v_probe` is the graded axis
(held-out ρ 0.94–1.00, cos(pc1, probe) 0.04–0.23; cached in
`/tmp/probe_vectors`, `vecs_probe.tgz`). E-up may fail because PC1's up-half is
the switch's "already high" plateau.

- **Success** → adopt probe direction for E (re-verify the judge still passes
  with it); consider probe re-runs for all traits for uniformity.
- **Failure** → direction family isn't the issue; E3.

### E3. Layer scan for E-up inventory (half a session)

The ordering-first rule (`resolve_steering_layer_for_direction`, rho×mono)
fixed A by moving L20→L15. Scan E at L13–L17 with in-span grids, pick by the
same rule applied to *inventory* response.

- **Success** → per-instrument layer choice is legitimate; document that the
  inventory reads out at a different depth than free-text style.
- **Failure** → inventory-specific block for E; E5/E6.

### E4. N-up guardrail test (cheap, one session)

Hypothesis: safety training resists first-person distress endorsements
("I get stressed easily → Agree") while free-text drama is permitted — which
would explain judge-pass/inventory-fail for N-up specifically.

Same steering, two instrument variants: (a) third-person items ("How well does
this describe this character?"), (b) explicit permission framing in the system
prompt. Compare N-up dose-response across variants.

- **Success (variant unlocks N-up)** → report as guardrail interaction; the
  inventory result is real but requires the reframed instrument.
- **Failure** → N-up on inventory is genuinely hard; fall back to E5.

### E5. Norm-preserving steering (small code change + one session)

Add αv, then rescale the hidden state to its pre-steering norm. The attractor
was shown to be largely sign-independent at high magnitude (±7273 of the same
random vector produce the same drift), i.e. a **norm** effect. Renormalizing
removes that channel; whatever movement survives is directional.

- **Success (drift gone, up-poles correct)** → mechanistic upgrade for every
  pole; re-run the full table with renorm steering. Strong paper section.
- **Failure (drift persists at matched norm)** → drift has a directional
  component; estimate it (mean ΔEV-inducing direction across random vectors)
  and orthogonalize trait vectors against it, then retest.

### E6. Rollout mean-difference extraction (expensive; only if E1–E5 fail for a pole)

The paper pipeline in `app/persona/rollouts.py`: judged contrastive rollouts →
mean-difference vectors. Lesson recorded in `docs/GPU_HOUR_SCOREBOARD.md` from
the evil axis: **volume alone does not rescue a bad bundle** (`evil_scale_v0`
failed at 2× rollouts; the fixes were bf16 and better contrast prompts). So:
run the small gate tier first (split-half cosine, separation), scale
`rollouts_per_q` 1→3→5→10 only after gates pass.

- **Success** → better vectors, re-run everything.
- **Failure after gates pass** → combined with the patch-bound result (single-
  and multi-layer patches recover ≤20% of the prompted inventory gap), this is
  strong evidence the inventory readout is not writable by any single-layer
  residual direction for these traits — publish that as a finding, with the
  judge result as the primary claim.

### E7. Moderate-ρ cleanup for E-down / A-down (piggybacks on E1)

Both look like rung scarcity, not sign problems (E-down +0.50 on 3 in-span
rungs; A-low beat its control 23.9×). Denser in-span grids + per-rung repeats
should tighten ρ toward the judge values. No separate branch: if they stay
noisy at 8 rungs × 3 repeats, report them with confidence intervals.

### E8. Unified final run (one session, after the winners are known)

All 10 poles, one recipe: per-trait layer (ordering rule), in-span grid,
repeats, random controls at matched doses, IPIP **and** judge on the same
generations, plus all five judge traits per text (this also measures Big-Two
covariance on behavior, currently unmeasured). One artifact + README.

## Cost note

E1 + E7 are one Colab session and are expected to resolve N-up, A-down,
E-down, and possibly E-up outright. E2–E5 are each ≤1 session and target E-up
/ N-up specifically. E6 is multi-session and gated. E8 is the packaging run.
