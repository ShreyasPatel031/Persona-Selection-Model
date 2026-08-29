# Multi-layer residual patch

Prompted high−low inventory separation versus what the residual patch recovers
when the same displacement is applied at many layers at once
(`position=all`, persona-free baseline).

| Trait | prompted | single | mid_band (17) | every_2 (9) | all (34) |
|---|---|---|---|---|---|
| Extraversion | +3.26 | +0.67 (20%) | 0.00 (0%) | −0.03 (−1%) | 0.00 (0%) |
| Agreeableness | +3.07 | +0.04 (1%) | −0.02 (−1%) | −0.10 (−3%) | 0.00 (0%) |
| Conscientiousness | +4.00 | +0.14 (3%) | 0.00 (0%) | −0.04 (−1%) | −0.05 (−1%) |
| Neuroticism | +2.80 | +0.11 (4%) | 0.00 (0%) | −0.19 (−7%) | 0.00 (0%) |
| Openness | +1.72 | +0.50 (29%) | −0.02 (−1%) | −0.63 (−37%) | +0.10 (6%) |

More layers do not help. Mid-band and all-layer patches collapse the inventory
to option 3 (score ≈ 3.0) — the classic lock fingerprint — for both poles.
Single-layer remains the least-bad intervention and still recovers essentially
nothing directional.

**Conclusion:** the write-failure is not a one-site limit. Personality as
measured by this inventory is not an additive residual-stream property in
Gemma-3-4B at any number of layers. Cross-layer SAEs would interpret a
representation that cannot be written this way; they do not reopen residual
steering for this target.
