# Plain answer: did we find ↑ and ↓ OCEAN vectors?

**Short answer: not for all five traits. Only some directions work.**

Model: `unsloth/gemma-3-4b-it`. Instrument: 120-item IPIP-NEO. Both poles
tested from a persona-free baseline.

## Yes / no table

| Trait | Increase (high) | Decrease (low) | Both? |
|---|---|---|---|
| Agreeableness | score moves (+0.27) | **yes** (−0.14, strict pass) | not proven cleanly |
| Conscientiousness | **likely yes** (+0.53) | **likely yes** (−1.05) | **best bipolar candidate** |
| Extraversion | no (wrong dose-response) | weak (−0.17) | no |
| Neuroticism | no | partial (−0.42, <2× control) | no |
| Openness | weak (+0.10) | partial (−0.38, <2× control) | no |

## What “works” means here

A direction counts only if:

1. Inventory score moves the right way as steering strength grows
2. At least 3 unlocked magnitude rungs
3. Effect is ≥2× a matched random control
4. Free-text answers are not refusals / “I’m an AI” collapse

## Bottom line

- We **did** find real steering effects for **some** traits/directions.
- We **did not** get a clean “every OCEAN trait can be turned up and down” result.
- Strongest evidence: **Conscientiousness both ways**, **Agreeableness down**.
- Extraversion / Neuroticism / Openness are **not** reliably bipolar on this model under this protocol.
