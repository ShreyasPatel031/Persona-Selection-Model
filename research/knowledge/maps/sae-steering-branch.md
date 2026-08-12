# SAE Steering Branch

Curated view: sparse autoencoder methods for trait steering, including dead ends.

```mermaid
flowchart BT
  SAE[SAE sparse basis]
  EncDec[SAE W_enc / W_dec]
  FStat[F-stat feature ranking]
  OutFeat[Output-side feature selection]
  SSV[SAE-SSV]
  Clamp[Encode-modify-decode clamp]
  OMP[OMP decomposition]
  STA[Steering Target Atoms]
  DeadEnd[Per-feature clamp dead end]
  Dense[Dense CAA steering]

  SAE --> EncDec
  EncDec --> FStat
  EncDec --> Clamp
  EncDec --> OMP
  EncDec --> STA
  FStat --> SSV
  Dense --> OMP
  Dense --> SSV
  Clamp --> DeadEnd
  STA --> DeadEnd
  OMP --> DeadEnd
  OutFeat -.->|not yet in pipeline| SSV
```

## Method comparison (this repo)

| Method | Multi-neuron traits | Interpretable | Status |
|--------|---------------------|---------------|--------|
| Dense CAA | N/A (full dim) | Low | ✓ baseline |
| Per-feature clamp | ✗ | Medium | Dead end |
| OMP d-sweep | Partial (evil/lawful) | Medium | Negative for good/chaotic |
| STA projection | ✗ | Medium | Dead end |
| **SAE-SSV** | ✓ all four traits | Medium | **Breakthrough** |

## Epistemic limits (read after methods)

- [Prior-resident traits](../concepts/prior-resident-traits.md)
- [Non-identifiability](../concepts/non-identifiability.md)
- [Reconstruction error / dark matter](../concepts/reconstruction-error-dark-matter.md)

## Key code

- `scripts/sae_ssv_optimize.py` — SSV
- `scripts/ssv_omp_dsweep.py` — OMP d-sweep
- `scripts/sae_clamp_experiment.py` — clamp negative results
- `app/persona/sae_causality.py` — hook implementations
