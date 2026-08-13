# Research notes

Persistent research log for the Persona Selection Model project. Pipeline outputs and large artifacts stay in `persona_runs/` (gitignored); this directory stores **interpretive checkpoints** — what we tried, what worked, what failed, and where to look in code.

## Knowledge tree

Structured concept map from axioms → papers → repo code: **[knowledge/README.md](knowledge/README.md)**

- [Axioms & foundations](knowledge/axioms/) — residual stream, causal intervention, superposition
- [Concepts](knowledge/concepts/) — pipeline, SAE-SSV, epistemic limits
- [Papers](knowledge/papers/) — Chen 2025, He 2025, Mayne 2024, etc.
- [Curated views](knowledge/maps/) — project subtree, SAE steering branch

## Checkpoints

| # | Date | Title | Status |
|---|------|-------|--------|
| [001](checkpoints/001-sae-persona-steering.md) | 2026-06-16 | Persona extraction → SAE decomposition → **SAE-SSV** multi-neuron steering | **Success** (Good/Evil/Lawful/Chaotic via joint SAE optimization; per-feature clamp = single-neuron only) |
| [002](checkpoints/002-interpretability-causation-steering-conflict.md) | 2026-06-26 | Interpretability vs causation vs steering; prior-resident traits (good); non-identifiability | **Open** (addition interprets diff not trait; ablation needed for good; F-stat was correlation all along) |

## Conventions

- **Checkpoint files** live in `research/checkpoints/` as `NNN-short-slug.md`.
- Reference **run IDs** (e.g. `dnd_good_scale`) and **scripts** by path; do not duplicate large JSON/logs here.
- When a checkpoint supersedes earlier conclusions, add a new numbered file and link backward.

## Related docs (repo root)

- [README.md](../README.md) — production pipeline requirements (rollouts, layer sweep, quality gates)
- [docs/REPLICATION_EVIL_PAPER_V0.md](../docs/REPLICATION_EVIL_PAPER_V0.md) — paper-scale evil replication runbook
- [docs/GPU_HOUR_SCOREBOARD.md](../docs/GPU_HOUR_SCOREBOARD.md) — throughput and gate outcomes by run
- [docs/directory_structure.md](../docs/directory_structure.md) — `persona_runs/` layout
