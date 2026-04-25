# GPU probe workflow (phased, low effort)

This implements the **Low-Effort GPU-Hour Optimization** plan: small probes first, gates, then scale.

## One-shot GPU orchestration (`gpu-probe`)

From **repo root** on a machine with `gcloud` and project access:

```bash
export HF_TOKEN=...           # required on remote for Gemma load
export GOOGLE_CLOUD_PROJECT=your-project
python -m app.persona.run gpu-probe \
  --zone us-central1-a \
  --run-id evil_gpu_probe_v0 \
  --limit 2 \
  --rollouts-per-q 1
```

Behavior:

- Tries **cheapest GPU shape first** (T4 on `n1-standard-4`, then L4 on `g2-standard-4`) until `gcloud create` succeeds.
- Bootstraps **`~/gemma-chat-probe`** on the VM, syncs `app/` + `requirements.txt` + `persona_runs/<run-id>/`.
- Starts Uvicorn with **`GEMMA_MAX_NEW_TOKENS=128`** and **CUDA device 0** when available ([`app/main.py`](../app/main.py)).
- Runs **step-c** on the VM against `http://127.0.0.1:8080`.
- **Always deletes** the probe VM on success or failure (unless `--keep-vm`).

Options:

| Flag | Meaning |
|------|--------|
| `--keep-vm` | Do not delete the instance after the run (debugging). |
| `--instance-name NAME` | Fixed name (default: `gemma-gpu-probe-<unixtime>`). |
| `--reuse-instance NAME` | Skip create; use existing VM (no delete unless you omit `--keep-vm` and… **warning**: reuse + no keep still deletes — use `--reuse-instance` with `--keep-vm` for safety). |

**Reuse safety**: If `--reuse-instance` is set, **default is not to delete** that instance at end (only ephemeral creates are deleted). Implemented as: teardown deletes only if we created this run’s instance.

| `--skip-step-c` | Only create VM + start server + health check (smoke). |

## Phase 0 — Tiny probe (CPU or existing VM)

Same as GPU tiny probe but you start Uvicorn yourself:

```bash
# step-c
python -m app.persona.run step-c --run-id evil_probe_cpu --bundle persona_runs/evil_paper_v0/artifacts/trait_bundle.json \
  --gemma-url http://127.0.0.1:8080 --limit 2 --rollouts-per-q 1 --no-paragraph-cap \
  --project "$GOOGLE_CLOUD_PROJECT" --location us-central1

python -m app.persona.run step-d --run-id evil_probe_cpu
python -m app.persona.run validate --run-id evil_probe_cpu --skip-model-gates
```

Record results on [GPU_HOUR_SCOREBOARD.md](GPU_HOUR_SCOREBOARD.md).

## Phase 1 — GPU tiny probe

Use `gpu-probe` (above). Compare **rollouts/h** and **stats** to Phase 0.

Target: **≥5×** throughput vs CPU baseline with acceptable keep-rate (see gates below).

## Phase 2 — Medium gating probe

On GPU (or fast box), partial scale:

```bash
python -m app.persona.run step-c --run-id evil_gate_v0 --bundle persona_runs/evil_paper_v0/artifacts/trait_bundle.json \
  --gemma-url http://127.0.0.1:8080 --limit 8 --rollouts-per-q 1 --no-paragraph-cap \
  --project "$GOOGLE_CLOUD_PROJECT" --location us-central1

python -m app.persona.run step-d --run-id evil_gate_v0
python -m app.persona.run sanity-eval-projection --run-id evil_gate_v0

python -m app.persona.run validate --run-id evil_gate_v0 \
  --n-candidate-layers 3 --n-questions 2 --alphas 0.5,1.0,1.5
```

**Gates (manual):**

- Keep-rate ≥ **0.6** overall; pos/neg not collapsed.
- Split-half cosine ≥ **0.5**.
- `fraction_margin_positive_at_default_layer` ≥ **0.7**.

If any fail: fix bundle/thresholds; **do not** scale `rollouts_per_q` yet.

## Phase 3 — Incremental scaling (paper replication)

Only after Phase 2 passes:

1. `rollouts_per_q = 1` → **3** → **5** → **10** (paper).
2. After each tier: `step-d` + `sanity-eval-projection` + reduced `validate`.
3. If split-half / margin **plateau**, stop increasing replicates (save GPU hours).

## Phase 4 — Full paper-scale run

Run full step-c (5×20×10, `--no-paragraph-cap`) only when Phase 3 shows clear marginal gain vs previous tier.

## Environment reference

| Variable | Effect |
|----------|--------|
| `GEMMA_MAX_NEW_TOKENS` | Server max new tokens ([`app/main.py`](../app/main.py)); `gpu-probe` sets **128** for the remote server. |
| `HF_TOKEN` | Hugging Face auth for model load. |
| `GEMMA_FORCE_CPU=1` | Force CPU on server even if CUDA exists. |

## Implementation details

- Orchestration code: [`app/persona/gpu_orchestrate.py`](../app/persona/gpu_orchestrate.py).
- CLI entry: `python -m app.persona.run gpu-probe`.
