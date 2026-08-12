<p align="center"><a href="https://app.atelier-inc.net/?repo=persona-selection-model"><img src="https://app.atelier-inc.net/repos/persona-selection-model/architecture.svg" alt="Persona Selection Model architecture" /></a></p>

# Gemma chat MVP (GCE N1 CPU)

Minimal **FastAPI** app that loads **`google/gemma-3-4b-it`** with Hugging Face **Transformers** on **CPU**, serves a tiny web UI, and exposes `POST /chat`.

## Deployed VM (project `applied-ai-practice00`)

- **Instance:** `gemma-mvp` in **`us-central1-a`**, **`n1-standard-8`**, Ubuntu 22.04; app in **`~/gemma-chat`**. Persona / GPU workflows use **`~/gemma-chat-probe`** — see **[docs/VM_GEMMA_MVP.md](docs/VM_GEMMA_MVP.md)** (default VM, attach/remove GPU, drivers).
- **SSH:** This org often allows SSH only via **IAP** (not your public IP). Use **`--tunnel-through-iap`** for `gcloud compute ssh` / `scp`.
- **Uvicorn:** Started on the VM at **`127.0.0.1:8080`** (no public firewall on 8080). If `HF_TOKEN` is not set, **`/health`** returns `"model_loaded": false` until you restart with a token.

**D&D alignment grid (persona vectors):** After syncing `app/persona/` to `~/gemma-chat`, run **`scripts/dnd_gemma_mvp.sh`** — `sync-code`, `start-uvicorn` (until `curl http://127.0.0.1:8080/health` works on the VM), `step-b-all`, `step-c-all`, `step-d-all`, then copy `persona_runs/dnd_config.example.json` → `dnd_config.json`, `push-config`, `calibrate`, `dnd-grid`, `fetch-runs`. CLI entrypoints: `python -m app.persona.vector_compose calibrate|dnd-grid`.

**Tunnel from your laptop** (leave running in a terminal):

```bash
chmod +x scripts/ssh-tunnel.sh
./scripts/ssh-tunnel.sh
```

Or manually:

```bash
gcloud compute ssh gemma-mvp --project=applied-ai-practice00 --zone=us-central1-a \
  --tunnel-through-iap -- -L 8080:127.0.0.1:8080 -N
```

Then open **http://127.0.0.1:8080/** .

**Terminal chat (no JSON — just type):** with the tunnel still running, in another terminal:

```bash
./scripts/gemma-chat
```

Uses Python’s stdlib only. Optional: `GEMMA_URL` (default `http://127.0.0.1:8080`). Quit with `/quit` or Ctrl+D. Empty lines are skipped.

The server also serves **`POST /chat/stream`** (SSE) for streaming; **`gemma-chat`** uses that under the hood.

## Phase 2 — Pretrained SAE (Gemma Scope 2)

After `pip install -r requirements.txt` (adds **`sae-lens`**), the server loads a **public** SAE from [SAE Lens pretrained releases](https://decoderesearch.github.io/SAELens/) matching **`google/gemma-3-4b-it`**:

- Default **`SAE_RELEASE`:** `gemma-scope-2-4b-it-res`
- Default **`SAE_ID`:** `layer_22_width_16k_l0_medium` (16k width; swap for `layer_22_width_262k_l0_medium` etc. if you have RAM/GPU headroom)

**Endpoints**

- `GET /health` → includes `phase2_sae` (`loaded`, `hook_name`, `d_sae`, …).
- `POST /phase2/sae_snapshot` — JSON `{ "message", "system", "topk" }` → SAE code stats at **last prefill token**.
- `POST /phase2/sae_compare` — `{ "message", "system_a", "system_b", "topk" }` → two snapshots + top‑k **Jaccard** overlap.

**UI:** [http://127.0.0.1:8080/phase2.html](http://127.0.0.1:8080/phase2.html) (same tunnel as chat).

**Env overrides**

| Variable | Purpose |
|----------|---------|
| `DISABLE_SAE=1` | Skip SAE load (Phase 1 only). |
| `SAE_RELEASE` | e.g. `gemma-scope-2-4b-it-res` |
| `SAE_ID` | e.g. `layer_22_width_16k_l0_medium` |
| `SAE_HIDDEN_STATE_INDEX` | If dim mismatch, set HF `hidden_states` index manually (default = resid layer + 1). |

**Note:** This is **prefill-only** (two full forwards for compare). Per-token SAE during generation is a later step. CPU + 4B + SAE is heavy; **GPU** recommended for interactive use.

**Enable Gemma on the VM** (one-time; replace with your real token):

```bash
gcloud compute ssh gemma-mvp --project=applied-ai-practice00 --zone=us-central1-a \
  --tunnel-through-iap --command='pkill -f "uvicorn app.main:app" || true'
# Paste token only in your own terminal (not in chat logs):
gcloud compute ssh gemma-mvp --project=applied-ai-practice00 --zone=us-central1-a \
  --tunnel-through-iap --command="cd ~/gemma-chat && export HF_TOKEN='YOUR_HF_TOKEN' && nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn.log 2>&1 &"
```

First model download + CPU generation can take **many minutes**.

## Prerequisites

- Hugging Face: accept the Gemma model license and create a **read** access token.
- A VM with enough **RAM** for 4B on CPU (e.g. **n1-standard-8** on GCE).

## Setup (on the VM)

```bash
cd "/path/to/Persona Selection Model"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="hf_..."   # or huggingface-cli login
```

## Run

Bind to loopback so the service is only reachable via **SSH port forwarding** (no public `:8080` rule needed):

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## SSH tunnel (from your laptop)

If your project restricts SSH to IAP, add **`--tunnel-through-iap`**:

```bash
gcloud compute ssh gemma-mvp --project=YOUR_PROJECT --zone=YOUR_ZONE \
  --tunnel-through-iap -- -L 8080:127.0.0.1:8080 -N
```

Then open **http://127.0.0.1:8080/** or:

```bash
curl -s http://127.0.0.1:8080/health
curl -s -X POST http://127.0.0.1:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Say hello in one sentence.","system":"You are a helpful assistant."}'
```

Optional JSON fields for **`POST /chat`**: `do_sample` (bool), `temperature` (0–2, used when sampling), `seed` (int, reproducibility).

## Persona Vectors Pipeline

Implements the pipeline from **Chen et al. (2025)** "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)).

CLI: `python -m app.persona.run` (`step-b`, `step-c`, `step-d`, `quality-gates`, `calibrate`, …).

### Pipeline Steps (DO NOT skip or shortcut)

| Step | What | Critical Parameters |
|------|------|-------------------|
| **B** | Generate trait artifacts (prompts, questions, rubric) | 5 prompt pairs, 20 extraction + 20 eval questions |
| **C** | Generate contrastive rollouts + judge filtering | **10 rollouts/question**, judge enabled (score >50/<50) |
| **D** | Extract persona vectors (mean residuals over response tokens) | All layers saved |
| **Quality Gates** | **Causal layer sweep** + steering effectiveness | Picks best layer by actual trait expression |
| **Calibrate** | Alpha sweep at the chosen layer | Finds α where trait is high + coherence ≥80 |

### CRITICAL: What the Paper Requires (and what goes wrong if you skip it)

#### 1. Layer Selection — MUST be causal, NEVER hardcoded

The paper (Appendix B.4) selects the steering layer by sweeping all layers with a fixed α and measuring trait expression. **DO NOT** use:
- The SAE layer (layer 22) — this is for interpretability only, not steering
- The argmax-norm layer — late layers have high norms but produce incoherence
- Any hardcoded default

**What happens if you get this wrong:** The vector produces incoherent text before any behavioral change occurs. At layer 22 on Gemma-3-4b, "good" gets trait=0.75 at α=1.5 (useless). At layer 16, the same vector gets trait=80.3 (works perfectly).

Run: `python -m app.persona.run quality-gates --run-id <id>` — this performs the causal sweep.

#### 2. Rollouts Per Question — MUST be 10, not 1

The paper uses 10 rollouts per (question × prompt pair). With 5 pairs × 20 questions × 10 rollouts = 1000 items per arm before filtering. This gives:
- Enough samples for stable mean estimation
- Split-half cosine ≥0.8 (direction is reliable)

**What happens if you use 1:** 100 items per arm (10x less data). The mean direction is noisy, split-half cosine drops, and the vector captures vocabulary shifts instead of behavioral ones.

#### 3. Quality Filtering — MUST use the judge

The paper filters rollouts by trait score: keep pos only if score >50, neg only if score <50. This removes cases where the model didn't follow the system prompt.

**What happens if you skip the judge:** Noisy rollouts where pos and neg sound similar dilute the contrastive signal. The resulting vector points toward superficial differences (tone words) rather than deep behavioral directions.

`--skip-judge` is acceptable ONLY for rapid prototyping. Any production vector MUST be judge-filtered.

#### 4. Extraction Questions — MUST be behavioral scenarios

Questions must force a behavioral choice, not ask for abstract essays. Good: "You find a wallet with $500, nobody is watching, you are broke." Bad: "What does it mean to be a good person?"

### Quick Reference: Full Production Run

```bash
# On VM with GPU:
python -m app.persona.run step-c \
  --run-id my_trait \
  --gemma-url http://127.0.0.1:8080 \
  --rollouts-per-q 10 \
  --project applied-ai-practice00

python -m app.persona.run step-d --run-id my_trait

python -m app.persona.run quality-gates --run-id my_trait \
  --project applied-ai-practice00

# quality-gates output tells you the correct layer and alpha.
```

### Older Docs

**Evil trait, paper-scale run:** see [docs/REPLICATION_EVIL_PAPER_V0.md](docs/REPLICATION_EVIL_PAPER_V0.md).

**GPU-hour optimization:** [docs/GPU_HOUR_SCOREBOARD.md](docs/GPU_HOUR_SCOREBOARD.md), [docs/GPU_PROBE_WORKFLOW.md](docs/GPU_PROBE_WORKFLOW.md).

**Server device:** with a GPU and drivers, Uvicorn loads Gemma on **CUDA:0** (bf16/fp16) unless `GEMMA_FORCE_CPU=1`. `GEMMA_MAX_NEW_TOKENS` still caps generation length.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | — | Hugging Face token for gated models |
| `GEMMA_MODEL_ID` | `google/gemma-3-4b-it` | Model repo id |
| `GEMMA_MAX_NEW_TOKENS` | `256` | Generation cap |

## Local development (optional)

Same as VM; CPU inference is slow. Set `HF_TOKEN` and run Uvicorn as above.
