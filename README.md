<p align="center"><a href="https://app.atelier-inc.net/?repo=persona-selection-model"><img src="https://app.atelier-inc.net/repos/persona-selection-model/architecture.svg" alt="Persona Selection Model architecture" /></a></p>
<p align="center"><sub>⌘-click or Ctrl+click the diagram to open the interactive viewer in a new tab.</sub></p>

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

## Persona vectors (paper replication)

Pipeline CLI: `python -m app.persona.run` (`step-b`, `step-c`, `step-d`, `validate`, …). Heavy steps run on the VM with in-process Gemma.

**Evil trait, paper-scale run:** see [docs/REPLICATION_EVIL_PAPER_V0.md](docs/REPLICATION_EVIL_PAPER_V0.md) (`evil_paper_v0` bundle under `persona_runs/`).

**GPU-hour optimization (scoreboard, phased probes, one-shot `gpu-probe`):** [docs/GPU_HOUR_SCOREBOARD.md](docs/GPU_HOUR_SCOREBOARD.md), [docs/GPU_PROBE_WORKFLOW.md](docs/GPU_PROBE_WORKFLOW.md). Ephemeral GPU VM + tiny step-c: `python -m app.persona.run gpu-probe --gpu-run` (needs `gcloud`, `HF_TOKEN`, `GOOGLE_CLOUD_PROJECT`). Local CPU tiny probe: [scripts/tiny_cpu_probe.sh](scripts/tiny_cpu_probe.sh).

**Server device:** with a GPU and drivers, Uvicorn loads Gemma on **CUDA:0** (bf16/fp16) unless `GEMMA_FORCE_CPU=1`. `GEMMA_MAX_NEW_TOKENS` still caps generation length.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | — | Hugging Face token for gated models |
| `GEMMA_MODEL_ID` | `google/gemma-3-4b-it` | Model repo id |
| `GEMMA_MAX_NEW_TOKENS` | `256` | Generation cap |

## Local development (optional)

Same as VM; CPU inference is slow. Set `HF_TOKEN` and run Uvicorn as above.
