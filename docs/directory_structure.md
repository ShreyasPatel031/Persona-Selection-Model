# Directory structure

This document describes how the **Persona Selection Model** repository is laid out on disk, what is version-controlled versus generated locally, and how **`persona_runs/`** will organize pipeline outputs.

## Repository root

| Path | Role |
|------|------|
| [`README.md`](../README.md) | Setup, VM deploy notes, Phase 2 API summary |
| [`requirements.txt`](../requirements.txt) | Python dependencies (FastAPI, Transformers, `sae-lens`, Vertex AI client, …) |
| [`.gitignore`](../.gitignore) | Ignores virtualenvs, secrets (`.hf.env`), and **`persona_runs/`** |
| **`docs/`** | Design and structure notes (this file) |
| **`app/`** | FastAPI service and Python packages |
| **`scripts/`** | Shell helpers and local CLI utilities |

## Application code (`app/`)

```
app/
├── main.py              # FastAPI: /chat, /chat/stream, /health, Phase 2 routes
├── phase2.py            # Gemma Scope 2 SAE: snapshot / compare (prefill)
├── persona/             # Persona-vector pipeline (Phase 3b; growing)
│   ├── __init__.py
│   ├── config.py        # PERSONA_RUNS_DIR, Vertex model/region env defaults
│   ├── schemas.py       # Pydantic: PersonaTraitArtifact, JudgeRubric, …
│   ├── artifact_gen.py   # Step B: Vertex Gemini → validated trait JSON
│   ├── gemma_client.py   # POST /chat helper
│   ├── response_style.py # One-paragraph suffix for Gemma system prompts
│   ├── eval_answers.py   # Eval questions → pos/neg replies
│   ├── judge_vertex.py   # Step C: Vertex judge → JSON score 0–100 + short_reason
│   ├── rollouts.py       # Step C: Gemma rollouts + judge + filter + rollouts.jsonl
│   ├── activations.py    # Step D: teacher-forward, mean assistant hiddens, v_ℓ = pos−neg
│   ├── layer_heuristics.py  # Appendix B.4 v1: heuristics from v_ℓ (written in Step D summary)
│   ├── steering_demo.py  # Residual add α·(v_ℓ/||v_ℓ||) during generate (`steering-ramp` CLI; tune --alpha-max / --layer)
│   ├── grid_nine.py      # 3×3 / 2×2 dual-axis steering; `--orthogonalize-chaos`, `--norm-budget`, `--combined-layer`
│   ├── vector_compose.py # `cosine|ortho-save|alpha-grid|calibrate|dnd-grid` — multi-vector diagnostics + D&D grid
│   ├── vector_probe.py   # Optional eval alignment probe (not B.4 layer selection)
│   └── run.py            # CLI: `step-b`, …, `steering-ramp`, `sanity-eval-projection`, …
└── static/
    ├── index.html       # Chat UI
    └── phase2.html      # SAE compare UI
```

- **Phase 1–2:** Uvicorn loads `app.main:app`. Phase 2 hooks optional SAE logic from `phase2.py`.
- **Phase 3b:** New modules will live under `app/persona/` (schemas, Vertex judge client, rollouts, activations, CLI). Outputs are written only under **`persona_runs/`** (see below).

## Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| [`ssh-tunnel.sh`](../scripts/ssh-tunnel.sh) | IAP SSH local forward `127.0.0.1:8080` → VM Uvicorn |
| [`gemma-chat`](../scripts/gemma-chat) | Terminal client for `POST /chat/stream` (SSE) |
| [`vm-restart.sh`](../scripts/vm-restart.sh) | VM-side helper to restart Uvicorn (if present on the instance) |
| [`dnd_gemma_mvp.sh`](../scripts/dnd_gemma_mvp.sh) | Sync `app/persona` to `gemma-mvp`, run D&D step-b/c/d, `vector_compose calibrate` / `dnd-grid`, fetch artifacts |

## Generated and local-only paths (not in git)

These are created on your machine or VM; they stay out of version control.

| Path | Notes |
|------|--------|
| **`.venv/`**, **`.venv-sae/`**, **`.venv-persona-check/`** | Local virtual environments |
| **`persona_runs/`** | All persona pipeline run artifacts (default root; override with `PERSONA_RUNS_DIR`) |
| **`.hf.env`** / **`.gemma_hf.env`** | Hugging Face tokens (never commit) |

## `persona_runs/` layout (convention)

The default output root is **`persona_runs/`** at the repo root (resolved to an absolute path in [`app/persona/config.py`](../app/persona/config.py)). Each logical run gets its own subdirectory:

```
persona_runs/
└── <run_id>/                    # e.g. timestamp or UUID
    ├── manifest.json            # run metadata, git sha, env snapshot (optional)
    ├── artifacts/               # validated JSON: contrast prompts, Q lists, judge rubric
    ├── rollouts/                # Gemma generations per question / condition
    ├── judge/                   # Vertex Gemini judge outputs + scores
    ├── activations/             # (optional future) raw caches
    └── vectors/                 # Step D: persona_vectors.pt + summary.json
```

Exact filenames will match the implementing modules; the table above is the intended separation of concerns.

## Deployed VM layout (reference)

On the **GCE** instance (`gemma-mvp` in `applied-ai-practice00`, zone `us-central1-a`), the app is typically checked out as **`~/gemma-chat`** with a **`.venv`** there and Uvicorn bound to **`127.0.0.1:8080`**. That path is deployment convention, not enforced by this repo’s directory tree.

## Environment variables tied to layout / cloud

| Variable | Effect |
|----------|--------|
| `PERSONA_RUNS_DIR` | Override default `persona_runs` output root |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI project (also used by `gcloud`) |
| `VERTEX_LOCATION` | Vertex region (default `us-central1` in config) |
| `PERSONA_JUDGE_MODEL` | Judge model id (default `gemini-2.5-flash`) |
| `JUDGE_MAX_OUTPUT_TOKENS` | Vertex judge `max_output_tokens` (default `4096`) |
| `GEMMA_MODEL_ID` | HF id for Step D forwards (default `google/gemma-3-4b-it`) |
| `PERSONA_SAE_LAYER` | Appendix B.4 v1: block index aligned with default Phase 2 SAE (`layer_22` → `22`) |
| `PERSONA_FORCE_CPU` | Set to `1` to force CPU for Step D |
| `PERSONA_ARTIFACT_MODEL` | Step B artifact generator model (defaults to judge model) |
| `HF_TOKEN` | Hugging Face access for gated Gemma weights |

## Pipeline steps (reference)

**Step B — trait bundle:** from the repo root, with `GOOGLE_CLOUD_PROJECT` set and Application Default Credentials:

```bash
PYTHONPATH=. python -m app.persona.run step-b \
  --trait "…" \
  --trait-description "…" \
  --run-id my_run_id
```

Writes `persona_runs/<run_id>/artifacts/trait_bundle.json` plus `manifest.json`. Use `--from-json path.json` to skip Vertex and only validate/save.

**Eval answers (pos vs neg on Gemma):** `POST /chat` accepts optional JSON field **`system`** (default: helpful assistant). Run where Gemma is reachable (VM loopback or SSH tunnel):

```bash
cd ~/gemma-chat   # on the VM
PYTHONPATH=. .venv/bin/python -m app.persona.run eval-answers \
  --run-id my_run_id \
  --gemma-url http://127.0.0.1:8080
```

Writes `persona_runs/<run_id>/eval/eval_answers.json` (or `--out` / `--bundle path`).

**Step C — §2.2 rollouts + Vertex judge + filter** (paper §2.2): runs Gemma for each extraction question (pos/neg), then Vertex Gemini scores each `(system, user_q, assistant)` with **`judge_rubric`** flattened to instructions; response JSON `{"score": 0-100, "short_reason": "..."}`; keeps pos if `score > PERSONA_JUDGE_POS_MIN` (default 50), neg if `score < PERSONA_JUDGE_NEG_MAX` (default 50). Writes `rollouts/extraction_rollouts.json` + **`rollouts/rollouts.jsonl`** (one JSON object per line).

```bash
PYTHONPATH=. python -m app.persona.run step-c --run-id my_run_id --gemma-url http://127.0.0.1:8080
# Requires GOOGLE_CLOUD_PROJECT (or --project) for Vertex unless --skip-judge.
# Re-score existing file only:
PYTHONPATH=. python -m app.persona.run step-c --run-id my_run_id \
  --from-rollouts persona_runs/my_run_id/rollouts/extraction_rollouts.json
```

By default, **eval-answers** and **step-c** append a **one-paragraph (≤5 sentences)** instruction; use `--no-paragraph-cap` to disable.

**Step D — activations + persona vectors:** requires local HF weights + `HF_TOKEN`, reads **kept** rows from `rollouts/rollouts.jsonl`, runs **in-process** `AutoModelForCausalLM` with `output_hidden_states=True`, mean-pools **assistant** token positions per layer, aggregates mean over kept pos vs kept neg, saves `v = h_pos_mean - h_neg_mean`:

```bash
PYTHONPATH=. python -m app.persona.run step-d --run-id my_run_id
# outputs persona_runs/<run_id>/vectors/persona_vectors.pt and summary.json
# summary.json includes Appendix B.4 **v1** `layer_recommendation_v1` (mid-depth, argmax ||v_ℓ||,
# argmax excluding last two layers, default SAE block `PERSONA_SAE_LAYER` default 22) plus
# `recommended_layer` copied into `manifest.json` → `steps.D` (override there if you want).
```

**Appendix B.4 v1 (layer hint):** produced automatically in **Step D** — no extra command. **v2** (steering sweep + Gemini re-judge to pick ℓ) is not implemented yet.

**`sanity-eval-projection` (plan testing §4):** *not* plan Step E. Loads `persona_vectors.pt`, teacher-forwards each eval question’s **pos** vs **neg** reply (from `eval/eval_answers.json`), reports per-layer margins `dot(h_pos,v) - dot(h_neg,v)`. Manifest key: `steps.sanity_eval_projection`. The deprecated alias **`step-e`** still runs the same code but logs a warning.

```bash
PYTHONPATH=. python -m app.persona.run sanity-eval-projection --run-id my_run_id \
  --refresh-eval --gemma-url http://127.0.0.1:8080
# writes persona_runs/<run_id>/eval/sanity_eval_projection.json
```

**`steering-ramp`:** neg system fixed; at layer ℓ adds α·(v_ℓ/‖v_ℓ‖) to the decoder residual (α sweeps 0→`--alpha-max`). Writes `eval/steering_ramp.json`.

```bash
PYTHONPATH=. python -m app.persona.run steering-ramp --run-id my_run_id --eval-index 0 --steps 5 --alpha-max 8 --layer 22
```

---

*Update this file when new top-level packages or run subfolders are added.*
