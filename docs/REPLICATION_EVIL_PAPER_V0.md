# Paper replication v0 — Evil trait (arXiv:2507.21509 Appendix A.1)

This run matches paper §2.1–§2.2 counts: **5** contrast pairs, **20** extraction + **20** eval questions, **10** rollouts per (pair, question), pos/neg arms → **2000** Gemma calls before filtering.

**Subject model:** Gemma-3-4b-it (VM). **Judge:** Vertex Gemini (unchanged). **Bundle:** [persona_runs/evil_paper_v0/artifacts/trait_bundle.json](../persona_runs/evil_paper_v0/artifacts/trait_bundle.json) (paper text, no step-b).

## Prerequisites

- VM with Gemma FastAPI and `HF_TOKEN`; Vertex auth for judge.
- Restart Uvicorn after deploying `app/main.py` so `/chat` accepts `do_sample`, `temperature`, `seed`.

## 1. Sync repo to VM

```bash
# From repo root on your laptop
gcloud compute scp --recurse --tunnel-through-iap \
  app/persona/ app/main.py gemma-mvp:~/gemma-chat/ \
  --zone us-central1-a --project applied-ai-practice00

# Or copy app/main.py into app/ on VM if your layout is ~/gemma-chat/app/main.py:
gcloud compute scp --tunnel-through-iap \
  app/main.py gemma-mvp:~/gemma-chat/app/main.py \
  --zone us-central1-a --project applied-ai-practice00

gcloud compute scp --recurse --tunnel-through-iap \
  persona_runs/evil_paper_v0 gemma-mvp:~/gemma-chat/persona_runs/ \
  --zone us-central1-a --project applied-ai-practice00
```

## 2. Restart Gemma server (VM)

Restart the process that serves `127.0.0.1:8080` so updated `/chat` is loaded.

## 3. Step C — rollouts + judge (~hours, ~2000×2 completions + judges)

Use **`--no-paragraph-cap`** so system prompts match the paper (no extra one-paragraph suffix).

**Foreground** (SSH session must stay open):

```bash
cd ~/gemma-chat && PYTHONPATH=. .venv/bin/python -m app.persona.run step-c \
  --run-id evil_paper_v0 \
  --bundle ~/gemma-chat/persona_runs/evil_paper_v0/artifacts/trait_bundle.json \
  --gemma-url http://127.0.0.1:8080 \
  --rollouts-per-q 10 \
  --sampling-temperature 1.0 \
  --no-paragraph-cap \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --location us-central1
```

**Background** (recommended): use **`;`** before `nohup … &`, not `&& … &`. Otherwise Bash backgrounds the whole `cd && …` chain and `$!` is not the Python PID.

```bash
cd ~/gemma-chat || exit 1
nohup env PYTHONPATH=. GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}" \
  .venv/bin/python -m app.persona.run step-c \
  --run-id evil_paper_v0 \
  --bundle ~/gemma-chat/persona_runs/evil_paper_v0/artifacts/trait_bundle.json \
  --gemma-url http://127.0.0.1:8080 \
  --rollouts-per-q 10 \
  --sampling-temperature 1.0 \
  --no-paragraph-cap \
  --project "${GOOGLE_CLOUD_PROJECT:-applied-ai-practice00}" \
  --location us-central1 \
  > /tmp/stepc_evil_paper_v0.log 2>&1 &
echo $! | tee /tmp/stepc_evil.pid
tail -f /tmp/stepc_evil_paper_v0.log   # optional; Ctrl+C does not kill step-c
```

**Monitor:** `pgrep -af "app.persona.run step-c"` or `tail -f /tmp/stepc_evil_paper_v0.log`.

**Uvicorn:** step-c talks to Gemma over HTTP. Start Uvicorn with **`HF_TOKEN`** set (or export from `~/.cache/huggingface/token`); otherwise `/chat` is up but `model_loaded` is false and rollouts fail or hang.

Outputs: `persona_runs/evil_paper_v0/rollouts/extraction_rollouts.json` and `rollouts.jsonl`.

## 4. Step D — vectors (VM, in-process Gemma)

```bash
cd ~/gemma-chat && PYTHONPATH=. .venv/bin/python -m app.persona.run step-d \
  --run-id evil_paper_v0
```

## 5. Eval answers (optional, for projection sanity)

```bash
cd ~/gemma-chat && PYTHONPATH=. .venv/bin/python -m app.persona.run eval-answers \
  --bundle ~/gemma-chat/persona_runs/evil_paper_v0/artifacts/trait_bundle.json \
  --gemma-url http://127.0.0.1:8080 \
  --no-paragraph-cap
```

## 6. Sanity projection + validate

```bash
cd ~/gemma-chat && PYTHONPATH=. .venv/bin/python -m app.persona.run sanity-eval-projection \
  --run-id evil_paper_v0

cd ~/gemma-chat && PYTHONPATH=. .venv/bin/python -m app.persona.run validate \
  --run-id evil_paper_v0 \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --location us-central1
```

## Pilot (cheap smoke test)

```bash
# 1 pair worth of traffic: use --limit 2 --rollouts-per-q 2 on a copy of the bundle with 1 contrast pair,
# or keep full bundle and use --limit 1 --rollouts-per-q 1 for a minimal judge run.
```

## Decision policy (manual)

After `validate`, use the decision tree in the project plan: the agent **suggests** next steps; **you** choose whether to change data, layer search, or accept results.
