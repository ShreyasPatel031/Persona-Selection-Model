# Cloud handoff — Persona Selection Model

Local `gcloud` / Colab OAuth will not exist on the cloud VM. Auth is packed under
`secrets/` (gitignored; lives in the workspace snapshot).

```
secrets/gcloud_adc.json           # Application Default Credentials (Searce user)
secrets/huggingface_token        # HF downloads (Gemma gated weights)
secrets/env                      # GCP_PROJECT, HF_TOKEN, GOOGLE_APPLICATION_CREDENTIALS
secrets/colab/sessions.json      # last Colab CLI session metadata (may be stale)
secrets/gcloud/searce_adc.json   # backup legacy ADC
```

## Load and verify

```bash
set -a && source secrets/env && set +a
python3 -c "
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json
from pathlib import Path
d=json.loads(Path('secrets/gcloud_adc.json').read_text())
c=Credentials(
    token=None,
    refresh_token=d['refresh_token'],
    token_uri='https://oauth2.googleapis.com/token',
    client_id=d['client_id'],
    client_secret=d['client_secret'],
)
c.refresh(Request())
print('gcloud adc ok', bool(c.token))
print('hf token len', len(Path('secrets/huggingface_token').read_text().strip()))
"
```

Project: `project-amer-scs-sandbox`. Account behind ADC: `shreyas.patel@searce.com`.

## Colab GPU runs

Colab CLI supports ADC auth (no browser):

```bash
set -a && source secrets/env && set +a
# Prefer ADC over interactive OAuth:
colab --auth adc new --name floor-probe --accelerator L4
```

If `colab` is missing on the VM, install with:
`uv tool install google-colab-cli` (or `pip install google-colab-cli`).

Notebook-style one-cells for prior sweeps: `docs/E0_COLAB.md`, `docs/E1_COLAB.md`.

## First GPU job after handoff

Openness floor probe — does the new 104-marker set actually lower the
openness floor below 2.78?

```bash
set -a && source secrets/env && set +a
python3 scripts/floor_probe.py \
  --items-csv data/mpi_120.csv \
  --traits openness \
  --out results/floor_probe/summary.json
```

Then the full final-cycle plan in `docs/FINAL_CYCLE_PLAN.md`.

## Important

- Never commit `secrets/`.
- If ADC refresh fails on cloud, the user must re-run
  `gcloud auth application-default login` locally and re-copy the JSON.
- Cursor cloud VMs have no GPU; Colab/GCE is the GPU path.
