# Colab secrets & env (VRAM / inventory / OCEAN)

**Source of truth (local, gitignored):** repo-root `.env` + `secrets/`  
(see `secrets/COLAB_PASTE.md` for the shortest Colab paste path).

Cursor Cloud secrets do **not** transfer to Colab. Copy from `.env` / `secrets/` into Colab Secrets once.

## 1. Colab Secrets (🔑 left sidebar → Secrets)

| Name | Source on laptop |
|------|------------------|
| `HF_TOKEN` | `.env` → `HF_TOKEN` |
| `GOOGLE_CLOUD_PROJECT` | `.env` → `GOOGLE_CLOUD_PROJECT` |
| `GOOGLE_APPLICATION_CREDENTIALS_B64` | `secrets/gcp-adc.b64` (one line) |
| `GEMMA_MODEL_ID` | optional; default `google/gemma-3-4b-it` |

Toggle **Notebook access** on for each secret.

## 2. First cell — load secrets into `os.environ`

```python
# Colab env bootstrap — Runtime → GPU (L4 or T4)
import os, base64
from google.colab import userdata
from huggingface_hub import login

def _get(name: str, default: str | None = None, *, required: bool = False) -> str:
    try:
        val = userdata.get(name)
    except Exception:
        val = None
    val = val or default
    if required and not val:
        raise RuntimeError(f"Missing Colab secret {name!r} (🔑 Secrets → Notebook access)")
    return val

os.environ["HF_TOKEN"] = _get("HF_TOKEN", required=True)
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["GEMMA_MODEL_ID"] = _get("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
os.environ["GOOGLE_CLOUD_PROJECT"] = _get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
os.environ["VERTEX_LOCATION"] = _get("VERTEX_LOCATION", "us-central1")
os.environ["INVENTORY_BATCH"] = _get("INVENTORY_BATCH", "16")
os.environ["DISABLE_SAE"] = "1"
os.environ["GEMMA_MAX_NEW_TOKENS"] = "128"
os.environ["PERSONA_STEER_LAYER"] = "15"

# ADC from base64 secret (full GCP user credentials)
b64 = _get("GOOGLE_APPLICATION_CREDENTIALS_B64", required=True)
adc_path = "/tmp/gcp-adc.json"
open(adc_path, "wb").write(base64.b64decode(b64))
os.chmod(adc_path, 0o600)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

import torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("ADC:", adc_path)
print("project:", os.environ["GOOGLE_CLOUD_PROJECT"])
```

## 3. Optional — interactive gcloud login instead of ADC secret

Prefer the B64 ADC secret above. If you skip it:

```python
from google.colab import auth
auth.authenticate_user()
!gcloud config set project $GOOGLE_CLOUD_PROJECT
```

## 4. Clone repo + deps (after secrets cell)

```python
import subprocess, sys, os
from pathlib import Path

REPO = "https://github.com/ShreyasPatel031/Persona-Selection-Model.git"
ROOT = Path("/content/persona-selection-model")

if not (ROOT / ".git").exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(ROOT)])
os.chdir(ROOT)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "accelerate", "sentencepiece", "protobuf",
    "huggingface_hub",
])
```

## 5. What lives where (local)

| Path | Contents |
|------|----------|
| `.env` | All env vars including HF + ADC path + ADC base64 |
| `.hf.env` | HF tokens only |
| `secrets/gcp-adc.json` | Full Application Default Credentials |
| `secrets/gcp-adc.b64` | Same, one-line base64 for Cursor/Colab secrets |
| `secrets/materialize_adc.sh` | Decode B64 → file on cloud boot |
| `secrets/COLAB_PASTE.md` | Short Colab checklist |

All under `secrets/` and `.env` are **gitignored**.

## Security

- Never commit these files. Never `print` tokens or ADC JSON.
- Colab runtimes die → re-run the bootstrap cell; Secrets store persists.
- ADC type here is `authorized_user` (your Google login refresh token), not a service-account key.