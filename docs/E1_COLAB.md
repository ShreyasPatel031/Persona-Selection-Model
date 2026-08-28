# E1 Colab runner (L4)

No GPU on the cloud-agent VM. Run E1 on the same Colab L4 setup used for the
bipolar sweeps.

## One cell

```python
# E1 in-span IPIP redose — Colab L4
# Runtime → GPU (L4). Expect ~1–2 h for all four poles (1 control each).

import os, subprocess, sys
from pathlib import Path

BRANCH = "cursor/ocean-vector-validation-038d"
REPO = "https://github.com/ShreyasPatel031/Persona-Selection-Model.git"
ROOT = Path("/content/persona-selection-model")

if not (ROOT / ".git").exists():
    subprocess.check_call(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO, str(ROOT)])
else:
    subprocess.check_call(["git", "-C", str(ROOT), "fetch", "origin", BRANCH])
    subprocess.check_call(["git", "-C", str(ROOT), "checkout", BRANCH])
    subprocess.check_call(["git", "-C", str(ROOT), "pull", "origin", BRANCH])

os.chdir(ROOT)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "accelerate", "sentencepiece", "protobuf"])
# Gemma-3 IT via unsloth id; transformers≥4.50 usually enough. If load fails:
#   pip install -q unsloth

VECTORS = Path("/content/ladder")
if not (VECTORS / "ladder_vectors_extraversion.pt").exists():
    # Upload /tmp/vecs_probe.tgz (or the keep_vectors pack) to Colab Files, then:
    src = Path("/content/vecs_probe.tgz")
    assert src.exists(), "Upload vecs_probe.tgz to /content/ first"
    VECTORS.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["tar", "-xzf", str(src), "-C", "/content"])
    # tarball extracts to /content/ladder/
    assert (VECTORS / "ladder_vectors_extraversion.pt").exists()

OUT = ROOT / "results" / "e1_inspan"
OUT.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "scripts/e1_inspan_redose.py",
    "--vectors-dir", str(VECTORS),
    "--out-dir", str(OUT),
    "--model-id", "unsloth/gemma-3-4b-it",
    "--random-controls", "1",
    "--probes", "2",
]
print("RUNNING:", " ".join(cmd), flush=True)
rc = subprocess.call(cmd)
print("exit", rc)

# Download these back into the repo / PR:
#   results/e1_inspan/summary.json
#   results/e1_inspan/validated_sweep_*.json
from google.colab import files
files.download(str(OUT / "summary.json"))
```

## After the run

Drop the JSONs into `results/e1_inspan/` on this branch and either:

```bash
python3 scripts/e1_inspan_redose.py --vectors-dir /tmp/keep_vectors --out-dir results/e1_inspan --evaluate-only
```

or just open `summary.json` — it already carries the pre-registered gate.

## Branch logic (from the plan)

| Outcome | Next |
|---|---|
| All 4 pass | E8 unified final run |
| N-up pass, E-up fail | E2 (`v_probe`) → E3 (layer scan) |
| N-up fail in-span | E4 (guardrail / third-person items) |
| E/A-down still moderate ρ | denser repeats; still report with CIs |
