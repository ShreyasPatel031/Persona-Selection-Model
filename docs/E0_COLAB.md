# E0 Colab runner (L4) — injection scope ablation

No GPU on the cloud-agent VM. Run E0 on the same Colab L4 setup as the bipolar
and E1 sweeps.

## One cell

```python
# E0 injection-scope ablation — Colab L4
# Runtime → GPU (L4). Expect ~2–3 h for 3 poles × 2 scopes × 5 controls.

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
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "accelerate", "sentencepiece", "protobuf",
])

VECTORS = Path("/content/ladder")
if not (VECTORS / "ladder_vectors_conscientiousness.pt").exists():
    src = Path("/content/vecs_probe.tgz")
    assert src.exists(), "Upload vecs_probe.tgz to /content/ first"
    VECTORS.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["tar", "-xzf", str(src), "-C", "/content"])

OUT = ROOT / "results" / "injection_scope_ablation"
OUT.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "scripts/ablate_injection_scope.py",
    "--vectors-dir", str(VECTORS),
    "--out-dir", str(OUT),
    "--model-id", "unsloth/gemma-3-4b-it",
    "--random-controls", "5",
    "--probes", "0",
]
print("RUNNING:", " ".join(cmd), flush=True)
rc = subprocess.call(cmd)
print("exit", rc)

from google.colab import files
files.download(str(OUT / "summary.json"))
```

## After the run

Commit `results/injection_scope_ablation/summary.json` and the six
`validated_sweep_*_{full,assistant_span}.json` files into the PR branch.

Re-score without GPU:

```bash
python3 scripts/ablate_injection_scope.py --evaluate-only \
  --out-dir results/injection_scope_ablation
```

## Interpretation

| Outcome | Claim |
|---|---|
| `full` supported, `assistant_span` flat / loses to control | Exposure explains Blas inventory null |
| Both supported | Scope is not sufficient; need E1 (vector geometry) |
| Both flat | Our prior inventory claims need re-examination under both scopes |
