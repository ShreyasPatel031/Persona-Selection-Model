#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the Persona Selection Model repo.
# Creates a CPU-only Python venv and installs the FastAPI app + persona pipeline deps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# python3.12-venv is not in the default image; install it once (idempotent).
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3.12-venv
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip

# CPU-only torch: no GPU in the Cloud Agent VM, and this avoids the large CUDA wheel.
# (Full Gemma-3-4b inference is intended for a separate GPU VM per the README.)
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# Test runner used by the tests/ suite.
pip install pytest

echo "Install complete. Activate with: source .venv/bin/activate"
