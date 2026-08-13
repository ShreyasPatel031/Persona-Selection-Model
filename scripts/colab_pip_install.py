import subprocess, sys
pkgs = [
    "transformers>=4.51.0",
    "accelerate>=0.33.0",
    "huggingface_hub>=0.24.0",
    "sentencepiece",
    "protobuf",
]
cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs]
print("running", cmd, flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2000:] if r.stdout else "", flush=True)
print(r.stderr[-2000:] if r.stderr else "", flush=True)
print("pip_exit", r.returncode, flush=True)
import transformers, torch
print("transformers", transformers.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
