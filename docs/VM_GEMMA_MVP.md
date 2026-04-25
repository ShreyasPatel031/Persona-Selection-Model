# `gemma-mvp` — primary GCP VM for this repo

**Treat this instance as the default machine** for anything that should run in GCP (SSH, sync code, GPU work).

| Field | Value |
|--------|--------|
| **Name** | `gemma-mvp` |
| **Project** | `applied-ai-practice00` (or your `GOOGLE_CLOUD_PROJECT`) |
| **Zone** | `us-central1-a` |
| **SSH (IAP)** | `gcloud compute ssh gemma-mvp --project=applied-ai-practice00 --zone=us-central1-a --tunnel-through-iap` |

Work directory on the VM: **`~/gemma-chat-probe`** (sync `app/`, `requirements.txt`, and `persona_runs/<run-id>/` here).

---

## GPU: attach only when needed, remove when done

This VM is often **CPU-only** to save cost. For **any job that needs a GPU** (step-c against local Gemma, step-d on CUDA, training, etc.):

1. **Stop** the VM (Console or `gcloud compute instances stop gemma-mvp --zone=us-central1-a`).
2. **Attach 1× NVIDIA T4** (or your preferred accelerator in that zone) and set **On host maintenance** to **Terminate** (required for GPUs).  
   - **Console:** VM → Edit → GPUs → add GPU → save.  
   - **CLI:** export config, add `guestAccelerators` + `scheduling.onHostMaintenance: TERMINATE`, then update:

     ```bash
     gcloud compute instances export gemma-mvp --zone=us-central1-a --destination=/tmp/gemma-mvp.yaml
     # Edit YAML: add guestAccelerators (nvidia-tesla-t4 count 1) and scheduling.onHostMaintenance: TERMINATE
     gcloud compute instances update-from-file gemma-mvp --zone=us-central1-a \
       --source=/tmp/gemma-mvp-gpu.yaml --most-disruptive-allowed-action=RESTART
     ```

3. **Start** the VM.

4. **Drivers (stock Ubuntu):** attaching a GPU does not install drivers. If `nvidia-smi` is missing, install drivers once (see [Install GPU drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)), e.g. Ubuntu:

   ```bash
   sudo apt-get update
   sudo apt-get install -y linux-headers-$(uname -r) build-essential dkms
   sudo apt-get install -y nvidia-driver-550-server
   sudo reboot
   ```

   Prefer a **Deep Learning VM** image for new instances if you want drivers preinstalled; for `gemma-mvp` on Ubuntu, install as above.

5. **Python:** use a venv with CUDA-enabled PyTorch matching the driver/CUDA stack (see `app/persona/gpu_orchestrate.py` `_remote_bootstrap` for `cu128` + `requirements.txt` without duplicate torch).

6. **When finished:** stop the VM, **remove the GPU** (edit instance → 0 GPUs, or export YAML with `guestAccelerators: []` and `update-from-file`), optionally set **On host maintenance** back to **Migrate** if you use CPU-only again, then start.

---

## Quick checks

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## One-shot GPU without touching `gemma-mvp`

For throwaway GPU hour tests, you can still use **`python -m app.persona.run gpu-probe`** (creates a separate ephemeral GPU VM and tears it down unless `--keep-vm`). Use `gemma-mvp` for persistent work; use `gpu-probe` for isolated probes.

---

## Verified run (2026-04-01)

On **`gemma-mvp`** with **1× T4** attached (drivers installed), **`~/gemma-chat-probe`**, `python -m app.persona.run step-d --run-id gpu_nan_repro` (CUDA, **bf16** load): **split-half ~0.859**, **`v` all finite** (`torch.isfinite(v).all()`). GPU was **detached afterward** (CPU-only again) to avoid idle GPU billing.
