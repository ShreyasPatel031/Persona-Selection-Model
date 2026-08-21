# GPU VM handoff (`oprior-1787208583-uscentral1a`)

Shared T4 in `project-amer-scs-sandbox` / `us-central1-a`. SSH is via **IAP tunnel**
(no public SSH required). Tag `allow-iap-ssh` + firewall `allow-iap-ssh-op`.

As of 2026-08-21: VM `RUNNING`, GPU idle (~0 MiB), no agent jobs.

## Connect

```bash
export CLOUDSDK_CORE_PROJECT=project-amer-scs-sandbox
export ZONE=us-central1-a
export VM=oprior-1787208583-uscentral1a

# interactive shell
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap

# one-shot remote command
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap --command='nvidia-smi'
```

## Layout on the VM

Everything lives under `~/op`:

| path | what |
|---|---|
| `~/op/.venv` | Python env (use this) |
| `~/op/app`, `~/op/scripts`, `~/op/data` | code + IPIP CSV |
| `~/op/vectors` | original ladder vectors |
| `~/op/vectors_v2` | rebuilt vectors (calibrated prompts) |
| `~/op/results/` | run outputs |

```bash
cd ~/op
export PYTHONPATH=.
source .venv/bin/activate   # or: .venv/bin/python …
nvidia-smi                  # Tesla T4, 15360 MiB
```

## Copy files

```bash
# local → VM
gcloud compute scp ./my_script.py "$VM":~/op/scripts/ --zone="$ZONE" --tunnel-through-iap

# VM → local
gcloud compute scp "$VM":~/op/results/foo.json ./ --zone="$ZONE" --tunnel-through-iap

# recursive
gcloud compute scp --recurse "$VM":~/op/results/opposite_prior_ipip ./pull/ --zone="$ZONE" --tunnel-through-iap
```

## Run something in the background (survives SSH drop)

```bash
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap --command='
cd ~/op && export PYTHONPATH=. INVENTORY_BATCH=120
setsid nohup .venv/bin/python scripts/YOUR_SCRIPT.py \
  --vectors-dir vectors_v2 \
  > ~/op/your_job.log 2>&1 < /dev/null &
echo launched pid=$!
'

# later
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap \
  --command='tail -50 ~/op/your_job.log; nvidia-smi'
```

## Don't

- Don't `pkill -f python` blindly — use the job's PID or a unique pattern
  (`pgrep -af your_script`).
- Don't set `INVENTORY_BATCH` above ~120 without the answer-slot logits fix
  already in the tree (otherwise OOM-retry thrash).
- Don't delete the VM or the `allow-iap-ssh-op` firewall rule unless everyone
  is done — other agents need IAP SSH.

## If SSH fails

```bash
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap --troubleshoot

# instance should still carry the allow-iap-ssh tag
gcloud compute instances describe "$VM" --zone="$ZONE" --format='get(tags.items)'
```
