"""
One-shot GCP GPU VM provisioning for Gemma /chat + step-c tiny probe.

Contract (plan): lowest-tier GPU first, GEMMA_MAX_NEW_TOKENS=128 on server,
teardown on success/failure unless --keep-vm. Requires local `gcloud` + auth.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cheapest / smallest first (GCP): T4 on n1, then L4 on g2.
DEFAULT_GPU_PROFILES: list[dict[str, str]] = [
    # n1-standard-4 (~15GB RAM) often OOMs loading Gemma-4B; use 8 vCPU / 30GB RAM.
    {
        "machine_type": "n1-standard-8",
        "accelerator": "type=nvidia-tesla-t4,count=1",
        "label": "n1-8+t4",
    },
    {
        "machine_type": "g2-standard-4",
        "accelerator": "type=nvidia-l4,count=1",
        "label": "g2+l4",
    },
]

REMOTE_ROOT = "~/gemma-chat-probe"
REMOTE_DIR = "gemma-chat-probe"


@dataclass
class GpuProbeConfig:
    project: str
    zone: str
    vertex_location: str
    run_id: str
    instance_name: str
    repo_root: Path
    limit: int = 2
    rollouts_per_q: int = 1
    keep_vm: bool = False
    skip_step_c: bool = False
    reuse_instance: str = ""
    tunnel_iap: bool = True
    boot_disk_gb: int = 200
    max_new_tokens: int = 128
    # Set after successful create
    created_instance: bool = field(default=False, init=False)
    chosen_profile: str = field(default="", init=False)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    logger.info("Running: %s", " ".join(args))
    return subprocess.run(
        args,
        check=check,
        text=True,
        capture_output=True,
        env=env or os.environ.copy(),
    )


def _try_create_instance(cfg: GpuProbeConfig) -> None:
    last_err = ""
    for prof in DEFAULT_GPU_PROFILES:
        args = [
            "gcloud",
            "compute",
            "instances",
            "create",
            cfg.instance_name,
            "--project",
            cfg.project,
            "--zone",
            cfg.zone,
            "--machine-type",
            prof["machine_type"],
            "--accelerator",
            prof["accelerator"],
            "--maintenance-policy=TERMINATE",
            # pytorch-latest-gpu was removed; use a pinned DLVM family (see GCP DLVM docs).
            "--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570",
            "--image-project=deeplearning-platform-release",
            f"--boot-disk-size={cfg.boot_disk_gb}GB",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
        ]
        try:
            r = _run(args, check=False)
            if r.returncode == 0:
                cfg.chosen_profile = prof["label"]
                cfg.created_instance = True
                logger.info(
                    "Created instance %s with profile %s",
                    cfg.instance_name,
                    cfg.chosen_profile,
                )
                return
            last_err = (r.stderr or r.stdout or "").strip()
            logger.warning(
                "Create failed for profile %s: %s",
                prof["label"],
                last_err[:500],
            )
        except FileNotFoundError:
            raise RuntimeError(
                "`gcloud` not found. Install Google Cloud SDK and ensure it is on PATH."
            ) from None
    raise RuntimeError(
        f"Could not create GPU instance in {cfg.zone} after trying all profiles. Last error: {last_err[:800]}"
    )


def _delete_instance(cfg: GpuProbeConfig) -> None:
    if not cfg.created_instance:
        return
    if cfg.keep_vm:
        logger.info("Keeping VM (--keep-vm): %s", cfg.instance_name)
        return
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            cfg.instance_name,
            "--project",
            cfg.project,
            "--zone",
            cfg.zone,
            "--quiet",
        ],
        check=False,
    )
    logger.info("Deleted instance %s", cfg.instance_name)


def _ssh(cfg: GpuProbeConfig, remote_cmd: str) -> subprocess.CompletedProcess[str]:
    base = [
        "gcloud",
        "compute",
        "ssh",
        cfg.instance_name,
        f"--project={cfg.project}",
        f"--zone={cfg.zone}",
    ]
    if cfg.tunnel_iap:
        base.append("--tunnel-through-iap")
    base += ["--command", remote_cmd]
    return _run(base, check=True)


def _push_hf_token_file(cfg: GpuProbeConfig) -> None:
    """Copy HF_TOKEN to the VM via scp (avoids embedding the secret in ssh argv/logs)."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("HF_TOKEN must be set in the environment for gpu-probe.")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".hf_token"
    ) as f:
        f.write(tok)
        tmp = Path(f.name)
    try:
        _scp_to_remote(cfg, tmp, ".hf_token_once")
    finally:
        tmp.unlink(missing_ok=True)
    _ssh(cfg, f"chmod 600 ~/{REMOTE_DIR}/.hf_token_once")


def _scp_to_remote(cfg: GpuProbeConfig, local: Path, remote_suffix: str) -> None:
    dest = f"{cfg.instance_name}:~/{REMOTE_DIR}/{remote_suffix}"
    args = ["gcloud", "compute", "scp", "--recurse"]
    if cfg.tunnel_iap:
        args.append("--tunnel-through-iap")
    args += [str(local), dest, f"--project={cfg.project}", f"--zone={cfg.zone}"]
    _run(args, check=True)


def _wait_ssh(cfg: GpuProbeConfig, attempts: int = 36, delay_s: int = 10) -> None:
    for i in range(attempts):
        cmd = [
            "gcloud",
            "compute",
            "ssh",
            cfg.instance_name,
            f"--project={cfg.project}",
            f"--zone={cfg.zone}",
        ]
        if cfg.tunnel_iap:
            cmd.append("--tunnel-through-iap")
        cmd += ["--command", "echo ssh_ok"]
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            logger.info("SSH ready after %s attempts", i + 1)
            return
        logger.info("Waiting for SSH (%s/%s)...", i + 1, attempts)
        time.sleep(delay_s)
    raise TimeoutError("SSH to instance never became ready")


def _sync_project(cfg: GpuProbeConfig) -> None:
    app_dir = cfg.repo_root / "app"
    req = cfg.repo_root / "requirements.txt"
    runs = cfg.repo_root / "persona_runs" / cfg.run_id
    if not app_dir.is_dir():
        raise FileNotFoundError(f"Missing {app_dir}")
    if not req.is_file():
        raise FileNotFoundError(f"Missing {req}")
    if not runs.is_dir():
        raise FileNotFoundError(
            f"Missing persona_runs/{cfg.run_id}. Create it or use --run-id evil_paper_v0."
        )
    _ssh(cfg, f"mkdir -p ~/{REMOTE_DIR}/persona_runs")
    _scp_to_remote(cfg, app_dir, "app")
    _scp_to_remote(cfg, req, "requirements.txt")
    _scp_to_remote(cfg, runs, f"persona_runs/{cfg.run_id}")


def _remote_bootstrap(cfg: GpuProbeConfig) -> None:
    # Debian DLVM images may ship venv without pip; install python3-venv, then one venv + requirements.
    script = r"""
set -euo pipefail
cd ~/%s
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
rm -rf .venv
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade 2>/dev/null || true
.venv/bin/pip install -U pip wheel setuptools
# Match DLVM driver 570 / CUDA 12.8; cu124 wheels can fail CUDA init on this image.
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
grep -v '^[[:space:]]*torch' requirements.txt > /tmp/req_notorch.txt || cp requirements.txt /tmp/req_notorch.txt
.venv/bin/pip install -r /tmp/req_notorch.txt
""" % (
        REMOTE_DIR,
    )
    _ssh(cfg, script)


def _remote_start_uvicorn(cfg: GpuProbeConfig) -> None:
    # Long IAP SSH sessions often drop; start Uvicorn in a detached remote script, poll with short SSH.
    start_script = f"""#!/bin/bash
set -euo pipefail
cd ~/{REMOTE_DIR}
export HF_TOKEN=$(cat .hf_token_once)
export GEMMA_MAX_NEW_TOKENS={cfg.max_new_tokens}
export DISABLE_SAE=1
export PYTHONPATH=.
pkill -f 'uvicorn app.main:app' 2>/dev/null || true
sleep 2
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080 > /tmp/gemma-uvicorn-gpu-probe.log 2>&1 &
echo $! > /tmp/gemma-uvicorn-gpu-probe.pid
echo started_uvicorn
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as f:
        f.write(start_script)
        tmp_sh = Path(f.name)
    try:
        _scp_to_remote(cfg, tmp_sh, "_start_uvicorn_probe.sh")
    finally:
        tmp_sh.unlink(missing_ok=True)
    _ssh(cfg, f"chmod +x ~/{REMOTE_DIR}/_start_uvicorn_probe.sh")
    _ssh(cfg, f"bash ~/{REMOTE_DIR}/_start_uvicorn_probe.sh")
    # Poll from laptop with short SSH calls (avoids IAP disconnect during model load).
    for attempt in range(120):
        r = subprocess.run(
            [
                "gcloud",
                "compute",
                "ssh",
                cfg.instance_name,
                f"--project={cfg.project}",
                f"--zone={cfg.zone}",
                "--tunnel-through-iap",
                "--command",
                "curl -s http://127.0.0.1:8080/health",
            ],
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and '"model_loaded":true' in out:
            logger.info("Gemma health OK: model_loaded=true (attempt %s)", attempt + 1)
            _ssh(cfg, f"rm -f ~/{REMOTE_DIR}/.hf_token_once")
            return
        time.sleep(15)
    _ssh(cfg, "tail -120 /tmp/gemma-uvicorn-gpu-probe.log || true")
    raise TimeoutError("Gemma /health never reported model_loaded=true within wait budget")


def _remote_run_step_c(cfg: GpuProbeConfig) -> None:
    proj = cfg.project.replace("'", "'\"'\"'")
    bundle = f"/home/$USER/{REMOTE_DIR}/persona_runs/{cfg.run_id}/artifacts/trait_bundle.json"
    script = f"""
set -euo pipefail
cd ~/{REMOTE_DIR}
export HF_TOKEN=$(cat .hf_token_once)
export PYTHONPATH=.
export GOOGLE_CLOUD_PROJECT='{proj}'
.venv/bin/python -m app.persona.run step-c \\
  --run-id {cfg.run_id} \\
  --bundle {bundle} \\
  --gemma-url http://127.0.0.1:8080 \\
  --limit {cfg.limit} \\
  --rollouts-per-q {cfg.rollouts_per_q} \\
  --no-paragraph-cap \\
  --project '{proj}' \\
  --location {cfg.vertex_location}
rm -f .hf_token_once
"""
    _ssh(cfg, script)


def _write_report(cfg: GpuProbeConfig, path: Path, extra: dict) -> None:
    doc = {
        "instance_name": cfg.instance_name,
        "zone": cfg.zone,
        "project": cfg.project,
        "vertex_location": cfg.vertex_location,
        "chosen_profile": cfg.chosen_profile,
        "created_instance": cfg.created_instance,
        "keep_vm": cfg.keep_vm,
        "run_id": cfg.run_id,
        "limit": cfg.limit,
        "rollouts_per_q": cfg.rollouts_per_q,
        "max_new_tokens_server": cfg.max_new_tokens,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote report %s", path)


def cmd_gpu_probe(args: argparse.Namespace) -> int:
    if getattr(args, "gpu_run", False):
        logger.info("gpu-run flag set: ephemeral GPU probe orchestration active.")
    return run_gpu_probe(args)


def run_gpu_probe(args: argparse.Namespace) -> int:
    project = (args.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
    if not project:
        logger.error("Set GOOGLE_CLOUD_PROJECT or pass --project.")
        return 1

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    instance = (args.instance_name or "").strip() or f"gemma-gpu-probe-{int(time.time())}"
    loc = (getattr(args, "location", None) or "us-central1").strip()
    cfg = GpuProbeConfig(
        project=project,
        zone=args.zone.strip(),
        vertex_location=loc,
        run_id=args.run_id.strip(),
        instance_name=instance,
        repo_root=repo_root,
        limit=int(args.limit),
        rollouts_per_q=int(args.rollouts_per_q),
        keep_vm=bool(args.keep_vm),
        skip_step_c=bool(args.skip_step_c),
        reuse_instance=(args.reuse_instance or "").strip(),
        max_new_tokens=int(args.max_new_tokens),
    )

    report_path = (
        repo_root / "persona_runs" / f"{cfg.run_id}_gpu_probe" / "gpu_probe_report.json"
    )
    extra: dict = {"status": "unknown"}
    exit_code = 1

    if cfg.reuse_instance:
        cfg.instance_name = cfg.reuse_instance
        cfg.created_instance = False
        logger.info("Reusing instance %s (will not delete at end)", cfg.instance_name)
    else:
        try:
            _try_create_instance(cfg)
        except Exception as e:
            extra["status"] = "create_failed"
            extra["error"] = str(e)
            logger.exception("%s", e)
            _write_report(cfg, report_path, extra)
            return 1

    try:
        _wait_ssh(cfg)
        _sync_project(cfg)
        _remote_bootstrap(cfg)
        _push_hf_token_file(cfg)
        _remote_start_uvicorn(cfg)
        if not cfg.skip_step_c:
            _push_hf_token_file(cfg)
            _remote_run_step_c(cfg)
        extra["status"] = "ok"
        exit_code = 0
    except KeyboardInterrupt:
        extra["status"] = "interrupted"
        logger.warning("Interrupted by user.")
        exit_code = 130
    except Exception as e:
        extra["status"] = "failed"
        extra["error"] = str(e)
        logger.exception("%s", e)
        exit_code = 1
    finally:
        if cfg.created_instance and not cfg.keep_vm:
            _delete_instance(cfg)
        _write_report(cfg, report_path, extra)

    return exit_code


def build_gpu_probe_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "gpu-probe",
        help="One-shot: create cheapest GPU VM, sync app, run Gemma with GEMMA_MAX_NEW_TOKENS=128, tiny step-c, delete VM.",
    )
    p.add_argument(
        "--project",
        default="",
        help="GCP project (default: GOOGLE_CLOUD_PROJECT).",
    )
    p.add_argument(
        "--zone",
        default=os.environ.get("GPU_PROBE_ZONE", "us-central1-a"),
        help="Zone for the probe VM (default: us-central1-a or GPU_PROBE_ZONE).",
    )
    p.add_argument(
        "--location",
        default=os.environ.get("VERTEX_LOCATION", "us-central1"),
        help="Vertex region for step-c judge (default: VERTEX_LOCATION or us-central1).",
    )
    p.add_argument(
        "--run-id",
        default="evil_paper_v0",
        help="Run id under persona_runs/ to sync (default: evil_paper_v0).",
    )
    p.add_argument(
        "--instance-name",
        default="",
        help="VM name (default: gemma-gpu-probe-<unixtime>).",
    )
    p.add_argument(
        "--reuse-instance",
        default="",
        help="Skip create/delete; use this existing instance name.",
    )
    p.add_argument(
        "--repo-root",
        default="",
        help="Repo root (default: parent of app/).",
    )
    p.add_argument("--limit", type=int, default=2, help="step-c --limit (default 2).")
    p.add_argument(
        "--rollouts-per-q",
        type=int,
        default=1,
        help="step-c --rollouts-per-q (default 1).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="GEMMA_MAX_NEW_TOKENS on remote Uvicorn (default 128).",
    )
    p.add_argument(
        "--keep-vm",
        action="store_true",
        help="Do not delete the VM after the probe.",
    )
    p.add_argument(
        "--skip-step-c",
        action="store_true",
        help="Only bootstrap + Uvicorn + health; skip step-c.",
    )
    p.add_argument(
        "--gpu-run",
        action="store_true",
        help="Explicit opt-in marker (no-op); use with automation/logging to mark GPU one-shot runs.",
    )
    p.set_defaults(func=cmd_gpu_probe)
    return p
