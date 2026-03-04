"""RunPod pod management — wraps runpodctl + SSH for remote training.

CLI commands:
  peacock-asr pod status           check GPU, processes, checkpoints
  peacock-asr pod train            launch/resume training (auto-detects checkpoint)
  peacock-asr pod train --fresh    start training from scratch
  peacock-asr pod stop             stop the pod
  peacock-asr pod ssh              open interactive SSH session
  peacock-asr pod ssh <command>    run a command on the pod
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths on the pod (NFS-backed, survive restarts)
REPO_DIR = "/runpod/peacock-asr"
OUTPUT_DIR = "/runpod/w2v-bert-phoneme-en"
LOG_FILE = "/root/train-full.log"


def _require_runpodctl() -> str:
    path = shutil.which("runpodctl")
    if path is None:
        logger.error("runpodctl not found. Run: runpodctl doctor")
        sys.exit(1)
    return path


def _runpodctl(*args: str) -> str:
    cmd = [_require_runpodctl(), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.error("runpodctl %s failed: %s", " ".join(args), result.stderr.strip())
        sys.exit(1)
    return result.stdout


def _get_ssh_info(pod_id: str) -> dict:
    """Get SSH connection info from runpodctl."""
    raw = _runpodctl("ssh", "info", pod_id)
    return json.loads(raw)


def _get_pod_status(pod_id: str) -> str:
    raw = _runpodctl("pod", "get", pod_id)
    return json.loads(raw).get("desiredStatus", "UNKNOWN")


def _ssh_base(pod_id: str) -> list[str]:
    """Build base SSH command from runpodctl ssh info."""
    info = _get_ssh_info(pod_id)
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        "-i", info["ssh_key"]["path"],
        f"root@{info['ip']}",
        "-p", str(info["port"]),
    ]


def _run_on_pod(pod_id: str, command: str) -> str:
    """Run a command on the pod via SSH. Returns stdout."""
    cmd = [*_ssh_base(pod_id), command]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode == 255:
        stderr = result.stderr.strip()
        if "timed out" in stderr.lower() or "refused" in stderr.lower():
            logger.error("SSH failed — pod may still be booting. Try again in 30s.")
            sys.exit(1)
    return result.stdout


def _wait_for_ssh(pod_id: str, max_wait: int = 120) -> None:
    start = time.time()
    while time.time() - start < max_wait:
        cmd = [*_ssh_base(pod_id), "echo OK"]
        try:
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=False, timeout=15,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                return
        except subprocess.TimeoutExpired:
            pass
        elapsed = int(time.time() - start)
        logger.info("Waiting for SSH... (%ds)", elapsed)
        time.sleep(10)
    logger.error("SSH not responding after %ds", max_wait)
    sys.exit(1)


def _ensure_running(pod_id: str) -> None:
    status = _get_pod_status(pod_id)
    if status != "RUNNING":
        logger.info("Pod is %s. Starting...", status)
        _runpodctl("pod", "start", pod_id)
        logger.info("Waiting 30s for boot...")
        time.sleep(30)
    _wait_for_ssh(pod_id)


def cmd_status(pod_id: str) -> None:
    status = _get_pod_status(pod_id)
    info = _get_ssh_info(pod_id)
    print(f"Pod: {pod_id}  Status: {status}")
    print(f"SSH: root@{info['ip']} -p {info['port']}")

    if status != "RUNNING":
        return

    # Quick SSH connectivity check
    cmd = [*_ssh_base(pod_id), "echo OK"]
    try:
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=False, timeout=10,
        )
    except subprocess.TimeoutExpired:
        print("SSH not responding (pod may still be booting).")
        return

    if result.returncode != 0:
        print("SSH not responding.")
        return

    output = _run_on_pod(pod_id, f"""
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A'
echo '--- RAM ---'
free -h | grep Mem
echo '--- Disk ---'
df -h /runpod/ | tail -1
echo '--- Training ---'
ps aux | grep train_phoneme | grep -v grep || echo 'Not running'
echo '--- Checkpoints ---'
ls -d {OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n || echo 'None'
echo '--- Log (last 10) ---'
tail -10 {LOG_FILE} 2>/dev/null || echo 'No log'
""")
    print(output)


def cmd_train(pod_id: str, *, fresh: bool = False) -> None:
    _ensure_running(pod_id)

    # Check if already running (ps, not pgrep — pgrep matches stale entries)
    running = _run_on_pod(
        pod_id, "ps aux | grep train_phoneme_head | grep -v grep || true",
    ).strip()
    if running:
        print(f"Training already running:\n{running}")
        print("Use 'peacock-asr pod status' to check progress.")
        return

    # Update repo
    _run_on_pod(pod_id, f"cd {REPO_DIR} && git pull --ff-only 2>/dev/null || true")

    # Detect checkpoint
    resume_arg = ""
    if not fresh:
        latest = _run_on_pod(
            pod_id,
            f"ls -d {OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1",
        ).strip()
        if latest:
            resume_arg = f"--resume-from-checkpoint {latest}"
            print(f"Resuming from: {latest}")
        else:
            print("No checkpoint found. Starting fresh.")

    # Pod-level env vars (set via runpodctl pod update) live in /proc/1/environ
    # but SSH sessions don't inherit them. Source them explicitly.
    source_env = (
        'export $(cat /proc/1/environ | tr "\\0" "\\n"'
        ' | grep -E "WANDB|HF_HOME" | xargs)'
    )
    launch_cmd = (
        f"{source_env} && cd {REPO_DIR} && "
        f"nohup .venv/bin/python -u training/train_phoneme_head.py "
        f"{resume_arg} > {LOG_FILE} 2>&1 &"
    )
    _run_on_pod(pod_id, launch_cmd)
    print("Training launched.")
    print()
    print("Monitor:")
    print("  peacock-asr pod status")
    print("  W&B: https://wandb.ai/peacockery/w2v-bert-phoneme-en")


def cmd_stop(pod_id: str) -> None:
    _runpodctl("pod", "stop", pod_id)
    print(f"Pod {pod_id} stopped.")


def cmd_ssh(pod_id: str, command: str | None = None) -> None:
    status = _get_pod_status(pod_id)
    if status != "RUNNING":
        logger.error("Pod is %s. Start with: peacock-asr pod train", status)
        sys.exit(1)

    base = _ssh_base(pod_id)
    if command:
        output = _run_on_pod(pod_id, command)
        print(output, end="")
    else:
        info = _get_ssh_info(pod_id)
        print(f"Connecting: root@{info['ip']} -p {info['port']}")
        subprocess.run(base, check=False)  # noqa: S603
