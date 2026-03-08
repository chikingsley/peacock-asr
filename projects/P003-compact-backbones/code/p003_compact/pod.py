"""RunPod pod management for remote P003 training and repo sync."""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_ID = "a1hkwx3tzh"
DEFAULT_GPU = "NVIDIA RTX A4000"
DEFAULT_VOLUME_GB = 200
RUNPOD_REPO_DIR = "/runpod/peacock-asr"
RUNPOD_OUTPUT_DIR = "/runpod/w2v-bert-phoneme-en"
RUNPOD_LOG_FILE = "/root/train-full.log"
SSH_ERROR_CODE = 255
SSH_BOOT_WAIT_SECONDS = 30
NVIDIA_SMI_CSV = (
    "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
    "--format=csv,noheader 2>/dev/null"
)
TRAIN_PROCESS_PATTERN = (
    "projects/P003-compact-backbones/code/training/train_phoneme_head.py"
)


def _emit(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tail_log_command(*, bytes_to_read: int, lines: int) -> str:
    return (
        f"{{ cat {RUNPOD_LOG_FILE} 2>/dev/null || "
        "cat /root/train.log 2>/dev/null; }} | "
        f"tail -c {bytes_to_read} | tr '\\r' '\\n' | tail -{lines}"
    )


def _masked_env_command() -> str:
    return (
        f"cat {RUNPOD_REPO_DIR}/.env 2>/dev/null | "
        "grep -E 'WANDB|HF_TOKEN' | sed 's/=.*/=<set>/'"
    )


def _training_process_command() -> str:
    return "ps aux | grep train_phoneme | grep -v grep || echo 'no training running'"


def _training_process_grep() -> str:
    return f"ps aux | grep {shlex.quote(TRAIN_PROCESS_PATTERN)} | grep -v grep || true"


def _latest_checkpoint_command() -> str:
    return (
        f"ls -d {RUNPOD_OUTPUT_DIR}/checkpoint-* 2>/dev/null | "
        "sort -t- -k2 -n | tail -1"
    )


def _kill_training_command() -> str:
    return (
        f"pkill -f {shlex.quote(TRAIN_PROCESS_PATTERN)} && "
        "echo 'stopped' || echo 'nothing running'"
    )


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


def _get_ssh_info(pod_id: str) -> dict[str, Any]:
    raw = _runpodctl("ssh", "info", pod_id)
    return json.loads(raw)


def _get_pod_status(pod_id: str) -> str:
    raw = _runpodctl("pod", "get", pod_id)
    return json.loads(raw).get("desiredStatus", "UNKNOWN")


def _ssh_base(pod_id: str) -> list[str]:
    info = _get_ssh_info(pod_id)
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        "-i",
        info["ssh_key"]["path"],
        f"root@{info['ip']}",
        "-p",
        str(info["port"]),
    ]


def _run_on_pod(pod_id: str, command: str) -> str:
    cmd = [*_ssh_base(pod_id), command]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode == SSH_ERROR_CODE:
        stderr = result.stderr.strip().lower()
        if "timed out" in stderr or "refused" in stderr:
            logger.error("SSH failed. Pod may still be booting. Try again in 30s.")
            sys.exit(1)
    return result.stdout


def _wait_for_ssh(pod_id: str, max_wait: int = 120) -> None:
    start = time.time()
    while time.time() - start < max_wait:
        cmd = [*_ssh_base(pod_id), "echo OK"]
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
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
        logger.info("Waiting %ds for boot...", SSH_BOOT_WAIT_SECONDS)
        time.sleep(SSH_BOOT_WAIT_SECONDS)
    _wait_for_ssh(pod_id)


def _format_pod_row(pod: dict[str, Any]) -> str:
    pod_id = pod.get("id", "?")
    name = pod.get("name", "")
    cost = pod.get("costPerHr", "?")
    gpu_count = pod.get("gpuCount", "?")
    volume = pod.get("volumeInGb", "?")
    status = pod.get("desiredStatus", "?")
    return (
        f"{pod_id}  {name:30s}  ${cost}/hr  gpu:{gpu_count}  "
        f"vol:{volume}GB  {status}"
    )


def _format_gpu_row(gpu: dict[str, Any]) -> str:
    display = gpu.get("displayName", gpu.get("gpuId", "?"))
    memory = gpu.get("memoryInGb", "?")
    stock = gpu.get("stockStatus", "?")
    gpu_id = gpu.get("gpuId", "?")
    return f"{display:20s} {memory}GB  stock:{stock:8s}  id: {gpu_id}"


def cmd_list() -> None:
    pods = json.loads(_runpodctl("pod", "list"))
    for pod in pods:
        _emit(_format_pod_row(pod))


def cmd_gpus(max_gb: int) -> None:
    gpus = json.loads(_runpodctl("gpu", "list"))
    for gpu in gpus:
        if not gpu.get("available"):
            continue
        if int(gpu.get("memoryInGb", 0)) > max_gb:
            continue
        _emit(_format_gpu_row(gpu))


def cmd_create(
    *,
    gpu: str,
    volume_gb: int,
    name: str,
    template_id: str,
) -> None:
    raw = _runpodctl(
        "pod",
        "create",
        "--template-id",
        template_id,
        "--gpu-id",
        gpu,
        "--gpu-count",
        "1",
        "--volume-in-gb",
        str(volume_gb),
        "--volume-mount-path",
        "/runpod",
        "--name",
        name,
    )
    _emit(raw.strip())


def cmd_start(pod_id: str) -> None:
    _runpodctl("pod", "start", pod_id)
    _emit(f"Pod {pod_id} started.")


def cmd_stop(pod_id: str) -> None:
    _runpodctl("pod", "stop", pod_id)
    _emit(f"Pod {pod_id} stopped.")


def cmd_destroy(pod_id: str) -> None:
    _runpodctl("pod", "stop", pod_id)
    _runpodctl("pod", "remove", pod_id)
    _emit(f"Pod {pod_id} destroyed.")


def cmd_status(pod_id: str) -> None:
    status = _get_pod_status(pod_id)
    info = _get_ssh_info(pod_id)
    _emit(f"Pod: {pod_id}  Status: {status}")
    _emit(f"SSH: root@{info['ip']} -p {info['port']}")
    if status != "RUNNING":
        return

    cmd = [*_ssh_base(pod_id), "echo OK"]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        _emit("SSH not responding (pod may still be booting).")
        return

    if result.returncode != 0:
        _emit("SSH not responding.")
        return

    output = _run_on_pod(
        pod_id,
        "\n".join(
            [
                NVIDIA_SMI_CSV,
                "echo '---'",
                _tail_log_command(bytes_to_read=2000, lines=5),
            ],
        ),
    )
    sys.stdout.write(output)


def cmd_info(pod_id: str) -> None:
    _ensure_running(pod_id)
    output = _run_on_pod(
        pod_id,
        "\n".join(
            [
                NVIDIA_SMI_CSV,
                "echo '===DISK==='",
                "df -h /runpod",
                "echo '===ENV==='",
                _masked_env_command(),
                "echo '===PROCS==='",
                _training_process_command(),
                "echo '===LOG==='",
                _tail_log_command(bytes_to_read=3000, lines=15),
            ],
        ),
    )
    sys.stdout.write(output)


def cmd_train(
    pod_id: str,
    *,
    fresh: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    _ensure_running(pod_id)
    extra_args = extra_args or []

    running = _run_on_pod(
        pod_id,
        _training_process_grep(),
    ).strip()
    if running:
        _emit(f"Training already running:\n{running}")
        _emit("Use 'peacock-asr pod status' to check progress.")
        return

    _run_on_pod(
        pod_id,
        f"cd {shlex.quote(RUNPOD_REPO_DIR)} && git pull --ff-only 2>/dev/null || true",
    )

    resume_args: list[str] = []
    if not fresh:
        latest = _run_on_pod(
            pod_id,
            _latest_checkpoint_command(),
        ).strip()
        if latest:
            resume_args = ["--resume-from-checkpoint", latest]
            _emit(f"Resuming from: {latest}")
        else:
            _emit("No checkpoint found. Starting fresh.")

    source_env = (
        'export $(cat /proc/1/environ | tr "\\0" "\\n"'
        ' | grep -E "WANDB|HF_HOME" | xargs)'
    )
    training_script = (
        f"{RUNPOD_REPO_DIR}/projects/P003-compact-backbones/code/training/"
        "train_phoneme_head.py"
    )
    script_args = shlex.join([*resume_args, *extra_args])
    launch_cmd = (
        f"{source_env} && cd {shlex.quote(RUNPOD_REPO_DIR)} && "
        f"nohup uv run --project {shlex.quote(RUNPOD_REPO_DIR)} python "
        f"{shlex.quote(training_script)} {script_args} > "
        f"{shlex.quote(RUNPOD_LOG_FILE)} 2>&1 &"
    )
    _run_on_pod(pod_id, launch_cmd)
    _emit("Training launched.")
    _emit()
    _emit("Monitor:")
    _emit("  peacock-asr pod status --pod-id <id>")
    _emit("  peacock-asr pod log --pod-id <id>")


def cmd_kill(pod_id: str) -> None:
    _ensure_running(pod_id)
    output = _run_on_pod(
        pod_id,
        _kill_training_command(),
    )
    sys.stdout.write(output)


def cmd_log(pod_id: str) -> None:
    _ensure_running(pod_id)
    log_path = _run_on_pod(
        pod_id,
        "ls -t /root/train-full.log /root/train.log 2>/dev/null | head -1",
    ).strip()
    if log_path == "":
        log_path = "/root/train.log"
    tail_command = (
        f"tail -f {shlex.quote(log_path)} | stdbuf -oL tr '\\r' '\\n'"
    )
    subprocess.run([*_ssh_base(pod_id), "-t", tail_command], check=False)  # noqa: S603


def cmd_sync(pod_id: str) -> None:
    _ensure_running(pod_id)
    info = _get_ssh_info(pod_id)
    repo_root = _repo_root()
    rsync_cmd = [
        "rsync",
        "-avz",
        "--exclude=.venv",
        "--exclude=wandb",
        "--exclude=projects/P003-compact-backbones/experiments/checkpoints",
        "--exclude=processed-features",
        "--exclude=references",
        "--exclude=.git",
        "--exclude=.agents",
        "--exclude=.claude",
        "--exclude=.pi",
        "-e",
        (
            "ssh -o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null "
            "-o ConnectTimeout=10 "
            f"-p {info['port']} -i {info['ssh_key']['path']}"
        ),
        f"{repo_root}/",
        f"root@{info['ip']}:{RUNPOD_REPO_DIR}/",
    ]
    subprocess.run(rsync_cmd, check=True)  # noqa: S603


def cmd_ssh(pod_id: str, command: str | None = None) -> None:
    _ensure_running(pod_id)
    base = _ssh_base(pod_id)
    if command:
        output = _run_on_pod(pod_id, command)
        sys.stdout.write(output)
        return
    info = _get_ssh_info(pod_id)
    _emit(f"Connecting: root@{info['ip']} -p {info['port']}")
    subprocess.run(base, check=False)  # noqa: S603
