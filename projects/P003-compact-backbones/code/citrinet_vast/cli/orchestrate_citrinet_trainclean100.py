from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from citrinet_vast import LaunchInstanceSpec, VastClient
from citrinet_vast.models import SSHConnection, VastInstanceSummary

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_LOCAL_DATA_ROOT = (
    PROJECT_ROOT / "experiments/citrinet/data/train_clean_100_full"
)
DEFAULT_LOCAL_CODE_ROOT = PROJECT_ROOT / "code/citrinet"
DEFAULT_LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "experiments/citrinet/checkpoints"
DEFAULT_LOCAL_LOG_ROOT = PROJECT_ROOT / "experiments/citrinet/logs"
DEFAULT_LOCAL_CONTROL_ROOT = PROJECT_ROOT / "experiments/citrinet/control_plane"
DEFAULT_REMOTE_ROOT = Path("/workspace/peacock-asr")
DEFAULT_REMOTE_PROJECT_ROOT = DEFAULT_REMOTE_ROOT / "projects/P003-compact-backbones"
DEFAULT_REMOTE_CODE_ROOT = DEFAULT_REMOTE_PROJECT_ROOT / "code/citrinet"
DEFAULT_REMOTE_OUTPUT_ROOT = (
    DEFAULT_REMOTE_PROJECT_ROOT / "experiments/citrinet/checkpoints"
)
DEFAULT_REMOTE_LOG_ROOT = DEFAULT_REMOTE_PROJECT_ROOT / "experiments/citrinet/logs"
DEFAULT_VOLUME_MOUNT = Path("/data")
DEFAULT_TEMPLATE_HASH = "ab21436ee2fe8894e2aef98578790fe9"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch, run, sync, and auto-destroy the first real Citrinet P2-B "
            "train_clean_100 experiment on Vast."
        )
    )
    parser.add_argument("--gpu-name", default="RTX 4090")
    parser.add_argument(
        "--image",
        default="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--disk-gb", type=float, default=150.0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--order", default="dph")
    parser.add_argument("--label", default="p003-citrinet")
    parser.add_argument("--region", default=None)
    parser.add_argument(
        "--query-clause",
        action="append",
        default=[],
        help="Additional Vast query clause, repeated as needed.",
    )
    parser.add_argument("--template-hash", default=DEFAULT_TEMPLATE_HASH)
    parser.add_argument("--attach-volume-name", default=None)
    parser.add_argument("--volume-mount", default=str(DEFAULT_VOLUME_MOUNT))
    parser.add_argument("--create-volume-on-success", default=None)
    parser.add_argument("--create-volume-size-gb", type=float, default=150.0)
    parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh/id_ed25519")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--wandb-project", default="peacock-asr-p003-citrinet")
    parser.add_argument("--wandb-entity", default="peacockery")
    parser.add_argument("--wandb-group", default="p003-citrinet-p2b")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"Required binary not found in PATH: {name}"
        raise SystemExit(msg)
    return path


def _tupled(items: list[str]) -> tuple[str, ...]:
    return tuple(item for item in items if item.strip())


def _default_query_clauses(extra: list[str]) -> tuple[str, ...]:
    base = ["reliability > 0.995", "cpu_cores >= 16"]
    return tuple(base + [item for item in extra if item.strip()])


def _instance_by_id(
    instances: list[VastInstanceSummary],
    instance_id: int,
) -> VastInstanceSummary | None:
    return next((row for row in instances if row.instance_id == instance_id), None)


def _remote_persistent_root(
    attach_volume_name: str | None,
    volume_mount: Path,
) -> Path:
    if attach_volume_name:
        return volume_mount / "p003-citrinet"
    return DEFAULT_REMOTE_PROJECT_ROOT / "persistent"


def _remote_data_root(
    attach_volume_name: str | None,
    volume_mount: Path,
) -> Path:
    return _remote_persistent_root(attach_volume_name, volume_mount) / "train_clean_100_full"


def _remote_venv_root(
    attach_volume_name: str | None,
    volume_mount: Path,
) -> Path:
    return (
        _remote_persistent_root(attach_volume_name, volume_mount)
        / "env/citrinet/.venv"
    )


def _ssh_base(ssh: SSHConnection, ssh_key: Path) -> list[str]:
    ssh_bin = _require_binary("ssh")
    return [
        ssh_bin,
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        f"{ssh.user}@{ssh.host}",
        "-p",
        str(ssh.port),
    ]


def _rsync_ssh_command(ssh: SSHConnection, ssh_key: Path) -> str:
    ssh_bin = _require_binary("ssh")
    return shlex.join(
        [
            ssh_bin,
            "-i",
            str(ssh_key),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=20",
            "-p",
            str(ssh.port),
        ]
    )


def _ssh_run(
    ssh: SSHConnection,
    ssh_key: Path,
    remote_command: str,
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [*_ssh_base(ssh, ssh_key), remote_command],
        text=True,
        capture_output=capture_output,
        check=False,
    )


def _wait_for_ssh(
    client: VastClient,
    instance_id: int,
    ssh_key: Path,
    *,
    poll_seconds: int,
    timeout_seconds: int = 900,
) -> VastInstanceSummary:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        instance = _instance_by_id(client.show_instances(), instance_id)
        if instance and instance.ssh_connection:
            probe = _ssh_run(
                instance.ssh_connection,
                ssh_key,
                "echo ready",
            )
            if probe.returncode == 0 and "ready" in probe.stdout:
                return instance
        time.sleep(poll_seconds)
    msg = f"Timed out waiting for SSH readiness on Vast instance {instance_id}"
    raise SystemExit(msg)


def _bootstrap_remote(
    ssh: SSHConnection,
    ssh_key: Path,
    *,
    attach_volume_name: str | None,
    volume_mount: Path,
) -> None:
    remote_data_root = _remote_data_root(attach_volume_name, volume_mount)
    remote_venv_root = _remote_venv_root(attach_volume_name, volume_mount)
    remote_python = remote_venv_root / "bin/python"
    remote_pip = remote_venv_root / "bin/pip"
    remote = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip rsync git curl ffmpeg libsndfile1 tmux
mkdir -p {shlex.quote(str(DEFAULT_REMOTE_CODE_ROOT))}
mkdir -p {shlex.quote(str(remote_data_root))}
mkdir -p {shlex.quote(str(DEFAULT_REMOTE_OUTPUT_ROOT))}
mkdir -p {shlex.quote(str(DEFAULT_REMOTE_LOG_ROOT))}
mkdir -p {shlex.quote(str(remote_venv_root.parent))}
if [ ! -x {shlex.quote(str(remote_python))} ]; then
  python3 -m venv {shlex.quote(str(remote_venv_root))}
  {shlex.quote(str(remote_pip))} install --upgrade pip setuptools wheel
  {shlex.quote(str(remote_pip))} install "nemo_toolkit[asr]" "wandb>=0.25.0"
fi
"""
    result = _ssh_run(ssh, ssh_key, remote)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)


def _rsync_to_remote(
    local_path: Path,
    remote_parent: str,
    ssh: SSHConnection,
    ssh_key: Path,
) -> None:
    rsync_bin = _require_binary("rsync")
    ssh_cmd = _rsync_ssh_command(ssh, ssh_key)
    result = subprocess.run(  # noqa: S603
        [
            rsync_bin,
            "-az",
            "--delete",
            "-e",
            ssh_cmd,
            f"{local_path}/",
            f"{ssh.user}@{ssh.host}:{remote_parent}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)


def _rsync_from_remote(
    remote_path: str,
    local_parent: Path,
    ssh: SSHConnection,
    ssh_key: Path,
) -> None:
    rsync_bin = _require_binary("rsync")
    ssh_cmd = _rsync_ssh_command(ssh, ssh_key)
    local_parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(  # noqa: S603
        [
            rsync_bin,
            "-az",
            "-e",
            ssh_cmd,
            f"{ssh.user}@{ssh.host}:{remote_path}/",
            f"{local_parent}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)


def _load_local_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    msg = f"Missing required environment variable: {name}"
    raise SystemExit(msg)


def _launch_remote_training(
    ssh: SSHConnection,
    ssh_key: Path,
    *,
    attach_volume_name: str | None,
    volume_mount: Path,
    run_name: str,
    batch_size: int,
    num_workers: int,
    lr: float,
    max_epochs: int,
    max_steps: int,
    precision: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    keep_instance: bool,
) -> str:
    remote_output_dir = DEFAULT_REMOTE_OUTPUT_ROOT / run_name
    remote_log_path = DEFAULT_REMOTE_LOG_ROOT / f"{run_name}.log"
    remote_script = DEFAULT_REMOTE_CODE_ROOT / "scripts/train_citrinet_p2b.py"
    remote_python = _remote_venv_root(attach_volume_name, volume_mount) / "bin/python"
    remote_data_root = _remote_data_root(attach_volume_name, volume_mount)
    remote_train_manifest = remote_data_root / "manifests/train.jsonl"
    remote_eval_manifest = remote_data_root / "manifests/eval.jsonl"
    keepalive_tail = (
        "\nstatus=$?\n"
        "echo \"$status\" > "
        f"{shlex.quote(str(remote_output_dir / 'trainer_exit_code.txt'))}\n"
        "if [ \"$status\" -eq 0 ]; then\n"
        "  while true; do sleep 3600; done\n"
        "fi\n"
    ) if keep_instance else "\n"
    remote = f"""
set -euo pipefail
mkdir -p {shlex.quote(str(DEFAULT_REMOTE_LOG_ROOT))}
export WANDB_API_KEY={shlex.quote(_require_env("WANDB_API_KEY"))}
export WANDB_MODE=online
nohup bash -lc '
{shlex.quote(str(remote_python))} {shlex.quote(str(remote_script))} \
  --train-manifest {shlex.quote(str(remote_train_manifest))} \
  --eval-manifest {shlex.quote(str(remote_eval_manifest))} \
  --output-dir {shlex.quote(str(remote_output_dir))} \
  --run-name {shlex.quote(run_name)} \
  --batch-size {batch_size} \
  --num-workers {num_workers} \
  --lr {lr} \
  --max-epochs {max_epochs} \
  --max-steps {max_steps} \
  --precision {shlex.quote(precision)} \
  --wandb-project {shlex.quote(wandb_project)} \
  --wandb-entity {shlex.quote(wandb_entity)} \
  --wandb-group {shlex.quote(wandb_group)}
{keepalive_tail}
' > {shlex.quote(str(remote_log_path))} 2>&1 < /dev/null &
"""
    result = _ssh_run(ssh, ssh_key, remote)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    return str(remote_log_path)


def _remote_training_done(
    ssh: SSHConnection,
    ssh_key: Path,
    run_name: str,
) -> tuple[bool, bool]:
    remote_output_dir = DEFAULT_REMOTE_OUTPUT_ROOT / run_name
    remote_script_name = "train_citrinet_p2b.py"
    remote = f"""
set -euo pipefail
if pgrep -af {shlex.quote(remote_script_name)} >/dev/null; then
  echo RUNNING
elif [ -f {shlex.quote(str(remote_output_dir / "report.json"))} ]; then
  echo FINISHED
else
  echo STOPPED
fi
"""
    result = _ssh_run(ssh, ssh_key, remote)
    if result.returncode != 0:
        return False, False
    state = result.stdout.strip()
    return state == "FINISHED", state == "RUNNING"


def _remote_data_ready(
    ssh: SSHConnection,
    ssh_key: Path,
    *,
    attach_volume_name: str | None,
    volume_mount: Path,
) -> bool:
    remote_manifest = (
        _remote_data_root(attach_volume_name, volume_mount) / "manifests/train.jsonl"
    )
    result = _ssh_run(
        ssh,
        ssh_key,
        f"test -f {shlex.quote(str(remote_manifest))} && echo READY || echo MISSING",
    )
    return result.returncode == 0 and result.stdout.strip() == "READY"


def _rewrite_remote_manifests(
    ssh: SSHConnection,
    ssh_key: Path,
    *,
    attach_volume_name: str | None,
    volume_mount: Path,
) -> None:
    remote_data_root = _remote_data_root(attach_volume_name, volume_mount)
    remote_manifests_root = remote_data_root / "manifests"
    local_root = str(DEFAULT_LOCAL_DATA_ROOT.resolve())
    remote_root = str(remote_data_root)
    remote = f"""
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path

manifests_root = Path({remote_manifests_root.as_posix()!r})
local_root = {local_root!r}
remote_root = {remote_root!r}
for name in ["train.jsonl", "eval.jsonl"]:
    path = manifests_root / name
    if not path.is_file():
        continue
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            audio_path = record.get("audio_filepath")
            if isinstance(audio_path, str) and audio_path.startswith(local_root):
                record["audio_filepath"] = remote_root + audio_path[len(local_root):]
            rows.append(record)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, sort_keys=True) + "\\n")
PY
"""
    result = _ssh_run(ssh, ssh_key, remote)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)


def _destroy_instance(client: VastClient, instance_id: int) -> None:
    client.destroy_instance(instance_id=instance_id)


def _snapshot_remote_log_tail(
    ssh: SSHConnection,
    ssh_key: Path,
    remote_log_path: str,
    local_tail_path: Path,
    *,
    lines: int = 200,
) -> None:
    result = _ssh_run(
        ssh,
        ssh_key,
        (
            f"if [ -f {shlex.quote(remote_log_path)} ]; then "
            f"tail -n {lines} {shlex.quote(remote_log_path)}; "
            "fi"
        ),
    )
    if result.returncode == 0 and result.stdout:
        local_tail_path.parent.mkdir(parents=True, exist_ok=True)
        local_tail_path.write_text(result.stdout, encoding="utf-8")


def _write_contract(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    _load_local_env()
    args = build_parser().parse_args()
    run_name = args.run_name or time.strftime("citrinet_256_p2b_vast_%Y%m%d_%H%M%S")
    local_log_root = DEFAULT_LOCAL_LOG_ROOT
    local_log_root.mkdir(parents=True, exist_ok=True)
    local_control_root = DEFAULT_LOCAL_CONTROL_ROOT
    local_control_root.mkdir(parents=True, exist_ok=True)

    client = VastClient.from_env()
    spec = LaunchInstanceSpec(
        gpu_name=args.gpu_name,
        image=args.image,
        num_gpus=args.num_gpus,
        disk_gb=args.disk_gb,
        limit=args.limit,
        order=args.order,
        label=f"{args.label}-{run_name}",
        region=args.region,
        template_hash=args.template_hash,
        query_clauses=_default_query_clauses(args.query_clause),
        docker_options=(
            (f"-v {args.attach_volume_name}:{args.volume_mount}",)
            if args.attach_volume_name
            else ()
        ),
    )
    if args.dry_run:
        print(
            json.dumps(
                {"launch_spec": dataclasses.asdict(spec), "run_name": run_name},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    launch = client.launch_instance(spec)
    if not launch.success or launch.instance_id is None:
        raise SystemExit(json.dumps(launch.to_dict(), indent=2, sort_keys=True))

    instance_id = launch.instance_id
    contract_path = local_control_root / f"{run_name}_vast_contract.json"
    _write_contract(
        contract_path,
        {
            "run_name": run_name,
            "launch_result": launch.to_dict(),
            "billing_policy": {
                "destroy_on_completion": not args.keep_instance,
                "reason": (
                    "Vast bills compute per second while running; storage keeps "
                    "billing while the instance exists."
                ),
            },
        },
    )
    final_state = "unknown"
    try:
        instance = _wait_for_ssh(
            client,
            instance_id,
            args.ssh_key,
            poll_seconds=args.poll_seconds,
        )
        ssh = instance.ssh_connection
        if ssh is None:
            raise SystemExit("SSH details were not available after readiness check.")

        _bootstrap_remote(
            ssh,
            args.ssh_key,
            attach_volume_name=args.attach_volume_name,
            volume_mount=Path(args.volume_mount),
        )
        _rsync_to_remote(
            DEFAULT_LOCAL_CODE_ROOT,
            str(DEFAULT_REMOTE_CODE_ROOT),
            ssh,
            args.ssh_key,
        )
        if not _remote_data_ready(
            ssh,
            args.ssh_key,
            attach_volume_name=args.attach_volume_name,
            volume_mount=Path(args.volume_mount),
        ):
            _rsync_to_remote(
                DEFAULT_LOCAL_DATA_ROOT,
                str(_remote_data_root(args.attach_volume_name, Path(args.volume_mount))),
                ssh,
                args.ssh_key,
            )
        _rewrite_remote_manifests(
            ssh,
            args.ssh_key,
            attach_volume_name=args.attach_volume_name,
            volume_mount=Path(args.volume_mount),
        )
        remote_log_path = _launch_remote_training(
            ssh,
            args.ssh_key,
            attach_volume_name=args.attach_volume_name,
            volume_mount=Path(args.volume_mount),
            run_name=run_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            max_epochs=args.max_epochs,
            max_steps=args.max_steps,
            precision=args.precision,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            keep_instance=args.keep_instance,
        )
        local_tail_path = DEFAULT_LOCAL_LOG_ROOT / f"{run_name}.tail.log"

        finished = False
        while True:
            _snapshot_remote_log_tail(
                ssh,
                args.ssh_key,
                remote_log_path,
                local_tail_path,
            )
            finished, running = _remote_training_done(ssh, args.ssh_key, run_name)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract.update(
                {
                    "last_polled_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                    "last_observed_running": running,
                    "last_observed_finished": finished,
                    "local_tail_path": str(local_tail_path),
                }
            )
            _write_contract(contract_path, contract)
            if finished:
                final_state = "finished"
                break
            if not running:
                final_state = "stopped"
                break
            time.sleep(args.poll_seconds)

        remote_output_dir = DEFAULT_REMOTE_OUTPUT_ROOT / run_name
        _rsync_from_remote(
            str(remote_output_dir),
            DEFAULT_LOCAL_OUTPUT_ROOT,
            ssh,
            args.ssh_key,
        )
        _rsync_from_remote(
            str(DEFAULT_REMOTE_LOG_ROOT),
            DEFAULT_LOCAL_LOG_ROOT,
            ssh,
            args.ssh_key,
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract.update(
            {
                "instance": instance.to_dict(),
                "remote_log_path": remote_log_path,
                "local_log_root": str(DEFAULT_LOCAL_LOG_ROOT),
                "local_output_root": str(DEFAULT_LOCAL_OUTPUT_ROOT),
                "final_state": final_state,
                "attach_volume_name": args.attach_volume_name,
                "volume_mount": args.volume_mount,
            }
        )
        _write_contract(contract_path, contract)
        if args.create_volume_on_success and final_state == "finished":
            volume_result = client.create_volume(
                instance_id=instance_id,
                size_gb=args.create_volume_size_gb,
                name=args.create_volume_on_success,
            )
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["created_volume"] = volume_result.to_dict()
            _write_contract(contract_path, contract)
        return 0 if finished else 1
    finally:
        if not args.keep_instance:
            try:
                _destroy_instance(client, instance_id)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
