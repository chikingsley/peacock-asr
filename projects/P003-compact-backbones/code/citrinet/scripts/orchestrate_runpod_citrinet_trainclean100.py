#!/usr/bin/env python3
"""Wait for RunPod Citrinet prerequisites, then launch the first real run."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", default="5gpoxrtamdt1wx")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--expected-train-files", type=int, default=28538)
    parser.add_argument("--expected-eval-files", type=int, default=2703)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--wandb-project", default="peacock-asr-p003-citrinet")
    parser.add_argument("--wandb-entity", default="peacockery")
    parser.add_argument("--wandb-group", default="p003-citrinet-p2b")
    return parser.parse_args()


def _runpod_ssh_info(pod_id: str) -> dict[str, Any]:
    runpodctl = shutil.which("runpodctl")
    if runpodctl is None:
        raise SystemExit("runpodctl not found in PATH")
    result = subprocess.run(  # noqa: S603
        [runpodctl, "ssh", "info", pod_id],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _ssh_base(info: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-i",
        info["ssh_key"]["path"],
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=30",
        f"root@{info['ip']}",
        "-p",
        str(info["port"]),
    ]


def _run_remote(info: dict[str, Any], command: str) -> str:
    result = subprocess.run(  # noqa: S603
        [*_ssh_base(info), command],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _status(info: dict[str, Any]) -> dict[str, Any]:
    script = r"""
set -euo pipefail
base=/runpod/peacock-asr/projects/P003-compact-backbones
train_dir="$base/experiments/citrinet/data/train_clean_100_full/audio/train"
eval_dir="$base/experiments/citrinet/data/train_clean_100_full/audio/eval"
train_count=$(find "$train_dir" -type f 2>/dev/null | wc -l || true)
eval_count=$(find "$eval_dir" -type f 2>/dev/null | wc -l || true)
env_ready=0
if [ -x "$base/env/citrinet/.venv/bin/python" ]; then
  env_ready=1
fi
install_running=0
install_lines=$(ps -eo args | grep -E \
  "uv pip install --python env/citrinet/.venv/bin/python|bootstrap_citrinet_env" | \
  grep -v "grep -E" || true)
if [ -n "$install_lines" ]; then
  install_running=1
fi
training_running=0
training_lines=$(ps -eo args | grep -E \
  "launch_citrinet_trainclean100.py|train_citrinet_p2b.py" | \
  grep -v "grep -E" || true)
if [ -n "$training_lines" ]; then
  training_running=1
fi
python3 - <<'PY'
import json, os
print(json.dumps({
    "train_count": int(os.environ["TRAIN_COUNT"]),
    "eval_count": int(os.environ["EVAL_COUNT"]),
    "env_ready": os.environ["ENV_READY"] == "1",
    "install_running": os.environ["INSTALL_RUNNING"] == "1",
    "training_running": os.environ["TRAINING_RUNNING"] == "1",
}))
PY
"""
    env_prefix = (
        'TRAIN_COUNT="$train_count" EVAL_COUNT="$eval_count" '
        'ENV_READY="$env_ready" INSTALL_RUNNING="$install_running" '
        'TRAINING_RUNNING="$training_running" '
    )
    remote = script.replace("python3 - <<'PY'", env_prefix + "python3 - <<'PY'")
    return json.loads(_run_remote(info, remote))


def _launch(info: dict[str, Any], args: argparse.Namespace) -> None:
    run_name_arg = ""
    if args.run_name:
        run_name_arg = f"--run-name {shlex.quote(args.run_name)} "
    remote = f"""
set -euo pipefail
base=/runpod/peacock-asr/projects/P003-compact-backbones
export $(cat /proc/1/environ | tr '\\0' '\\n' | grep -E 'WANDB|HF_HOME' | xargs)
cd "$base"
launcher="$base/code/citrinet/scripts/launch_citrinet_trainclean100.py"
python_bin="$base/env/citrinet/.venv/bin/python"
log_path="$base/experiments/citrinet/logs/train_citrinet_256_p2b_train_clean_100.log"
nohup "$python_bin" "$launcher" \
  --python-bin "$base/env/citrinet/.venv/bin/python" \
  {run_name_arg}\
  --batch-size {args.batch_size} \
  --num-workers {args.num_workers} \
  --lr {args.lr} \
  --max-epochs {args.max_epochs} \
  --precision {shlex.quote(args.precision)} \
  --wandb-project {shlex.quote(args.wandb_project)} \
  --wandb-entity {shlex.quote(args.wandb_entity)} \
  --wandb-group {shlex.quote(args.wandb_group)} \
  > "$log_path" 2>&1 &
"""
    _run_remote(info, remote)


def main() -> None:
    args = parse_args()
    info = _runpod_ssh_info(args.pod_id)
    sys.stdout.write(f"Watching pod {args.pod_id} at {info['ip']}:{info['port']}\n")
    while True:
        status = _status(info)
        sys.stdout.write(json.dumps(status, sort_keys=True) + "\n")
        sys.stdout.flush()
        if status["training_running"]:
            sys.stdout.write("training already running\n")
            return
        data_ready = (
            status["train_count"] >= args.expected_train_files
            and status["eval_count"] >= args.expected_eval_files
        )
        env_ready = status["env_ready"] and not status["install_running"]
        if data_ready and env_ready:
            _launch(info, args)
            sys.stdout.write("launched citrinet run\n")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
