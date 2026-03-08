#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# ///

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    msg = "Could not locate project root."
    raise RuntimeError(msg)


def repo_root() -> Path:
    return project_root().parents[1]


def default_source() -> Path:
    project_env = project_root() / ".env"
    if project_env.exists():
        return project_env
    return repo_root() / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy this project's .env file to a remote host over SSH.",
    )
    parser.add_argument("--host", required=True, help="Remote SSH host or IP.")
    parser.add_argument("--port", type=int, default=22, help="Remote SSH port.")
    parser.add_argument("--user", default="root", help="Remote SSH user.")
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Remote project directory that should receive .env.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source(),
        help="Local env file to copy (default: project .env, else repo root .env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        sys.stderr.write(
            f"Missing env file: {source}\n"
            f"Create {project_root() / '.env'} from "
            f"{project_root() / '.env.example'} or provide --source.",
        )
        sys.stderr.write("\n")
        return 1

    remote = f"{args.user}@{args.host}"
    mkdir_cmd = [
        "ssh",
        "-p",
        str(args.port),
        remote,
        f"mkdir -p {shlex.quote(args.remote_dir)}",
    ]
    copy_cmd = [
        "scp",
        "-P",
        str(args.port),
        str(source),
        f"{remote}:{args.remote_dir.rstrip('/')}/.env",
    ]

    if args.dry_run:
        sys.stdout.write(" ".join(shlex.quote(part) for part in mkdir_cmd))
        sys.stdout.write("\n")
        sys.stdout.write(" ".join(shlex.quote(part) for part in copy_cmd))
        sys.stdout.write("\n")
        return 0

    subprocess.run(mkdir_cmd, check=True)  # noqa: S603
    subprocess.run(copy_cmd, check=True)  # noqa: S603
    sys.stdout.write(
        f"Copied {source} -> {remote}:{args.remote_dir.rstrip('/')}/.env\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
