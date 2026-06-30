from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from moss_mlx_conversion.dump import write_json
from moss_mlx_conversion.paths import ARTIFACTS_DIR, MLX_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a private manifest for a local MOSS MLX artifact."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=MLX_DIR / "MOSS-Transcribe-preview-2B-bf16",
    )
    parser.add_argument("--eval-summary", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "packages" / "moss-mlx-manifest.json",
    )
    return parser.parse_args()


def file_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    *,
    artifact_dir: Path,
    eval_summaries: list[Path],
) -> dict[str, Any]:
    files = [
        file_manifest(path)
        for path in sorted(artifact_dir.iterdir())
        if path.is_file() and path.name != "weights.safetensors"
    ]
    weights_path = artifact_dir / "weights.safetensors"
    if weights_path.exists():
        files.append(file_manifest(weights_path))

    return {
        "artifact_dir": str(artifact_dir),
        "files": files,
        "config": load_json_if_present(artifact_dir / "config.json"),
        "conversion_report": load_json_if_present(artifact_dir / "conversion-report.json"),
        "quantization_report": load_json_if_present(artifact_dir / "quantization-report.json"),
        "eval_summaries": [
            {
                "path": str(summary),
                "summary": load_json_if_present(summary),
            }
            for summary in eval_summaries
        ],
        "public_actions": "none",
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        artifact_dir=args.artifact_dir.resolve(),
        eval_summaries=[path.resolve() for path in args.eval_summary],
    )
    write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
