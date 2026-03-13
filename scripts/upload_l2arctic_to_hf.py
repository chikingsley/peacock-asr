#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=1.4.1",
# ]
# ///
"""Upload the L2-ARCTIC v5.0 release to the Hugging Face Hub.

The local release is distributed as a collection of zip files plus metadata
files. This helper:

1. Validates the expected release structure.
2. Generates a Hugging Face dataset card with the correct CC BY-NC 4.0 license.
3. Optionally creates a dataset repo and uploads the release files.

Examples:
    uv run scripts/upload_l2arctic_to_hf.py \
        --repo-id Peacockery/l2-arctic-release-v5.0

    uv run scripts/upload_l2arctic_to_hf.py \
        --repo-id Peacockery/l2-arctic-release-v5.0 \
        --source-dir /Users/chiejimofor/Downloads/l2arctic_release_v5.0 \
        --execute
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_SOURCE_DIR = Path("/Users/chiejimofor/Downloads/l2arctic_release_v5.0")
EXPECTED_ZIPS = {
    "ABA.zip",
    "ASI.zip",
    "BWC.zip",
    "EBVS.zip",
    "ERMS.zip",
    "HJK.zip",
    "HKK.zip",
    "HQTV.zip",
    "LXC.zip",
    "MBMPS.zip",
    "NCC.zip",
    "NJS.zip",
    "PNV.zip",
    "RRBI.zip",
    "SKA.zip",
    "SVBI.zip",
    "THV.zip",
    "TLV.zip",
    "TNI.zip",
    "TXHC.zip",
    "YBAA.zip",
    "YDCK.zip",
    "YKWK.zip",
    "ZHAA.zip",
    "suitcase_corpus.zip",
}
EXPECTED_METADATA = {"LICENSE", "PROMPTS", "README.md", "README.pdf"}


@dataclass(frozen=True)
class UploadPlan:
    source_dir: Path
    files_to_upload: list[Path]
    total_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repo, e.g. Peacockery/l2-arctic-release-v5.0",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Path to the local L2-ARCTIC release (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private instead of public.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN if set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create the repo and upload files. Without this flag the script only does a dry run.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Upload only the generated Hub dataset card files (README.md and README.original.md).",
    )
    return parser.parse_args()


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def build_dataset_card() -> str:
    return textwrap.dedent(
        """\
        ---
        pretty_name: L2-ARCTIC v5.0
        license: cc-by-nc-4.0
        task_categories:
        - automatic-speech-recognition
        - audio-classification
        language:
        - en
        tags:
        - speech
        - pronunciation
        - phoneme
        - l2
        - mispronunciation-detection
        size_categories:
        - 10K<n<100K
        configs:
        - config_name: default
          data_files:
          - split: train
            path: "*.zip"
        ---

        # L2-ARCTIC v5.0

        L2-ARCTIC is a non-native English speech corpus intended for research in
        pronunciation assessment, accent conversion, voice conversion, and
        mispronunciation detection.

        This Hub dataset mirrors the `v5.0` release as distributed locally:
        speaker-level zip archives, the suitcase corpus archive, prompts, the
        original README, and the license.

        ## Summary

        - 24 non-native English speakers
        - 26,867 utterances
        - 27.1 hours of scripted speech
        - Manual phone-level annotations for 3,599 utterances
        - Additional `suitcase_corpus` spontaneous-speech subset
        - Native language groups: Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese

        ## Structure

        Each speaker archive contains:

        - `wav/`: 44.1 kHz WAV audio
        - `transcript/`: orthographic transcripts
        - `textgrid/`: forced-aligned phoneme transcriptions
        - `annotation/`: manual pronunciation annotations where available

        ## License

        This dataset is distributed under `CC BY-NC 4.0` as provided in the
        original release. Redistribution is for non-commercial use only.

        ## Citation

        ```bibtex
        @inproceedings{zhao2018l2arctic,
          author={Guanlong {Zhao} and Sinem {Sonsaat} and Alif {Silpachai}
                  and Ivana {Lucic} and Evgeny {Chukharev-Hudilainen}
                  and John {Levis} and Ricardo {Gutierrez-Osuna}},
          title={L2-ARCTIC: A Non-native English Speech Corpus},
          year=2018,
          booktitle={Proc. Interspeech},
          pages={2783--2787},
          doi={10.21437/Interspeech.2018-1110}
        }
        ```

        ## Notes

        - This card is a Hub-specific wrapper around the original release metadata.
        - Refer to `README.original.md` for the full upstream release notes.
        - The original `LICENSE` file is included unchanged.
        """
    )


def validate_source_dir(source_dir: Path) -> UploadPlan:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    present_files = {path.name for path in source_dir.iterdir() if path.is_file()}
    missing_zips = sorted(EXPECTED_ZIPS - present_files)
    missing_metadata = sorted(EXPECTED_METADATA - present_files)

    if missing_zips or missing_metadata:
        msg_lines = ["L2-ARCTIC source directory is missing expected files:"]
        if missing_zips:
            msg_lines.append(f"  missing archives: {', '.join(missing_zips)}")
        if missing_metadata:
            msg_lines.append(f"  missing metadata: {', '.join(missing_metadata)}")
        raise ValueError("\n".join(msg_lines))

    files_to_upload = sorted(
        (path for path in source_dir.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    total_bytes = sum(path.stat().st_size for path in files_to_upload)
    return UploadPlan(
        source_dir=source_dir,
        files_to_upload=files_to_upload,
        total_bytes=total_bytes,
    )


def write_temp_card(card_text: str, source_dir: Path) -> tuple[Path, Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="l2arctic_hf_card_"))
    card_path = temp_dir / "README.md"
    card_path.write_text(card_text, encoding="utf-8")

    original_readme_path = temp_dir / "README.original.md"
    original_readme_path.write_text(
        (source_dir / "README.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return card_path, original_readme_path


def ensure_token(token: str | None) -> str:
    if token:
        return token

    msg = (
        "No Hugging Face token found. Set HF_TOKEN or pass --token before using "
        "--execute."
    )
    raise RuntimeError(msg)


def dry_run(plan: UploadPlan, repo_id: str, private: bool) -> None:
    print(f"Dry run only. No files will be uploaded.")  # noqa: T201
    print(f"Target repo: {repo_id}")  # noqa: T201
    print(f"Private repo: {private}")  # noqa: T201
    print(f"Source dir: {plan.source_dir}")  # noqa: T201
    print(f"Files: {len(plan.files_to_upload)}")  # noqa: T201
    print(f"Total size: {format_bytes(plan.total_bytes)}")  # noqa: T201
    print("First files:")  # noqa: T201
    for path in plan.files_to_upload[:10]:
        print(f"  - {path.name} ({format_bytes(path.stat().st_size)})")  # noqa: T201


def upload_release(
    *,
    plan: UploadPlan,
    repo_id: str,
    token: str,
    private: bool,
    metadata_only: bool,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    card_path, original_readme_path = write_temp_card(build_dataset_card(), plan.source_dir)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(original_readme_path),
        path_in_repo="README.original.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    if metadata_only:
        return

    for path in plan.files_to_upload:
        if path.name == "README.md":
            continue
        print(f"Uploading {path.name} ...")  # noqa: T201
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )


def main() -> int:
    args = parse_args()

    try:
        plan = validate_source_dir(args.source_dir)
        if not args.execute:
            dry_run(plan, args.repo_id, args.private)
            return 0

        token = ensure_token(args.token)
        upload_release(
            plan=plan,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            metadata_only=args.metadata_only,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    print("Upload complete.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
