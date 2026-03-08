#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=1.4.1",
#   "lhotse>=1.32.2",
#   "numpy>=2.0",
#   "pyarrow>=18.0.0",
#   "scipy>=1.14",
#   "soundfile>=0.12",
# ]
# ///
# References:
# - Hugging Face dataset: https://huggingface.co/datasets/gilkeyio/librispeech-alignments
# - Lhotse documentation: https://lhotse.readthedocs.io/
#
# Usage here:
# - Materialize a phone-labeled LibriSpeech subset into a local, icefall-friendly
#   manifest layout so the reference lane can train without hidden data services.
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable


class AudioPayload(TypedDict, total=False):
    array: object
    sampling_rate: int
    bytes: bytes | bytearray
    path: str


class PhonemeEntry(TypedDict):
    phoneme: str
    start: float
    end: float


TARGET_SAMPLE_RATE = 16_000
STRESS_RE = re.compile(r"[012]$")
PROJECT_ROOT = Path(
    os.environ.get("P004_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).expanduser()
DEFAULT_SOURCE_REPO = "gilkeyio/librispeech-alignments"
DEFAULT_ALLOWED_PHONES = (
    PROJECT_ROOT / "experiments" / "data" / "lang_phone" / "phone_list.txt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "data" / "manifests_phone_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize LibriSpeech phone-labeled audio from HF and build "
            "icefall-compatible Lhotse cut manifests."
        )
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        default=DEFAULT_SOURCE_REPO,
        help=f"HF dataset repo (default: {DEFAULT_SOURCE_REPO})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_clean_100", "dev_clean"],
        help="Dataset splits to export (default: train_clean_100 dev_clean)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for manifests and wavs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=DEFAULT_ALLOWED_PHONES,
        help=(
            "Path to the allowed-phone vocabulary. Accepts the legacy JSON vocab, "
            "the project phone_list.txt, or tokens.txt "
            f"(default: {DEFAULT_ALLOWED_PHONES})"
        ),
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help=(
            "Optional maximum examples per split for smoke builds "
            "(default: full split)"
        ),
    )
    return parser.parse_args()


def strip_stress(phone: str) -> str:
    return STRESS_RE.sub("", phone)


def load_vocab(vocab_path: Path) -> set[str]:
    text = vocab_path.read_text(encoding="utf-8").strip()
    if not text:
        msg = f"allowed-phone vocabulary is empty: {vocab_path}"
        raise ValueError(msg)

    if vocab_path.suffix == ".json" or text.startswith(("{", "[")):
        vocab = json.loads(text)
        if isinstance(vocab, dict):
            items = vocab.keys()
        elif isinstance(vocab, list):
            items = vocab
        else:
            msg = f"Unsupported JSON vocabulary payload in {vocab_path}"
            raise TypeError(msg)
        return {
            str(token)
            for token in items
            if not str(token).startswith("[") and not str(token).startswith("<")
        }

    phones: set[str] = set()
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        token = stripped.split(maxsplit=1)[0]
        if token.startswith("[") or token.startswith("<"):
            continue
        phones.add(token)
    if not phones:
        msg = f"No usable phones found in vocabulary file: {vocab_path}"
        raise ValueError(msg)
    return phones


def _load_audio_array(audio: AudioPayload) -> tuple[np.ndarray, int]:
    array_value = audio.get("array")
    sampling_rate_value = audio.get("sampling_rate")
    bytes_value = audio.get("bytes")
    path_value = audio.get("path")
    if array_value is not None and sampling_rate_value is not None:
        array = np.asarray(array_value, dtype=np.float32)
        sample_rate = sampling_rate_value
    elif isinstance(bytes_value, (bytes, bytearray)):
        array, sample_rate = sf.read(io.BytesIO(bytes_value), dtype="float32")
    elif isinstance(path_value, str):
        array, sample_rate = sf.read(path_value, dtype="float32")
    else:
        msg = (
            "Audio payload must contain decoded array data or raw bytes/path values."
        )
        raise ValueError(msg)

    if array.ndim > 1:
        array = np.mean(array, axis=1)
    return array.astype(np.float32, copy=False), sample_rate


def _resample_audio(array: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return array.astype(np.float32, copy=False)
    divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // divisor
    down = sample_rate // divisor
    return resample_poly(array, up, down).astype(np.float32, copy=False)


def icefall_split_name(hf_split: str) -> str:
    return hf_split.replace("_", "-")


def iter_examples(
    *,
    source_repo: str,
    split: str,
    max_examples: int | None,
) -> Iterable[dict[str, object]]:
    repo_files = list_repo_files(source_repo, repo_type="dataset")
    parquet_files = sorted(
        file
        for file in repo_files
        if file.startswith(f"data/{split}-") and file.endswith(".parquet")
    )
    if not parquet_files:
        msg = f"No parquet files found for split '{split}' in dataset '{source_repo}'"
        raise FileNotFoundError(msg)

    yielded = 0
    for parquet_file in parquet_files:
        local_path = hf_hub_download(
            source_repo,
            parquet_file,
            repo_type="dataset",
        )
        table = pq.read_table(local_path)
        for example in table.to_pylist():
            yield example
            yielded += 1
            if max_examples is not None and yielded >= max_examples:
                return


def main() -> int:
    args = parse_args()
    if not args.vocab_json.is_file():
        msg = f"Vocab JSON not found: {args.vocab_json}"
        raise FileNotFoundError(msg)

    allowed_phones = load_vocab(args.vocab_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = args.output_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "build_summary.json"
    if summary_path.is_file():
        build_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        build_summary = {}

    for split in args.splits:
        split_audio_dir = audio_root / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        cuts = []
        kept = 0
        skipped = 0

        for idx, example in enumerate(
            iter_examples(
                source_repo=args.source_repo,
                split=split,
                max_examples=args.max_examples_per_split,
            )
        ):
            phoneme_entries = example["phonemes"]
            if not isinstance(phoneme_entries, list):
                msg = f"Unexpected phoneme field type: {type(phoneme_entries)!r}"
                raise TypeError(msg)
            typed_phoneme_entries = cast("list[PhonemeEntry]", phoneme_entries)

            phones = [
                strip_stress(item["phoneme"]) for item in typed_phoneme_entries
            ]
            phones = [phone for phone in phones if phone in allowed_phones]
            if not phones:
                skipped += 1
                continue

            audio_payload = example["audio"]
            if not isinstance(audio_payload, dict):
                msg = f"Unexpected audio field type: {type(audio_payload)!r}"
                raise TypeError(msg)
            typed_audio_payload = cast("AudioPayload", audio_payload)

            audio_array, sample_rate = _load_audio_array(typed_audio_payload)
            if sample_rate != TARGET_SAMPLE_RATE:
                audio_array = _resample_audio(audio_array, sample_rate)
                sample_rate = TARGET_SAMPLE_RATE

            cut_id = f"{icefall_split_name(split)}-{idx:08d}"
            wav_path = split_audio_dir / f"{cut_id}.wav"
            sf.write(wav_path, audio_array, sample_rate)

            recording = Recording.from_file(str(wav_path), recording_id=cut_id)
            cut = MonoCut(
                id=cut_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )
            supervision = SupervisionSegment(
                id=cut_id,
                recording_id=cut_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=" ".join(phones),
                language="English",
            )
            cut.supervisions = [supervision]
            cuts.append(cut)
            kept += 1

        manifest_name = f"librispeech_cuts_{icefall_split_name(split)}.jsonl.gz"
        manifest_path = args.output_dir / manifest_name
        CutSet.from_cuts(cuts).to_file(manifest_path)
        build_summary[split] = {
            "kept": kept,
            "skipped": skipped,
            "manifest": str(manifest_path),
            "audio_dir": str(split_audio_dir),
        }
        logger.info(
            "Built %s with %d cuts (skipped %d)",
            manifest_path,
            kept,
            skipped,
        )

    summary_path.write_text(json.dumps(build_summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
