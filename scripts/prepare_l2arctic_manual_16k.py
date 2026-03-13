#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "huggingface_hub>=1.4.1",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "soundfile>=0.12",
# ]
# ///
"""Prepare a 16 kHz manual-annotation L2-ARCTIC dataset and optionally upload it.

This script creates a Hugging Face `datasets`-native export that is directly
usable by `training/train_phoneme_head.py`:

- `audio`: 16 kHz mono FLAC
- `phonemes`: perceived phone sequence with start/end timestamps

It also preserves reference pronunciation data for mispronunciation work:

- `canonical_phonemes`
- `manual_events`

The scripted manual subset is emitted as `train` / `validation` / `test`
according to the common 12/6/6 speaker split used in prior work. The
spontaneous suitcase recordings are emitted as a separate `suitcase` split.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import HfApi
from scipy.signal import resample_poly


DEFAULT_SOURCE_DIR = Path("/Users/chiejimofor/Downloads/l2arctic_release_v5.0")
DEFAULT_OUTPUT_DIR = Path.home() / ".cache" / "peacock-asr" / "l2_arctic_manual_v5_16k"
SCRIPTED_DEV_SPEAKERS = {"MBMPS", "NCC", "SVBI", "THV", "YBAA", "YDCK"}
SCRIPTED_TEST_SPEAKERS = {"NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"}
SILENCE_PHONES = {"", "SIL", "SP"}
PHONE_MAP = {
    "AX": "AH",
    "AXR": "ER",
    "IX": "IH",
}
STRESS_RE = re.compile(r"\d")
TIER_RE = re.compile(
    r'item \[\d+\]:\s+class = "IntervalTier"\s+name = "(?P<name>[^"]+)"'
    r"\s+xmin = [^\n]+\s+xmax = [^\n]+\s+intervals: size = \d+\s+"
    r"(?P<body>.*?)(?=(?:\n\s*item \[\d+\]:)|\Z)",
    re.DOTALL,
)
INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s+"
    r"xmin = (?P<xmin>[^\n]+)\s+"
    r"xmax = (?P<xmax>[^\n]+)\s+"
    r'text = "(?P<text>(?:[^"]|"")*)"',
    re.DOTALL,
)
SPEAKER_META: dict[str, dict[str, str]] = {
    "ABA": {"gender": "M", "native_language": "Arabic"},
    "ASI": {"gender": "M", "native_language": "Hindi"},
    "BWC": {"gender": "M", "native_language": "Chinese"},
    "EBVS": {"gender": "M", "native_language": "Spanish"},
    "ERMS": {"gender": "M", "native_language": "Spanish"},
    "HJK": {"gender": "F", "native_language": "Korean"},
    "HKK": {"gender": "M", "native_language": "Korean"},
    "HQTV": {"gender": "M", "native_language": "Vietnamese"},
    "LXC": {"gender": "F", "native_language": "Chinese"},
    "MBMPS": {"gender": "F", "native_language": "Spanish"},
    "NCC": {"gender": "F", "native_language": "Chinese"},
    "NJS": {"gender": "F", "native_language": "Spanish"},
    "PNV": {"gender": "F", "native_language": "Vietnamese"},
    "RRBI": {"gender": "M", "native_language": "Hindi"},
    "SKA": {"gender": "F", "native_language": "Arabic"},
    "SVBI": {"gender": "F", "native_language": "Hindi"},
    "THV": {"gender": "F", "native_language": "Vietnamese"},
    "TLV": {"gender": "M", "native_language": "Vietnamese"},
    "TNI": {"gender": "F", "native_language": "Hindi"},
    "TXHC": {"gender": "M", "native_language": "Chinese"},
    "YBAA": {"gender": "M", "native_language": "Arabic"},
    "YDCK": {"gender": "F", "native_language": "Korean"},
    "YKWK": {"gender": "M", "native_language": "Korean"},
    "ZHAA": {"gender": "F", "native_language": "Arabic"},
}


@dataclass(frozen=True)
class Stats:
    rows: int = 0
    duration_hours: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Path to the raw L2-ARCTIC v5.0 release (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated FLAC files and cached artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Target sample rate for exported audio.",
    )
    parser.add_argument(
        "--repo-id",
        help="Optional Hugging Face dataset repo to upload to, e.g. chikingsley/l2-arctic-manual-v5.0-16k",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN if set.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub dataset as private instead of public.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually push the prepared dataset to Hugging Face.",
    )
    parser.add_argument(
        "--skip-suitcase",
        action="store_true",
        help="Skip the spontaneous suitcase subset.",
    )
    parser.add_argument(
        "--speakers",
        nargs="*",
        help="Optional uppercase speaker codes to restrict processing for testing.",
    )
    parser.add_argument(
        "--limit-per-speaker",
        type=int,
        help="Optional limit on scripted manual utterances per speaker for testing.",
    )
    return parser.parse_args()


def ensure_token(token: str | None) -> str:
    if token:
        return token
    msg = "No Hugging Face token found. Set HF_TOKEN or pass --token before using --execute."
    raise RuntimeError(msg)


def speaker_split(speaker_id: str) -> str:
    if speaker_id in SCRIPTED_TEST_SPEAKERS:
        return "test"
    if speaker_id in SCRIPTED_DEV_SPEAKERS:
        return "validation"
    return "train"


def parse_textgrid(text: str) -> dict[str, list[dict[str, float | str]]]:
    tiers: dict[str, list[dict[str, float | str]]] = {}
    for tier_match in TIER_RE.finditer(text):
        name = tier_match.group("name")
        body = tier_match.group("body")
        intervals: list[dict[str, float | str]] = []
        for interval_match in INTERVAL_RE.finditer(body):
            intervals.append(
                {
                    "start": float(interval_match.group("xmin")),
                    "end": float(interval_match.group("xmax")),
                    "text": interval_match.group("text").replace('""', '"').strip(),
                }
            )
        tiers[name] = intervals
    return tiers


def parse_phone_label(label: str) -> dict[str, str]:
    cleaned = label.strip()
    if not cleaned:
        return {
            "canonical_raw": "",
            "perceived_raw": "",
            "error_type": "silence",
            "raw_label": "",
        }

    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) == 1:
        return {
            "canonical_raw": parts[0],
            "perceived_raw": parts[0],
            "error_type": "correct",
            "raw_label": cleaned,
        }

    if len(parts) >= 3:
        code = parts[-1].lower()
        error_type = {"s": "substitution", "a": "addition", "d": "deletion"}.get(
            code,
            "tagged",
        )
        return {
            "canonical_raw": parts[0],
            "perceived_raw": parts[1],
            "error_type": error_type,
            "raw_label": cleaned,
        }

    return {
        "canonical_raw": cleaned,
        "perceived_raw": cleaned,
        "error_type": "unknown",
        "raw_label": cleaned,
    }


def normalize_phone(phone: str) -> str | None:
    cleaned = STRESS_RE.sub("", phone.strip().upper().replace("*", ""))
    if cleaned in SILENCE_PHONES or cleaned == "ERR":
        return None
    mapped = PHONE_MAP.get(cleaned, cleaned)
    return mapped if mapped else None


def interval_to_event(interval: dict[str, float | str]) -> dict[str, float | str | None]:
    parsed = parse_phone_label(str(interval["text"]))
    return {
        "start": float(interval["start"]),
        "end": float(interval["end"]),
        "raw_label": parsed["raw_label"],
        "canonical_raw": parsed["canonical_raw"],
        "perceived_raw": parsed["perceived_raw"],
        "canonical_phoneme": normalize_phone(parsed["canonical_raw"]),
        "perceived_phoneme": normalize_phone(parsed["perceived_raw"]),
        "error_type": parsed["error_type"],
    }


def build_phone_sequences(
    events: list[dict[str, float | str | None]],
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], dict[str, int]]:
    perceived: list[dict[str, float | str]] = []
    canonical: list[dict[str, float | str]] = []
    counts = {"substitutions": 0, "additions": 0, "deletions": 0}

    for event in events:
        error_type = str(event["error_type"])
        start = float(event["start"])
        end = float(event["end"])
        canonical_phone = event["canonical_phoneme"]
        perceived_phone = event["perceived_phoneme"]

        if error_type == "substitution":
            counts["substitutions"] += 1
        elif error_type == "addition":
            counts["additions"] += 1
        elif error_type == "deletion":
            counts["deletions"] += 1

        if perceived_phone is not None:
            perceived.append(
                {
                    "phoneme": str(perceived_phone),
                    "start": start,
                    "end": end,
                }
            )

        if canonical_phone is not None and error_type != "addition":
            canonical.append(
                {
                    "phoneme": str(canonical_phone),
                    "start": start,
                    "end": end,
                }
            )

    return perceived, canonical, counts


def load_audio_from_zip(
    zf: ZipFile,
    member: str,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    audio_bytes = zf.read(member)
    audio_array, source_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        up = sample_rate // divisor
        down = source_rate // divisor
        audio_array = resample_poly(audio_array, up, down).astype(np.float32, copy=False)
        source_rate = sample_rate
    return audio_array.astype(np.float32, copy=False), source_rate


def write_flac(path: Path, audio_array: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio_array, sample_rate, format="FLAC")


def validate_source_dir(source_dir: Path) -> None:
    expected_files = {
        "README.md",
        "LICENSE",
        "suitcase_corpus.zip",
    }
    missing = [name for name in expected_files if not (source_dir / name).exists()]
    if missing:
        msg = f"Source directory is missing expected files: {', '.join(missing)}"
        raise FileNotFoundError(msg)

    for speaker_id in SPEAKER_META:
        archive = source_dir / f"{speaker_id}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"Missing speaker archive: {archive}")


def read_text(zf: ZipFile, member: str) -> str:
    return zf.read(member).decode("utf-8").strip()


def scripted_rows_for_speaker(
    zip_path: Path,
    output_dir: Path,
    sample_rate: int,
    limit_per_speaker: int | None,
) -> dict[str, list[dict[str, object]]]:
    speaker_id = zip_path.stem.upper()
    split = speaker_split(speaker_id)
    rows_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)

    with ZipFile(zip_path) as zf:
        annotation_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{speaker_id}/annotation/") and name.endswith(".TextGrid")
        )
        if limit_per_speaker is not None:
            annotation_members = annotation_members[:limit_per_speaker]

        for index, annotation_member in enumerate(annotation_members, start=1):
            utterance_id = Path(annotation_member).stem
            transcript_member = f"{speaker_id}/transcript/{utterance_id}.txt"
            audio_member = f"{speaker_id}/wav/{utterance_id}.wav"
            output_audio = output_dir / "audio" / split / speaker_id / f"{utterance_id}.flac"

            annotation_tiers = parse_textgrid(read_text(zf, annotation_member))
            phone_intervals = annotation_tiers.get("phones")
            if not phone_intervals:
                raise ValueError(f"No phones tier found in {annotation_member}")

            events = [interval_to_event(interval) for interval in phone_intervals]
            phonemes, canonical_phonemes, counts = build_phone_sequences(events)

            if output_audio.exists():
                audio_info = sf.info(output_audio)
                duration_seconds = float(audio_info.frames) / float(audio_info.samplerate)
            else:
                audio_array, actual_rate = load_audio_from_zip(zf, audio_member, sample_rate)
                write_flac(output_audio, audio_array, actual_rate)
                duration_seconds = float(len(audio_array)) / float(actual_rate)

            rows_by_split[split].append(
                {
                    "audio": str(output_audio),
                    "speaker_id": speaker_id,
                    "gender": SPEAKER_META[speaker_id]["gender"],
                    "native_language": SPEAKER_META[speaker_id]["native_language"],
                    "subset": "scripted_manual",
                    "corpus_split": split,
                    "utterance_id": utterance_id,
                    "transcript": read_text(zf, transcript_member),
                    "audio_duration_sec": duration_seconds,
                    "phonemes": phonemes,
                    "canonical_phonemes": canonical_phonemes,
                    "manual_events": events,
                    "num_substitutions": counts["substitutions"],
                    "num_additions": counts["additions"],
                    "num_deletions": counts["deletions"],
                    "num_phonemes": len(phonemes),
                    "num_canonical_phonemes": len(canonical_phonemes),
                    "source_release": "L2-ARCTIC v5.0",
                }
            )

            if index % 50 == 0:
                print(  # noqa: T201
                    f"  {speaker_id}: processed {index}/{len(annotation_members)} scripted utterances"
                )

    return rows_by_split


def suitcase_rows(
    zip_path: Path,
    output_dir: Path,
    sample_rate: int,
    allowed_speakers: set[str] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with ZipFile(zip_path) as zf:
        annotation_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith("suitcase_corpus/annotation/") and name.endswith(".TextGrid")
        )

        for annotation_member in annotation_members:
            speaker_lower = Path(annotation_member).stem
            speaker_id = speaker_lower.upper()
            if allowed_speakers is not None and speaker_id not in allowed_speakers:
                continue

            transcript_member = f"suitcase_corpus/transcript/{speaker_lower}.txt"
            audio_member = f"suitcase_corpus/wav/{speaker_lower}.wav"
            output_audio = output_dir / "audio" / "suitcase" / f"{speaker_id}.flac"

            annotation_tiers = parse_textgrid(read_text(zf, annotation_member))
            phone_intervals = annotation_tiers.get("phones")
            if not phone_intervals:
                raise ValueError(f"No phones tier found in {annotation_member}")

            events = [interval_to_event(interval) for interval in phone_intervals]
            phonemes, canonical_phonemes, counts = build_phone_sequences(events)

            if output_audio.exists():
                audio_info = sf.info(output_audio)
                duration_seconds = float(audio_info.frames) / float(audio_info.samplerate)
            else:
                audio_array, actual_rate = load_audio_from_zip(zf, audio_member, sample_rate)
                write_flac(output_audio, audio_array, actual_rate)
                duration_seconds = float(len(audio_array)) / float(actual_rate)

            rows.append(
                {
                    "audio": str(output_audio),
                    "speaker_id": speaker_id,
                    "gender": SPEAKER_META[speaker_id]["gender"],
                    "native_language": SPEAKER_META[speaker_id]["native_language"],
                    "subset": "suitcase_manual",
                    "corpus_split": "suitcase",
                    "utterance_id": f"{speaker_id.lower()}_suitcase",
                    "transcript": read_text(zf, transcript_member),
                    "audio_duration_sec": duration_seconds,
                    "phonemes": phonemes,
                    "canonical_phonemes": canonical_phonemes,
                    "manual_events": events,
                    "num_substitutions": counts["substitutions"],
                    "num_additions": counts["additions"],
                    "num_deletions": counts["deletions"],
                    "num_phonemes": len(phonemes),
                    "num_canonical_phonemes": len(canonical_phonemes),
                    "source_release": "L2-ARCTIC v5.0 suitcase corpus",
                }
            )

    return rows


def build_dataset_dict(rows_by_split: dict[str, list[dict[str, object]]], sample_rate: int) -> DatasetDict:
    split_map: dict[str, Dataset] = {}
    for split_name, rows in rows_by_split.items():
        dataset = Dataset.from_list(rows)
        split_map[split_name] = dataset.cast_column(
            "audio",
            Audio(sampling_rate=sample_rate),
        )
    return DatasetDict(split_map)


def compute_stats(dataset_dict: DatasetDict) -> dict[str, Stats]:
    stats: dict[str, Stats] = {}
    for split_name, dataset in dataset_dict.items():
        duration_hours = sum(float(value) for value in dataset["audio_duration_sec"]) / 3600.0
        stats[split_name] = Stats(rows=len(dataset), duration_hours=duration_hours)
    return stats


def build_readme(stats: dict[str, Stats], repo_id: str, sample_rate: int) -> str:
    split_lines = []
    for split_name in ["train", "validation", "test", "suitcase"]:
        if split_name not in stats:
            continue
        split_stat = stats[split_name]
        split_lines.append(
            f"- `{split_name}`: {split_stat.rows} rows, {split_stat.duration_hours:.2f} hours"
        )

    split_block = "\n".join(split_lines)
    return (
        "---\n"
        "pretty_name: L2-ARCTIC Manual 16k\n"
        "license: cc-by-nc-4.0\n"
        "task_categories:\n"
        "- automatic-speech-recognition\n"
        "- audio-classification\n"
        "language:\n"
        "- en\n"
        "tags:\n"
        "- speech\n"
        "- pronunciation\n"
        "- phoneme\n"
        "- l2\n"
        "- mispronunciation-detection\n"
        "size_categories:\n"
        "- 1K<n<10K\n"
        "---\n\n"
        f"# {repo_id.split('/')[-1]}\n\n"
        "This dataset is a prepared derivative of **L2-ARCTIC v5.0** that keeps only\n"
        "the manually annotated material and converts the audio to **16 kHz mono FLAC**.\n\n"
        "It is designed to plug into the current `peacock-asr` training code, which\n"
        "can consume a Hugging Face dataset with `audio` plus `phonemes`.\n\n"
        "## Included splits\n\n"
        f"{split_block}\n\n"
        "The scripted subset uses the common 12/6/6 speaker partition:\n\n"
        "- `train`: remaining scripted speakers\n"
        "- `validation`: `MBMPS`, `NCC`, `SVBI`, `THV`, `YBAA`, `YDCK`\n"
        "- `test`: `NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA`\n"
        "- `suitcase`: spontaneous story-retelling subset from the separate suitcase corpus\n\n"
        "## Columns\n\n"
        f"- `audio`: {sample_rate} Hz mono FLAC audio\n"
        "- `phonemes`: perceived phone sequence aligned to the audio\n"
        "- `canonical_phonemes`: reference phone sequence\n"
        "- `manual_events`: full interval-level manual annotations with raw and normalized labels\n"
        "- `transcript`, `speaker_id`, `gender`, `native_language`, `subset`\n"
        "- `num_substitutions`, `num_additions`, `num_deletions`\n\n"
        "`phonemes[*].phoneme` is normalized toward the current `peacock-asr`\n"
        "39-phone vocabulary. Silence/pause intervals and opaque `err` labels are\n"
        "excluded from that training-facing sequence, but preserved in\n"
        "`manual_events`.\n\n"
        "## License\n\n"
        "This derivative remains under the original **CC BY-NC 4.0** terms from\n"
        "L2-ARCTIC. Redistribution is for non-commercial use only.\n\n"
        "## Source\n\n"
        "Derived from the upstream raw mirror:\n\n"
        "- `chikingsley/l2-arctic-release-v5.0`\n\n"
        "## Citation\n\n"
        "```bibtex\n"
        "@inproceedings{zhao2018l2arctic,\n"
        "  author={Guanlong {Zhao} and Sinem {Sonsaat} and Alif {Silpachai}\n"
        "          and Ivana {Lucic} and Evgeny {Chukharev-Hudilainen}\n"
        "          and John {Levis} and Ricardo {Gutierrez-Osuna}},\n"
        "  title={L2-ARCTIC: A Non-native English Speech Corpus},\n"
        "  year=2018,\n"
        "  booktitle={Proc. Interspeech},\n"
        "  pages={2783--2787},\n"
        "  doi={10.21437/Interspeech.2018-1110}\n"
        "}\n"
        "```\n"
    )


def upload_dataset(
    dataset_dict: DatasetDict,
    repo_id: str,
    token: str,
    private: bool,
    readme_text: str,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset_dict.push_to_hub(repo_id, token=token, private=private, max_shard_size="1GB")

    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as handle:
        handle.write(readme_text)
        temp_path = Path(handle.name)
    try:
        api.upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    validate_source_dir(args.source_dir)

    requested_speakers = {speaker.upper() for speaker in args.speakers} if args.speakers else None
    if requested_speakers is not None:
        unknown = sorted(requested_speakers - set(SPEAKER_META))
        if unknown:
            raise ValueError(f"Unknown speaker codes: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    scripted_archives = sorted(args.source_dir.glob("*.zip"))
    scripted_archives = [path for path in scripted_archives if path.stem.upper() in SPEAKER_META]

    for archive in scripted_archives:
        speaker_id = archive.stem.upper()
        if requested_speakers is not None and speaker_id not in requested_speakers:
            continue
        print(f"Processing scripted speaker {speaker_id}...")  # noqa: T201
        speaker_rows = scripted_rows_for_speaker(
            archive,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            limit_per_speaker=args.limit_per_speaker,
        )
        for split_name, rows in speaker_rows.items():
            rows_by_split[split_name].extend(rows)

    if not args.skip_suitcase:
        print("Processing suitcase subset...")  # noqa: T201
        rows_by_split["suitcase"].extend(
            suitcase_rows(
                args.source_dir / "suitcase_corpus.zip",
                output_dir=args.output_dir,
                sample_rate=args.sample_rate,
                allowed_speakers=requested_speakers,
            )
        )

    if not rows_by_split:
        raise RuntimeError("No rows were produced. Check the speaker filters and input directory.")

    dataset_dict = build_dataset_dict(rows_by_split, sample_rate=args.sample_rate)
    stats = compute_stats(dataset_dict)
    for split_name in dataset_dict:
        split_stat = stats[split_name]
        print(  # noqa: T201
            f"Prepared split {split_name}: {split_stat.rows} rows, {split_stat.duration_hours:.2f} hours"
        )

    if args.execute:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --execute")
        token = ensure_token(args.token)
        readme_text = build_readme(stats, repo_id=args.repo_id, sample_rate=args.sample_rate)
        print(f"Uploading dataset to {args.repo_id}...")  # noqa: T201
        upload_dataset(
            dataset_dict,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            readme_text=readme_text,
        )
        print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")  # noqa: T201
        return

    print("Dry run only. Dataset prepared locally but not uploaded.")  # noqa: T201


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
