"""Materialize pinned Hugging Face ASR exams as immutable local NeMo manifests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, load_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded_audio(value: object) -> tuple[bytes, str]:
    if not isinstance(value, dict):
        raise TypeError("audio row must be a mapping")
    payload = value.get("bytes")
    raw_path = value.get("path")
    if payload is None and isinstance(raw_path, str):
        payload = Path(raw_path).read_bytes()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("audio row has no encoded bytes")
    suffix = Path(str(raw_path or "audio.flac")).suffix.lower()
    return payload, suffix if suffix in {".flac", ".wav", ".mp3", ".ogg"} else ".audio"


def _duration(payload: bytes) -> float:
    info = sf.info(io.BytesIO(payload))
    return float(info.frames) / float(info.samplerate)


def _reference(row: dict[str, Any], text_field: str, index: int) -> str:
    text = str(row[text_field]).strip()
    if not text:
        raise ValueError(f"empty exam reference at row {index}")
    return text


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing exam audio differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def materialize_exam(  # noqa: PLR0913
    *,
    repo: str,
    revision: str,
    config: str,
    split: str,
    output_dir: Path,
    cache_dir: Path | None = None,
    audio_field: str = "audio",
    text_field: str = "text",
    limit: int = 0,
    verification_mode: str | None = None,
) -> dict[str, Any]:
    """Download one exact exam config/split and atomically write audio plus a manifest."""
    if not revision.strip():
        raise ValueError("a pinned revision is required")
    if output_dir.exists():
        raise FileExistsError(f"immutable exam output already exists: {output_dir}")

    load_options: dict[str, Any] = {}
    if verification_mode is not None:
        load_options["verification_mode"] = verification_mode
    dataset = load_dataset(
        repo,
        config,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        **load_options,
    ).cast_column(audio_field, Audio(decode=False))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    manifest = temporary / "manifest.jsonl"
    rows = 0
    seconds = 0.0
    try:
        with manifest.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(dataset):
                if limit > 0 and rows >= limit:
                    break
                payload, suffix = _encoded_audio(row[audio_field])
                text = _reference(row, text_field, index)
                digest = hashlib.sha256(payload).hexdigest()
                final_audio = output_dir / "audio" / digest[:2] / f"{digest}{suffix}"
                temporary_audio = temporary / "audio" / digest[:2] / f"{digest}{suffix}"
                _write_exact(temporary_audio, payload)
                duration = _duration(payload)
                handle.write(
                    json.dumps(
                        {
                            "audio_filepath": str(final_audio),
                            "text": text,
                            "duration": round(duration, 6),
                            "sample_id": f"{config}:{split}:{index}",
                            "audio_sha256": digest,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
                seconds += duration
        summary = {
            "schema_version": 1,
            "repo": repo,
            "revision": revision,
            "config": config,
            "split": split,
            "dataset_fingerprint": dataset._fingerprint,  # noqa: SLF001
            "rows": rows,
            "hours": seconds / 3600.0,
            "audio_field": audio_field,
            "text_field": text_field,
            "limit": limit,
            "verification_mode": verification_mode,
            "manifest": "manifest.jsonl",
            "manifest_sha256": _sha256(manifest),
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--verification-mode",
        choices=["all_checks", "basic_checks", "no_checks"],
        help="Override datasets split verification for a pinned repo with broken split metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = materialize_exam(
        repo=args.repo,
        revision=args.revision,
        config=args.config,
        split=args.split,
        output_dir=args.output_dir.expanduser().resolve(),
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        audio_field=args.audio_field,
        text_field=args.text_field,
        limit=args.limit,
        verification_mode=args.verification_mode,
    )
    print(
        f"materialized {summary['config']}/{summary['split']}: "
        f"{summary['rows']} rows / {summary['hours']:.2f} h -> {args.output_dir}",
        flush=True,
    )
    return 0
