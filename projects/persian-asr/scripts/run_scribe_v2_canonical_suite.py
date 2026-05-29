from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jiwer
import pyarrow.parquet as pq
import soundfile as sf

from persian_omnilingual_asr.benchmarks.asr import clean_waveform, read_audio_bytes
from persian_omnilingual_asr.text_normalization import normalize_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = [
    "common_voice_25",
    "fleurs",
    "mana_tts",
    "neyshekar",
    "worldspeech",
    "youtube",
]


@dataclass(frozen=True)
class Sample:
    index: int
    id: str
    dataset: str
    split: str
    reference: str
    audio_path: str
    duration_seconds: float


@dataclass(frozen=True)
class SampleResult:
    index: int
    id: str
    dataset: str
    split: str
    reference: str
    hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    wer: float
    cer: float
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Scribe v2 on canonical Persian ASR tests.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--suite-name", default="canonical-tests-scribe-v2-20260525")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmarks" / "suites")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--key-file", type=Path, default=ROOT / "runs/secrets/elevenlabs-scribe.key")
    parser.add_argument("--model", default="scribe-v2")
    parser.add_argument("--language", default="fas", help="ElevenLabs language code hint.")
    parser.add_argument("--skip-transcribe", action="store_true")
    return parser.parse_args()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def iter_canonical_rows(canonical_path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in sorted(canonical_path.rglob("*.parquet")):
        parquet = pq.ParquetFile(file_path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                rows.append(dict(row))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def export_audio(dataset: str, split: str, rows: list[dict[str, Any]], output_dir: Path) -> list[Sample]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        waveform, sample_rate = read_audio_bytes(
            bytes(row["audio_bytes"]),
            str(row.get("audio_format") or ""),
        )
        waveform = clean_waveform(waveform)
        duration = float(row.get("duration_seconds") or len(waveform) / sample_rate)
        sample_id = str(row["sample_id"])
        audio_path = audio_dir / f"{index:08d}-{safe_id(sample_id)}.flac"
        if not audio_path.exists():
            sf.write(audio_path, waveform, sample_rate, format="FLAC")
        samples.append(
            Sample(
                index=index,
                id=sample_id,
                dataset=dataset,
                split=split,
                reference=str(row["text"]),
                audio_path=str(audio_path.resolve()),
                duration_seconds=duration,
            )
        )
    return samples


def write_inputs(samples: list[Sample], output_dir: Path) -> tuple[Path, Path]:
    metadata_path = output_dir / "metadata.jsonl"
    paths_path = output_dir / "paths.txt"
    metadata_path.write_text(
        "".join(json.dumps(asdict(sample), ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    paths_path.write_text("".join(sample.audio_path + "\n" for sample in samples), encoding="utf-8")
    return metadata_path, paths_path


def run_scribe(args: argparse.Namespace, paths_path: Path, output_dir: Path) -> None:
    key = args.key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"empty key file: {args.key_file}")
    command = [
        "uv",
        "run",
        "superwhisper-audio",
        "--paths-file",
        str(paths_path),
        "--jsonl",
        str(output_dir / "scribev2.jsonl"),
        "--fail-jsonl",
        str(output_dir / "scribev2.failures.jsonl"),
        "--model",
        args.model,
        "--language",
        args.language,
        "--max-workers",
        str(args.workers),
        "--key",
        key,
    ]
    log_path = output_dir / "scribev2.run.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%F %T %Z')}] {' '.join(command[:-1])} <key>\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_sample(sample: Sample, hypothesis: str) -> SampleResult:
    normalized_reference = normalize_text(sample.reference)
    normalized_hypothesis = normalize_text(hypothesis)
    return SampleResult(
        index=sample.index,
        id=sample.id,
        dataset=sample.dataset,
        split=sample.split,
        reference=sample.reference,
        hypothesis=hypothesis,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        wer=jiwer.wer(normalized_reference, normalized_hypothesis),
        cer=float(jiwer.cer(normalized_reference, normalized_hypothesis)),
        duration_seconds=sample.duration_seconds,
    )


def write_summary(
    *,
    output_dir: Path,
    dataset: str,
    split: str,
    results: list[SampleResult],
    elapsed_seconds: float,
) -> dict[str, Any]:
    refs = [result.normalized_reference for result in results]
    hyps = [result.normalized_hypothesis for result in results]
    word_output = jiwer.process_words(refs, hyps)
    char_output = jiwer.process_characters(refs, hyps)
    audio_seconds = sum(result.duration_seconds for result in results)
    rtf = elapsed_seconds / audio_seconds if audio_seconds else None
    rtfx = audio_seconds / elapsed_seconds if elapsed_seconds else None
    payload = {
        "source": "canonical",
        "dataset": dataset,
        "split": split,
        "model": "scribe-v2",
        "samples": len(results),
        "wer": word_output.wer,
        "wer_percent": word_output.wer * 100,
        "cer": char_output.cer,
        "cer_percent": char_output.cer * 100,
        "substitutions": word_output.substitutions,
        "deletions": word_output.deletions,
        "insertions": word_output.insertions,
        "hits": word_output.hits,
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed_seconds,
        "rtf": rtf,
        "rtfx": rtfx,
    }
    (output_dir / "samples.jsonl").write_text(
        "".join(json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    lines = [
        "# Persian ASR Benchmark Summary",
        "",
        f"- Source: `{payload['source']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Split: `{payload['split']}`",
        f"- Model: `{payload['model']}`",
        f"- Samples: `{payload['samples']}`",
        f"- Average WER: `{payload['wer_percent']:.2f}%` (`{payload['wer']:.4f}`)",
        f"- Average CER: `{payload['cer_percent']:.2f}%` (`{payload['cer']:.4f}`)",
        f"- Audio seconds: `{payload['audio_seconds']:.2f}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']:.2f}`",
        f"- RTF: `{payload['rtf']:.4f}`" if rtf is not None else "- RTF: `n/a`",
        f"- RTFx: `{payload['rtfx']:.2f}x`" if rtfx is not None else "- RTFx: `n/a`",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def score_dataset(dataset_dir: Path, samples: list[Sample]) -> tuple[list[SampleResult], float]:
    started_path = dataset_dir / "scribev2.started.monotonic"
    if started_path.exists():
        started = float(started_path.read_text(encoding="utf-8"))
    else:
        started = time.monotonic()
        started_path.write_text(str(started), encoding="utf-8")
    transcripts = {
        str(row.get("audio_path")): str(row.get("transcript") or "")
        for row in read_jsonl(dataset_dir / "scribev2.jsonl")
        if not row.get("error")
    }
    missing = [sample.audio_path for sample in samples if sample.audio_path not in transcripts]
    if missing:
        raise RuntimeError(f"{dataset_dir} missing {len(missing)} transcript(s); first: {missing[0]}")
    return [score_sample(sample, transcripts[sample.audio_path]) for sample in samples], time.monotonic() - started


def write_suite_summary(suite_root: Path, payloads: list[dict[str, Any]]) -> None:
    rows = []
    for payload in payloads:
        rows.append(
            {
                "model_name": "scribe-v2",
                "dataset": payload["dataset"],
                "split": payload["split"],
                "wer_percent": payload["wer_percent"],
                "cer_percent": payload["cer_percent"],
                "samples": payload["samples"],
                "audio_hours": payload["audio_seconds"] / 3600,
                "elapsed_seconds": payload["elapsed_seconds"],
                "rtfx": payload["rtfx"],
            }
        )
    (suite_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    header = ["model", "dataset", "split", "wer_percent", "cer_percent", "samples", "audio_hours", "rtfx"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["model_name"]),
                    str(row["dataset"]),
                    str(row["split"]),
                    f"{row['wer_percent']:.4f}",
                    f"{row['cer_percent']:.4f}",
                    str(row["samples"]),
                    f"{row['audio_hours']:.4f}",
                    f"{row['rtfx']:.4f}" if row["rtfx"] is not None else "",
                ]
            )
        )
    (suite_root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_lines = [
        "| Model | Dataset | Split | WER | CER | Samples | Audio h | RTFx |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rtfx = f"{row['rtfx']:.2f}x" if row["rtfx"] is not None else ""
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["dataset"]),
                    str(row["split"]),
                    f"{row['wer_percent']:.2f}%",
                    f"{row['cer_percent']:.2f}%",
                    str(row["samples"]),
                    f"{row['audio_hours']:.2f}",
                    rtfx,
                ]
            )
            + " |"
        )
    (suite_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    suite_root = args.output_root / args.suite_name
    suite_root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    combined_results: list[SampleResult] = []
    combined_started = time.monotonic()
    for dataset in args.datasets:
        canonical_path = args.data_root / "canonical" / dataset / args.split
        if not canonical_path.exists():
            raise FileNotFoundError(canonical_path)
        dataset_dir = suite_root / "scribe-v2" / f"{dataset}-{args.split}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = iter_canonical_rows(canonical_path, args.limit)
        samples = export_audio(dataset, args.split, rows, dataset_dir)
        _, paths_path = write_inputs(samples, dataset_dir)
        if not args.skip_transcribe:
            run_scribe(args, paths_path, dataset_dir)
        results, elapsed = score_dataset(dataset_dir, samples)
        combined_results.extend(results)
        payloads.append(
            write_summary(
                output_dir=dataset_dir,
                dataset=dataset,
                split=args.split,
                results=results,
                elapsed_seconds=elapsed,
            )
        )
        write_suite_summary(suite_root, payloads)
    combined_dir = suite_root / "scribe-v2" / f"combined-{args.split}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    payloads.append(
        write_summary(
            output_dir=combined_dir,
            dataset="combined",
            split=args.split,
            results=combined_results,
            elapsed_seconds=time.monotonic() - combined_started,
        )
    )
    write_suite_summary(suite_root, payloads)
    print(f"wrote {suite_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
