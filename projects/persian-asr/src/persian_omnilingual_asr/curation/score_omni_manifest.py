from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import jiwer
import soundfile as sf
from tqdm import tqdm

from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CARD = "omni_ctc_300m_v2_persian_wer35_fastconformer_best"


@dataclass(frozen=True)
class OmniScore:
    sample_id: str
    source: str
    source_split: str
    reference: str
    hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    wer: float
    cer: float
    duration_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a NeMo-style manifest with Omni ASR.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-card", default=DEFAULT_MODEL_CARD)
    parser.add_argument("--lang", default="fas_Arab")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def read_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def audio_input(row: dict[str, Any]) -> dict[str, Any]:
    waveform, sample_rate = sf.read(str(row["audio_filepath"]), dtype="float32", always_2d=False)
    return {"waveform": waveform, "sample_rate": sample_rate}


def score_row(row: dict[str, Any], hypothesis: str) -> OmniScore:
    reference = str(row["text"])
    normalized_reference = maybe_normalize(reference) or ""
    normalized_hypothesis = maybe_normalize(hypothesis) or ""
    return OmniScore(
        sample_id=str(row["sample_id"]),
        source=str(row["source"]),
        source_split=str(row["source_split"]),
        reference=reference,
        hypothesis=hypothesis,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        wer=jiwer.wer(normalized_reference, normalized_hypothesis),
        cer=cast("float", jiwer.cer(normalized_reference, normalized_hypothesis)),
        duration_seconds=float(row["duration"]),
    )


def load_pipeline(model_card: str) -> Any:
    os.environ.setdefault("FAIRSEQ2_ASSET_DIR", str(ROOT / ".fairseq2-assets"))
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    return ASRInferencePipeline(model_card=model_card)


def write_summary(
    args: argparse.Namespace,
    scores: list[OmniScore],
    elapsed_seconds: float,
) -> None:
    refs = [score.normalized_reference for score in scores]
    hyps = [score.normalized_hypothesis for score in scores]
    word_output = jiwer.process_words(refs, hyps)
    char_output = jiwer.process_characters(refs, hyps)
    audio_seconds = sum(score.duration_seconds for score in scores)
    payload = {
        "manifest": str(args.manifest),
        "model_card": args.model_card,
        "lang": args.lang,
        "samples": len(scores),
        "wer": word_output.wer,
        "wer_percent": word_output.wer * 100,
        "cer": char_output.cer,
        "cer_percent": char_output.cer * 100,
        "audio_seconds": audio_seconds,
        "audio_hours": audio_seconds / 3600,
        "elapsed_seconds": elapsed_seconds,
        "rtf": elapsed_seconds / audio_seconds if audio_seconds else None,
        "rtfx": audio_seconds / elapsed_seconds if elapsed_seconds else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.limit)
    pipeline = load_pipeline(args.model_card)
    score_path = args.output_dir / "scores.jsonl"
    scores: list[OmniScore] = []
    start = time.monotonic()
    with score_path.open("w", encoding="utf-8") as handle:
        batches = batched(rows, args.batch_size)
        progress = tqdm(batches, desc=f"omni score {args.manifest.name}", unit="batch")
        for batch in progress:
            audio = [audio_input(row) for row in batch]
            outputs = pipeline.transcribe(
                audio,
                batch_size=args.batch_size,
                lang=[args.lang] * len(batch),
            )
            hypotheses = (
                [outputs] if isinstance(outputs, str) else [str(output) for output in outputs]
            )
            for row, hypothesis in zip(batch, hypotheses, strict=True):
                score = score_row(row, hypothesis)
                scores.append(score)
                handle.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")
                handle.flush()
    write_summary(args, scores, time.monotonic() - start)
    print(f"wrote {score_path}")
    print(f"wrote {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
