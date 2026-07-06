"""KenLM-fused beam evaluation for the Farsi Omni CTC model."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from farsi_asr import LANGUAGE, ROOT

DEFAULT_MODEL_CARD = "omni_ctc_300m_farsi_hf"
DEFAULT_LM_PATH = ROOT / "experiments/lm_decoding/lm4.bin"
DEFAULT_CORPUS_PATH = ROOT / "experiments/lm_decoding/corpus.txt"
DEFAULT_TOKENIZER_PATH = ROOT.parents[1] / "base_models/omni/omniASR_tokenizer_written_v2.model"
BENCHMARKS = {
    "c1tech": ROOT / "data/benchmarks/data/data/train-00000-of-00001.parquet",
    "common_voice_25": ROOT / "data/common_voice_25/test",
    "fleurs": ROOT / "data/fleurs/test",
    "mana_tts": ROOT / "data/mana_tts/test",
    "neyshekar": ROOT / "data/neyshekar/test",
    "worldspeech": ROOT / "data/worldspeech/test",
    "youtube": ROOT / "data/youtube/test",
}


@dataclass
class DecodeResult:
    greedy_hyps: list[str]
    hyps: dict[str, list[str]]
    decode_secs: dict[str, float]
    model_secs: float
    audio_secs: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate greedy CTC, plain beam, and KenLM-fused beam on Farsi Omni CTC.",
    )
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="c1tech")
    parser.add_argument("--bench", type=Path, default=None, help="Explicit benchmark parquet path.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-card", default=DEFAULT_MODEL_CARD)
    parser.add_argument("--lm-path", type=Path, default=DEFAULT_LM_PATH)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--alpha", type=float, default=0.3, help="KenLM weight.")
    parser.add_argument("--beta", type=float, default=0.0, help="Word insertion bonus.")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=100)
    parser.add_argument("--decoder-workers", type=int, default=8)
    parser.add_argument("--sweep", action="store_true", help="Evaluate the fixed alpha/beta grid.")
    parser.add_argument("--device", default="cuda")
    return parser


def resolve_benchmark(args: argparse.Namespace) -> tuple[str, list[Path]]:
    if args.bench is not None:
        paths = expand_parquet_paths(args.bench)
        return (args.bench.stem, paths)
    path = BENCHMARKS[args.benchmark]
    return (args.benchmark, expand_parquet_paths(path))


def expand_parquet_paths(path: Path) -> list[Path]:
    raw_path = str(path)
    if any(char in raw_path for char in "*?[]"):
        paths = list(path.parent.glob(path.name))
    elif path.is_dir():
        paths = sorted(path.glob("*.parquet"))
    else:
        paths = [path]

    paths = sorted(candidate for candidate in paths if candidate.exists())
    if not paths:
        raise SystemExit(f"no parquet files found for benchmark path: {path}")
    return paths


def file_sha256_prefix(path: Path, *, length: int = 12) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


def _parquet_row_batches(parquet: Any, path: Path) -> Iterator[tuple[list[bytes], list[str]]]:
    schema_names = set(parquet.schema_arrow.names)
    if {"audio_bytes", "normalized_text"}.issubset(schema_names):
        for batch in parquet.iter_batches(
            batch_size=256, columns=["audio_bytes", "normalized_text"]
        ):
            yield (
                batch.column("audio_bytes").to_pylist(),
                batch.column("normalized_text").to_pylist(),
            )
        return
    if {"audio", "transcription"}.issubset(schema_names):
        for batch in parquet.iter_batches(batch_size=256, columns=["audio", "transcription"]):
            yield (
                [row["bytes"] for row in batch.column("audio").to_pylist()],
                batch.column("transcription").to_pylist(),
            )
        return
    raise SystemExit(f"unsupported benchmark schema in {path}: {sorted(schema_names)}")


def load_rows(paths: list[Path], limit: int) -> tuple[list[bytes], list[str]]:
    import pyarrow.parquet as pq

    audio: list[bytes] = []
    refs: list[str] = []
    remaining = limit or None

    for path in paths:
        parquet = pq.ParquetFile(path)
        for shard_audio, shard_refs in _parquet_row_batches(parquet, path):
            batch_audio = shard_audio
            batch_refs = shard_refs
            if remaining is not None:
                batch_audio = shard_audio[:remaining]
                batch_refs = shard_refs[:remaining]
                remaining -= len(batch_audio)
            audio.extend(batch_audio)
            refs.extend(batch_refs)
            if remaining == 0:
                break

        if remaining == 0:
            break

    return audio, refs


def load_unigrams(corpus_path: Path) -> list[str]:
    with corpus_path.open(encoding="utf-8") as stream:
        return sorted({word for line in stream for word in line.split()})


def assert_tokenizer_matches_model(tokenizer_path: Path) -> None:
    bundled = ROOT / "data/benchmarks/model/omniASR_tokenizer_written_v2.model"
    if bundled.exists() and file_sha256_prefix(tokenizer_path) != file_sha256_prefix(bundled):
        raise SystemExit("tokenizer mismatch: requested tokenizer differs from model bundle")


def build_labels(sp: Any, captured: list[Any]) -> tuple[list[str], int]:
    import numpy as np

    labels = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    counts = np.zeros(len(labels), dtype=np.int64)
    for log_probs in captured[:200]:
        idx, count = np.unique(log_probs.argmax(axis=-1), return_counts=True)
        counts[idx] += count

    blank = int(counts.argmax())
    print(f"blank index: {blank} (piece {labels[blank]!r})", flush=True)
    if blank != 0:
        raise SystemExit(f"inferred CTC blank {blank} != expected 0")

    active_multi = [
        (i, piece, int(counts[i]))
        for i, piece in enumerate(labels)
        if i != blank and len(piece) > 1 and counts[i] > 0
    ]
    if active_multi:
        preview = ", ".join(f"{i}:{piece!r}x{count}" for i, piece, count in active_multi[:8])
        raise SystemExit(
            f"active multi-character tokenizer pieces would corrupt pyctcdecode: {preview}",
        )

    normalized_labels = [
        chr(0xE000 + i) if len(piece) > 1 else piece for i, piece in enumerate(labels)
    ]
    normalized_labels[blank] = ""
    return normalized_labels, blank


def decoder_grid(args: argparse.Namespace) -> list[tuple[str, str | None, float, float]]:
    if not args.sweep:
        return [
            ("beam (no LM)", None, 0.0, 0.0),
            (f"beam+LM a={args.alpha} b={args.beta}", str(args.lm_path), args.alpha, args.beta),
        ]
    return [("beam (no LM)", None, 0.0, 0.0)] + [
        (f"beam+LM a={alpha} b={beta}", str(args.lm_path), alpha, beta)
        for alpha in (0.3, 0.5, 0.7, 1.0)
        for beta in (0.0, 1.5, 3.0)
    ]


def install_logprob_capture(pipe: Any, captured: list[Any]) -> None:
    import torch
    from fairseq2.nn import BatchLayout

    def apply_and_capture(batch: Any) -> list[str]:
        layout = BatchLayout(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )
        logits, batch_layout = pipe.model(batch.source_seqs, layout)
        texts = []
        for row_idx in range(logits.shape[0]):
            seq_len = batch_layout.seq_lens[row_idx]
            log_probs = torch.log_softmax(logits[row_idx, :seq_len].float(), dim=-1)
            captured.append(log_probs.to(torch.float16).cpu().numpy())
            seq = logits[row_idx, :seq_len].argmax(dim=-1)
            mask = torch.ones(seq.shape[0], dtype=torch.bool)
            mask[1:] = seq[1:] != seq[:-1]
            texts.append(pipe.token_decoder(seq[mask]))
        return texts

    pipe._apply_model_wav2vec2asr = apply_and_capture  # noqa: SLF001


def decode_rows(
    args: argparse.Namespace,
    audio: list[bytes],
    pipe: Any,
    grids: list[tuple[str, str | None, float, float]],
    unigrams: list[str],
    sp: Any,
    build_ctcdecoder: Any,
) -> DecodeResult:
    import numpy as np

    captured: list[Any] = []
    install_logprob_capture(pipe, captured)
    greedy_hyps: list[str] = []
    hyps = {name: [] for name, *_ in grids}
    decode_secs = dict.fromkeys(hyps, 0.0)
    decoders = None
    pool = None
    total_frames = 0
    model_secs = 0.0
    started = time.monotonic()

    try:
        for start in range(0, len(audio), args.chunk_rows):
            chunk = audio[start : start + args.chunk_rows]
            captured.clear()
            model_started = time.monotonic()
            greedy_hyps.extend(
                pipe.transcribe(
                    chunk,
                    lang=[LANGUAGE] * len(chunk),
                    batch_size=args.batch_size,
                ),
            )
            model_secs += time.monotonic() - model_started
            total_frames += sum(log_probs.shape[0] for log_probs in captured)

            if decoders is None:
                labels, _ = build_labels(sp, captured)
                decoders = [
                    (
                        name,
                        build_ctcdecoder(
                            labels,
                            kenlm_model_path=lm_path,
                            unigrams=unigrams if lm_path else None,
                            alpha=alpha,
                            beta=beta,
                        ),
                    )
                    for name, lm_path, alpha, beta in grids
                ]
                pool = mp.get_context("fork").Pool(args.decoder_workers)

            chunk_log_probs = [log_probs.astype(np.float32) for log_probs in captured]
            for name, decoder in decoders:
                decode_started = time.monotonic()
                hyps[name].extend(
                    decoder.decode_batch(pool, chunk_log_probs, beam_width=args.beam_width),
                )
                decode_secs[name] += time.monotonic() - decode_started

            done = min(start + args.chunk_rows, len(audio))
            rate = done / max(time.monotonic() - started, 1e-9)
            eta = (len(audio) - done) / max(rate, 1e-9)
            print(f"  {done}/{len(audio)} ({rate:.1f} rows/s, ETA {eta / 60:.0f} min)", flush=True)
    finally:
        if pool is not None:
            pool.terminate()

    return DecodeResult(
        greedy_hyps=greedy_hyps,
        hyps=hyps,
        decode_secs=decode_secs,
        model_secs=model_secs,
        audio_secs=total_frames * 0.02,
    )


def print_results(
    benchmark_name: str,
    refs_norm: list[str],
    result: DecodeResult,
) -> None:
    print(f"\n=== RESULTS ({benchmark_name}) ===", flush=True)
    print(
        f"model forward+greedy: {result.model_secs:.1f}s "
        f"for ~{result.audio_secs / 3600:.2f}h audio",
        flush=True,
    )
    print_score("greedy (production)", refs_norm, result.greedy_hyps, 0.0, result.audio_secs)
    for name, hyps in result.hyps.items():
        print_score(name, refs_norm, hyps, result.decode_secs[name], result.audio_secs)
    print("LM_RUN_DONE", flush=True)


def print_score(
    name: str,
    refs_norm: list[str],
    hyps: list[str],
    decode_seconds: float,
    audio_seconds: float,
) -> None:
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures

    # Reference and hypothesis must pass through the *same* normalizer (zwnj decision, rule 2);
    # the benchmark column is the raw/foreign-normalized surface, so normalize the ref here too.
    pairs = []
    for ref, hyp in zip(refs_norm, hyps, strict=True):
        ref_norm = normalize(ref, LANGUAGE)
        if ref_norm.strip():
            pairs.append((ref_norm, normalize(hyp, LANGUAGE)))
    measures = compute_measures([ref for ref, _ in pairs], [hyp for _, hyp in pairs])
    skipped = len(refs_norm) - len(pairs)
    skipped_note = f" skipped_empty_ref={skipped}" if skipped else ""
    rtfx_note = ""
    if decode_seconds > 0:
        rtfx_note = f" decode={decode_seconds:.1f}s/{audio_seconds / decode_seconds:.0f}xRT"
    print(
        f"{name:<24} WER {measures.wer:6.2f}%  CER {measures.cer:6.2f}%{rtfx_note}{skipped_note}",
        flush=True,
    )


def run_eval(args: argparse.Namespace) -> int:
    import sentencepiece as spm
    import torch
    from omni_curator.process import normalize
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from pyctcdecode import build_ctcdecoder

    benchmark_name, benchmark_paths = resolve_benchmark(args)
    assert_tokenizer_matches_model(args.tokenizer_path)
    print_preamble(benchmark_name, benchmark_paths, args.model_card)

    torch.set_num_threads(16)
    audio, refs = load_rows(benchmark_paths, args.limit)
    refs_norm = [normalize(ref, LANGUAGE) for ref in refs]
    print(f"rows: {len(audio)}", flush=True)

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    pipe = ASRInferencePipeline(args.model_card, device=args.device, dtype=dtype)
    unigrams = load_unigrams(args.corpus_path)
    print(f"unigrams: {len(unigrams)}", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    result = decode_rows(args, audio, pipe, decoder_grid(args), unigrams, sp, build_ctcdecoder)
    print_results(benchmark_name, refs_norm, result)
    return 0


def print_preamble(benchmark_name: str, benchmark_paths: list[Path], model_card: str) -> None:
    ckpt = ROOT / "data/benchmarks/model/model.pt"
    first_path = benchmark_paths[0]
    path_note = str(first_path)
    if len(benchmark_paths) > 1:
        path_note = f"{first_path} (+{len(benchmark_paths) - 1} more)"
    print(
        f"benchmark {benchmark_name}: {path_note}\n"
        f"model {model_card} ({ckpt.stat().st_size / 1e9:.2f} GB, "
        f"sha {file_sha256_prefix(ckpt)})\n"
        f"normalizer omni_curator.normalize(_, {LANGUAGE!r})",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    return run_eval(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
