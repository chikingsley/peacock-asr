"""KenLM-fused beam evaluation for the Farsi Omni CTC model."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
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
    "fleurs_dev": ROOT / "data/fleurs/dev",
    "mana_tts": ROOT / "data/mana_tts/test",
    "neyshekar": ROOT / "data/neyshekar/test",
    "neyshekar_dev": ROOT / "data/neyshekar/dev",
    "worldspeech": ROOT / "data/worldspeech/test",
    "worldspeech_dev": ROOT / "data/worldspeech/dev",
    "youtube": ROOT / "data/youtube/test",
    # Video-disjoint halves of the restored upstream YouTube test shards 0-1
    # (canonical youtube test symlink is dangling; upstream test rows were never trained on).
    "youtube_dev_conv": ROOT / "data/youtube_hf/dev_conv.parquet",
    "youtube_test_conv": ROOT / "data/youtube_hf/test_conv.parquet",
}
ORACLE_CUTOFFS = (1, 4, 8, 16)
# The omnilingual_asr inference pipeline rejects rows over 40 s.
MAX_AUDIO_SEC = 40.0


@dataclass
class DecodeResult:
    greedy_hyps: list[str]
    hyps: dict[str, list[str]]
    decode_secs: dict[str, float]
    model_secs: float
    audio_secs: float
    # per decoder: per row, list of (text, acoustic_score, combined_lm_score)
    nbest: dict[str, list[list[tuple[str, float, float]]]] = field(default_factory=dict)
    # per decoder: (total raw beams, total unique-text beams) for duplicate-rate reporting
    beam_counts: dict[str, list[int]] = field(default_factory=dict)


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
    parser.add_argument(
        "--nbest",
        type=int,
        default=0,
        help="Retain the top-N unique hypotheses per row and report oracle WER.",
    )
    parser.add_argument(
        "--logits-dir",
        type=Path,
        default=None,
        help="Cache fp16 log-probs here; a complete cache decodes without the acoustic model.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Persist plain-beam and KenLM predictions in the shared benchmark SQLite store.",
    )
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
        for alpha in (0.15, 0.3, 0.45, 0.6)
        for beta in (-0.25, 0.0, 0.25)
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


def dedupe_beams(beams: list[Any], limit: int) -> tuple[list[tuple[str, float, float]], int]:
    """Collapse output beams to unique texts (best-scored first); return (entries, raw count)."""
    seen: set[str] = set()
    entries: list[tuple[str, float, float]] = []
    for beam in beams:
        text = beam[0]
        if text in seen:
            continue
        seen.add(text)
        entries.append((text, float(beam[-2]), float(beam[-1])))
    return entries[:limit], len(beams)


def make_decoders(
    grids: list[tuple[str, str | None, float, float]],
    labels: list[str],
    unigrams: list[str],
    build_ctcdecoder: Any,
) -> list[tuple[str, Any]]:
    return [
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


def decode_chunk(
    args: argparse.Namespace,
    chunk_log_probs: list[Any],
    decoders: list[tuple[str, Any]],
    pool: Any,
    result: DecodeResult,
) -> None:
    for name, decoder in decoders:
        decode_started = time.monotonic()
        if args.nbest > 0:
            beams_lists = decoder.decode_beams_batch(
                pool, chunk_log_probs, beam_width=args.beam_width
            )
            for beams in beams_lists:
                entries, raw = dedupe_beams(beams, args.nbest)
                result.nbest[name].append(entries)
                result.hyps[name].append(entries[0][0] if entries else "")
                counts = result.beam_counts[name]
                counts[0] += raw
                counts[1] += len({beam[0] for beam in beams})
        else:
            result.hyps[name].extend(
                decoder.decode_batch(pool, chunk_log_probs, beam_width=args.beam_width),
            )
        result.decode_secs[name] += time.monotonic() - decode_started


def new_decode_result(grids: list[tuple[str, str | None, float, float]]) -> DecodeResult:
    names = [name for name, *_ in grids]
    return DecodeResult(
        greedy_hyps=[],
        hyps={name: [] for name in names},
        decode_secs=dict.fromkeys(names, 0.0),
        model_secs=0.0,
        audio_secs=0.0,
        nbest={name: [] for name in names},
        beam_counts={name: [0, 0] for name in names},
    )


class LogitsCache:
    """fp16 log-prob shards plus row metadata so sweeps never rerun the acoustic model."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.meta_path = root / "meta.json"

    def load_meta(self) -> dict[str, Any] | None:
        if not self.meta_path.exists():
            return None
        with self.meta_path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def is_complete(self) -> bool:
        meta = self.load_meta()
        if meta is None:
            return False
        return all((self.root / name).exists() for name in meta["chunks"])

    def validate(self, meta: dict[str, Any], benchmark_name: str, args: argparse.Namespace) -> None:
        model_sha = file_sha256_prefix(ROOT / "data/benchmarks/model/model.pt")
        checks = {
            "benchmark": (meta["benchmark"], benchmark_name),
            "model_sha": (meta["model_sha"], model_sha),
            "limit": (meta["limit"], args.limit),
        }
        for key, (cached, wanted) in checks.items():
            if cached != wanted:
                raise SystemExit(
                    f"logits cache mismatch on {key}: cache has {cached!r}, run wants {wanted!r}; "
                    f"use a different --logits-dir",
                )

    def chunk_name(self, start: int) -> str:
        return f"chunk_{start:06d}.npz"

    def write_chunk(self, start: int, log_probs: list[Any]) -> None:
        import numpy as np

        self.root.mkdir(parents=True, exist_ok=True)
        arrays = {str(i): array for i, array in enumerate(log_probs)}
        np.savez(self.root / self.chunk_name(start), **arrays)

    def read_chunk(self, name: str) -> list[Any]:
        import numpy as np

        with np.load(self.root / name) as data:
            return [data[str(i)] for i in range(len(data.files))]

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("w", encoding="utf-8") as stream:
            json.dump(meta, stream, ensure_ascii=False)


def decode_rows(
    args: argparse.Namespace,
    audio: list[bytes],
    pipe: Any,
    grids: list[tuple[str, str | None, float, float]],
    unigrams: list[str],
    sp: Any,
    build_ctcdecoder: Any,
    cache: LogitsCache | None,
) -> DecodeResult:
    import numpy as np

    captured: list[Any] = []
    install_logprob_capture(pipe, captured)
    result = new_decode_result(grids)
    decoders = None
    pool = None
    total_frames = 0
    started = time.monotonic()

    try:
        for start in range(0, len(audio), args.chunk_rows):
            chunk = audio[start : start + args.chunk_rows]
            captured.clear()
            model_started = time.monotonic()
            result.greedy_hyps.extend(
                pipe.transcribe(
                    chunk,
                    lang=[LANGUAGE] * len(chunk),
                    batch_size=args.batch_size,
                ),
            )
            result.model_secs += time.monotonic() - model_started
            total_frames += sum(log_probs.shape[0] for log_probs in captured)
            if cache is not None:
                cache.write_chunk(start, captured)

            if decoders is None:
                labels, _ = build_labels(sp, captured)
                decoders = make_decoders(grids, labels, unigrams, build_ctcdecoder)
                pool = mp.get_context("fork").Pool(args.decoder_workers)

            chunk_log_probs = [log_probs.astype(np.float32) for log_probs in captured]
            decode_chunk(args, chunk_log_probs, decoders, pool, result)

            done = min(start + args.chunk_rows, len(audio))
            rate = done / max(time.monotonic() - started, 1e-9)
            eta = (len(audio) - done) / max(rate, 1e-9)
            print(f"  {done}/{len(audio)} ({rate:.1f} rows/s, ETA {eta / 60:.0f} min)", flush=True)
    finally:
        if pool is not None:
            pool.terminate()

    result.audio_secs = total_frames * 0.02
    return result


def decode_rows_cached(
    args: argparse.Namespace,
    cache: LogitsCache,
    meta: dict[str, Any],
    grids: list[tuple[str, str | None, float, float]],
    unigrams: list[str],
    sp: Any,
    build_ctcdecoder: Any,
) -> DecodeResult:
    import numpy as np

    result = new_decode_result(grids)
    result.greedy_hyps = list(meta["greedy_hyps"])
    result.model_secs = meta["model_secs"]
    result.audio_secs = meta["audio_secs"]
    decoders = None
    pool = None
    total = meta["rows"]
    done = 0
    started = time.monotonic()

    try:
        for name in meta["chunks"]:
            captured = cache.read_chunk(name)
            if decoders is None:
                labels, _ = build_labels(sp, captured)
                decoders = make_decoders(grids, labels, unigrams, build_ctcdecoder)
                pool = mp.get_context("fork").Pool(args.decoder_workers)

            chunk_log_probs = [log_probs.astype(np.float32) for log_probs in captured]
            decode_chunk(args, chunk_log_probs, decoders, pool, result)

            done += len(captured)
            rate = done / max(time.monotonic() - started, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(f"  {done}/{total} ({rate:.1f} rows/s, ETA {eta / 60:.0f} min)", flush=True)
    finally:
        if pool is not None:
            pool.terminate()

    return result


def print_results(
    benchmark_name: str,
    refs_norm: list[str],
    result: DecodeResult,
    nbest: int,
) -> None:
    print(f"\n=== RESULTS ({benchmark_name}) ===", flush=True)
    print(
        f"logit generation (model forward+greedy): {result.model_secs:.1f}s "
        f"for ~{result.audio_secs / 3600:.2f}h audio",
        flush=True,
    )
    print_score("greedy (production)", refs_norm, result.greedy_hyps, 0.0, result.audio_secs)
    for name, hyps in result.hyps.items():
        print_score(name, refs_norm, hyps, result.decode_secs[name], result.audio_secs)
    if nbest > 0:
        for name, rows in result.nbest.items():
            if any(rows):
                print_nbest_report(name, refs_norm, rows, result.beam_counts[name], nbest)
    print("LM_RUN_DONE", flush=True)


def print_nbest_report(
    name: str,
    refs_norm: list[str],
    rows: list[list[tuple[str, float, float]]],
    beam_counts: list[int],
    nbest: int,
) -> None:
    from omni_curator.process import normalize

    raw, unique = beam_counts
    dup_rate = 100.0 * (1.0 - unique / raw) if raw else 0.0
    mean_unique = sum(len(row) for row in rows) / max(len(rows), 1)
    print(
        f"\n--- N-best report: {name} (kept top {nbest}) ---\n"
        f"raw beams {raw}, unique-text {unique} (duplicate rate {dup_rate:.1f}%), "
        f"mean kept candidates/row {mean_unique:.1f}",
        flush=True,
    )

    cutoffs = [cutoff for cutoff in ORACLE_CUTOFFS if cutoff <= nbest]
    totals = dict.fromkeys(cutoffs, 0)
    ref_words = 0
    for ref, candidates in zip(refs_norm, rows, strict=True):
        ref_norm = normalize(ref, LANGUAGE)
        if not ref_norm.strip():
            continue
        words = len(ref_norm.split())
        ref_words += words
        errors = [
            _word_errors(ref_norm, normalize(text, LANGUAGE))
            for text, _, _ in candidates or [("", 0.0, 0.0)]
        ]
        for cutoff in cutoffs:
            best = min(errors[:cutoff]) if errors[:cutoff] else words
            totals[cutoff] += best
    for cutoff in cutoffs:
        oracle = 100.0 * totals[cutoff] / max(ref_words, 1)
        label = "1-best" if cutoff == 1 else f"oracle@{cutoff}"
        print(f"{label:<12} WER {oracle:6.2f}%", flush=True)


def _word_errors(ref: str, hyp: str) -> int:
    import jiwer

    if not hyp.strip():
        return len(ref.split())
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions


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
    benchmark_name, benchmark_paths = resolve_benchmark(args)
    assert_tokenizer_matches_model(args.tokenizer_path)

    cache = LogitsCache(args.logits_dir) if args.logits_dir is not None else None
    if cache is not None and cache.is_complete():
        return run_eval_cached(args, benchmark_name, benchmark_paths, cache)
    return run_eval_live(args, benchmark_name, benchmark_paths, cache)


def run_eval_live(
    args: argparse.Namespace,
    benchmark_name: str,
    benchmark_paths: list[Path],
    cache: LogitsCache | None,
) -> int:
    import io

    import sentencepiece as spm
    import soundfile as sf
    import torch
    from omni_curator.process import normalize
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from pyctcdecode import build_ctcdecoder

    print_preamble(benchmark_name, benchmark_paths, args.model_card)

    torch.set_num_threads(16)
    audio, refs = load_rows(benchmark_paths, args.limit)
    durations = [float(sf.info(io.BytesIO(item)).duration) for item in audio]
    kept = [i for i, duration in enumerate(durations) if duration <= MAX_AUDIO_SEC]
    if len(kept) < len(audio):
        print(f"dropped {len(audio) - len(kept)} rows over the {MAX_AUDIO_SEC}s cap", flush=True)
        audio = [audio[i] for i in kept]
        refs = [refs[i] for i in kept]
        durations = [durations[i] for i in kept]
    refs_norm = [normalize(ref, LANGUAGE) for ref in refs]
    print(f"rows: {len(audio)}", flush=True)

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    pipe = ASRInferencePipeline(args.model_card, device=args.device, dtype=dtype)
    unigrams = load_unigrams(args.corpus_path)
    print(f"unigrams: {len(unigrams)}", flush=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer_path))
    result = decode_rows(
        args, audio, pipe, decoder_grid(args), unigrams, sp, build_ctcdecoder, cache
    )
    audio_seconds = durations

    if cache is not None:
        cache.write_meta(
            {
                "benchmark": benchmark_name,
                "model_sha": file_sha256_prefix(ROOT / "data/benchmarks/model/model.pt"),
                "limit": args.limit,
                "rows": len(audio),
                "chunks": [
                    cache.chunk_name(start) for start in range(0, len(audio), args.chunk_rows)
                ],
                "greedy_hyps": result.greedy_hyps,
                "refs": refs,
                "audio_seconds": audio_seconds,
                "model_secs": result.model_secs,
                "audio_secs": result.audio_secs,
            },
        )
        print(f"logits cached -> {cache.root}", flush=True)

    print_results(benchmark_name, refs_norm, result, args.nbest)
    if args.database is not None:
        persist_results(args, benchmark_name, benchmark_paths[0], audio_seconds, refs, result)
    return 0


def run_eval_cached(
    args: argparse.Namespace,
    benchmark_name: str,
    benchmark_paths: list[Path],
    cache: LogitsCache,
) -> int:
    import sentencepiece as spm
    from omni_curator.process import normalize
    from pyctcdecode import build_ctcdecoder

    meta = cache.load_meta()
    if meta is None:
        raise SystemExit(f"logits cache metadata missing: {cache.meta_path}")
    cache.validate(meta, benchmark_name, args)
    print_preamble(benchmark_name, benchmark_paths, args.model_card)
    print(f"decoding from logits cache: {cache.root} ({meta['rows']} rows)", flush=True)

    refs = meta["refs"]
    refs_norm = [normalize(ref, LANGUAGE) for ref in refs]
    unigrams = load_unigrams(args.corpus_path)
    print(f"unigrams: {len(unigrams)}", flush=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer_path))
    result = decode_rows_cached(
        args, cache, meta, decoder_grid(args), unigrams, sp, build_ctcdecoder
    )
    print_results(benchmark_name, refs_norm, result, args.nbest)
    if args.database is not None:
        persist_results(
            args, benchmark_name, benchmark_paths[0], meta["audio_seconds"], refs, result
        )
    return 0


def persist_results(
    args: argparse.Namespace,
    benchmark_name: str,
    benchmark_path: Path,
    audio_seconds: list[float],
    refs: list[str],
    result: DecodeResult,
) -> None:
    """Persist decoder variants beside the shared greedy/model-family benchmark runs."""
    from asr_benchmark_core.store import BenchmarkStore, NBestCandidate, Prediction

    model_path = ROOT / "data/benchmarks/model/model.pt"
    lm_name = f"beam+LM a={args.alpha} b={args.beta}"
    suffix = f"-nb{args.nbest}" if args.nbest else ""
    base = f"omni-ctc-300m-farsi-{benchmark_name}"
    variants = [
        (f"{base}-beam{args.beam_width}{suffix}", "beam", "beam (no LM)", None),
        (
            f"{base}-kenlm-a{args.alpha}-b{args.beta}-beam{args.beam_width}{suffix}",
            "kenlm",
            lm_name,
            args.lm_path,
        ),
    ]
    store = BenchmarkStore(args.database)
    try:
        for run_id, decoder, variant_name, lm_path in variants:
            hypotheses = result.hyps[variant_name]
            decode_seconds = result.decode_secs[variant_name]
            config = {
                "device": args.device,
                "batch_size": args.batch_size,
                "chunk_rows": args.chunk_rows,
                "decoder": decoder,
                "beam_width": args.beam_width,
                "decoder_workers": args.decoder_workers,
                "nbest": args.nbest,
                "alpha": args.alpha if lm_path else 0.0,
                "beta": args.beta if lm_path else 0.0,
                "lm_path": str(lm_path.resolve()) if lm_path else None,
                "tokenizer_path": str(args.tokenizer_path.resolve()),
            }
            store.ensure_run(
                run_id=run_id,
                adapter="omni",
                model_path=model_path,
                benchmark_path=benchmark_path,
                language=LANGUAGE,
                config=config,
            )
            seconds_per_row = (result.model_secs + decode_seconds) / len(hypotheses)
            store.add_predictions(
                run_id,
                [
                    Prediction(
                        row_index=row_index,
                        reference=reference,
                        hypothesis=hypothesis,
                        audio_seconds=duration,
                        inference_seconds=seconds_per_row,
                    )
                    for row_index, (reference, hypothesis, duration) in enumerate(
                        zip(refs, hypotheses, audio_seconds, strict=True)
                    )
                ],
            )
            if args.nbest > 0:
                candidates = [
                    NBestCandidate(
                        row_index=row_index,
                        rank=rank,
                        hypothesis=text,
                        acoustic_score=acoustic,
                        lm_score=lm_score,
                    )
                    for row_index, row in enumerate(result.nbest[variant_name])
                    for rank, (text, acoustic, lm_score) in enumerate(row)
                ]
                store.add_nbest(run_id, candidates)
                print(f"persisted {run_id}: {len(candidates)} n-best rows", flush=True)
            print(f"persisted {run_id}: {len(hypotheses)} rows -> {args.database}", flush=True)
    finally:
        store.close()


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
