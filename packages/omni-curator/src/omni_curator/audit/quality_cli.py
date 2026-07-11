"""Additive ASR boundary and CTC-alignment audits for JSONL manifests.

The command writes scored copies beside the source data. It never deletes rows and never promotes
a threshold into an export gate. That keeps the first run suitable for a bounded, manually audited
pilot and makes every later selection decision reproducible from recorded signals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from omni_curator.audit.benchmark import normalize, score_pair
from omni_curator.audit.quality import asr_edge_mismatch

CTM_FIELD_COUNT = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603 - resolved local git executable, read-only query
        [git, "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _reservoir(path: Path, *, limit: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    selected: list[dict[str, Any]] = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seen += 1
            if len(selected) < limit:
                selected.append(row)
                continue
            replacement = rng.randrange(seen)
            if replacement < limit:
                selected[replacement] = row
    return selected, seen


def cmd_sample(args: argparse.Namespace) -> int:
    rows, seen = _reservoir(args.input, limit=args.limit, seed=args.seed)
    for index, row in enumerate(rows):
        audio = Path(str(row[args.audio_field])).expanduser()
        row[args.audio_field] = str(audio.resolve())
        row.setdefault("sample_id", f"pilot-{args.seed}-{index:06d}")
    _write_jsonl(args.output, rows)
    print(f"sampled {len(rows)} of {seen} rows -> {args.output}")
    return 0


def cmd_edge(args: argparse.Namespace) -> int:
    if (args.beginning_threshold is None) != (args.end_threshold is None):
        raise SystemExit("set both --beginning-threshold and --end-threshold, or neither")
    rows = _read_jsonl(args.input)
    beginning_values: list[float] = []
    end_values: list[float] = []
    wers: list[float] = []
    cers: list[float] = []
    flagged = 0
    for row in rows:
        reference = str(row[args.reference_field])
        hypothesis = str(row[args.hypothesis_field])
        edge = asdict(asr_edge_mismatch(reference, hypothesis))
        agreement = score_pair(reference, hypothesis)
        would_flag: bool | None = None
        if args.beginning_threshold is not None:
            would_flag = bool(
                edge["beginning_error_chars"] > args.beginning_threshold
                or edge["end_error_chars"] > args.end_threshold
            )
            flagged += int(would_flag)
        quality = dict(row.get("quality") or {})
        quality["asr_edge"] = {
            **edge,
            "draft_model": args.model_id,
            "draft_model_sha256": args.model_sha256,
            "beginning_threshold": args.beginning_threshold,
            "end_threshold": args.end_threshold,
            "would_flag": would_flag,
        }
        quality["asr_agreement"] = agreement
        row["quality"] = quality
        beginning_values.append(float(edge["beginning_error_chars"]))
        end_values.append(float(edge["end_error_chars"]))
        wers.append(cast("float", agreement["wer"]))
        cers.append(cast("float", agreement["cer"]))
    _write_jsonl(args.output, rows)
    summary = {
        "rows": len(rows),
        "beginning_error_chars": _distribution(beginning_values),
        "end_error_chars": _distribution(end_values),
        "wer": _distribution(wers),
        "cer": _distribution(cers),
        "thresholds": {
            "beginning": args.beginning_threshold,
            "end": args.end_threshold,
        },
        "would_flag": flagged if args.beginning_threshold is not None else None,
    }
    _write_json(args.summary, summary)
    print(f"scored {len(rows)} ASR pairs -> {args.output}")
    return 0


def cmd_nfa_prepare(args: argparse.Namespace) -> int:
    from omni_curator.process.normalize import normalize as normalize_for_language

    rows = _read_jsonl(args.input)
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    changed = 0
    empty = 0
    tokenizer = None
    if args.tokenizer_model is not None:
        from nemo.collections.asr.models import ASRModel

        model = ASRModel.restore_from(
            restore_path=str(args.tokenizer_model.resolve()), map_location="cpu"
        )
        tokenizer = model.tokenizer
    for row in rows:
        raw = str(row[args.reference_field])
        normalized = normalize_for_language(raw, args.language)
        if not normalized:
            empty += 1
            continue
        incompatible_words = _nfa_incompatible_words(normalized, tokenizer) if tokenizer else []
        if incompatible_words:
            item = dict(row)
            quality = dict(item.get("quality") or {})
            quality["ctc_alignment_preflight"] = {
                "status": "token_case_incompatible",
                "words": incompatible_words,
                "tokenizer_model": str(args.tokenizer_model.resolve()),
                "tokenizer_model_sha256": _sha256(args.tokenizer_model),
            }
            item["quality"] = quality
            rejected.append(item)
            continue
        changed += int(normalized != raw)
        item = dict(row)
        item[args.reference_field] = normalized
        prepared.append(item)
    _write_jsonl(args.output, prepared)
    if args.rejected_output is not None:
        _write_jsonl(args.rejected_output, rejected)
    _write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "prepared_rows": len(prepared),
            "normalization_changed": changed,
            "normalization_empty": empty,
            "token_case_incompatible": len(rejected),
            "tokenizer_model": str(args.tokenizer_model.resolve())
            if args.tokenizer_model
            else None,
            "tokenizer_model_sha256": _sha256(args.tokenizer_model)
            if args.tokenizer_model
            else None,
            "language": args.language,
            "reference_field": args.reference_field,
        },
    )
    print(f"prepared {len(prepared)} of {len(rows)} NFA rows -> {args.output}")
    return 0


def _nfa_incompatible_words(text: str, tokenizer: Any) -> list[str]:
    from nemo.collections.asr.parts.utils.aligner_utils import restore_token_case

    incompatible: list[str] = []
    for word in text.split():
        try:
            restore_token_case(word, tokenizer.text_to_tokens(word))
        except (IndexError, RuntimeError):
            incompatible.append(word)
    return incompatible


def cmd_nfa_run(args: argparse.Namespace) -> int:
    align_script = args.nemo_root / "tools" / "nemo_forced_aligner" / "align.py"
    if not align_script.is_file():
        raise SystemExit(f"NeMo Forced Aligner script missing: {align_script}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(align_script),
        f"model_path={args.model.resolve()}",
        f"manifest_filepath={args.input.resolve()}",
        f"output_dir={args.output_dir.resolve()}",
        f"batch_size={args.batch_size}",
        f"transcribe_device={args.device}",
        f"viterbi_device={args.viterbi_device}",
        "use_local_attention=false",
        "save_output_file_formats=[ctm]",
        "ctm_file_config.remove_blank_tokens=true",
    ]
    metadata = {
        "input_manifest": str(args.input.resolve()),
        "input_manifest_sha256": _sha256(args.input),
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "nfa_script": str(align_script.resolve()),
        "nfa_script_sha256": _sha256(align_script),
        "nfa_git_revision": _git_revision(args.nemo_root),
        "batch_size": args.batch_size,
        "transcribe_device": args.device,
        "viterbi_device": args.viterbi_device,
        "command": command,
    }
    _write_json(args.output_dir / "omni-quality-nfa-run.json", metadata)
    print("running NeMo Forced Aligner:", " ".join(command), flush=True)
    subprocess.run(command, check=True)  # noqa: S603 - explicit local interpreter and script
    return 0


def _ctm_span(path: Path) -> tuple[int, float | None, float | None]:
    starts: list[float] = []
    ends: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(maxsplit=4)
            if len(parts) < CTM_FIELD_COUNT:
                continue
            start = float(parts[2])
            starts.append(start)
            ends.append(start + float(parts[3]))
    if not starts:
        return 0, None, None
    return len(starts), min(starts), max(ends)


def _missing_alignment_row(
    source_row: dict[str, Any],
    *,
    reference_field: str,
    run_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    original = dict(source_row)
    quality = dict(original.get("quality") or {})
    quality["ctc_alignment"] = {
        "status": "not_aligned",
        "word_count": 0,
        "reference_word_count": len(normalize(str(original[reference_field])).split()),
        "word_coverage": 0.0,
        "first_word_start_seconds": None,
        "last_word_end_seconds": None,
        "leading_margin_seconds": None,
        "trailing_margin_seconds": None,
        "end_overrun_seconds": None,
        "aligned_span_seconds": None,
        "aligned_span_ratio": None,
        "word_ctm": None,
        "alignment_reference": None,
        "normalization_changed": None,
        "provenance": run_metadata,
    }
    original["quality"] = quality
    return original


def _summarize_aligned_row(
    aligned_row: dict[str, Any],
    source_row: dict[str, Any],
    *,
    duration_field: str,
    reference_field: str,
    run_metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any], float | None, float | None, float | None, float | None, bool]:
    original = dict(source_row)
    duration_raw = original.get(duration_field)
    duration = float(duration_raw) if duration_raw is not None else None
    # NeMo 2.7 writes ``words_level_*``; newer docs describe ``word_level_*``.
    ctm_raw = aligned_row.get("words_level_ctm_filepath") or aligned_row.get(
        "word_level_ctm_filepath"
    )
    status = "aligned"
    word_count = 0
    first_start: float | None = None
    last_end: float | None = None
    if not ctm_raw or not Path(str(ctm_raw)).is_file():
        status = "missing_ctm"
    else:
        word_count, first_start, last_end = _ctm_span(Path(str(ctm_raw)))
        if word_count == 0:
            status = "empty_ctm"
    reference_word_count = len(normalize(str(original[reference_field])).split())
    leading = first_start
    trailing = None if duration is None or last_end is None else max(0.0, duration - last_end)
    end_overrun = None if duration is None or last_end is None else max(0.0, last_end - duration)
    span = None if first_start is None or last_end is None else max(0.0, last_end - first_start)
    span_ratio = None if duration in (None, 0.0) or span is None else min(1.0, span / duration)
    word_coverage = (
        None if reference_word_count == 0 else min(1.0, word_count / reference_word_count)
    )
    quality = dict(original.get("quality") or {})
    quality["ctc_alignment"] = {
        "status": status,
        "word_count": word_count,
        "reference_word_count": reference_word_count,
        "word_coverage": word_coverage,
        "first_word_start_seconds": first_start,
        "last_word_end_seconds": last_end,
        "leading_margin_seconds": leading,
        "trailing_margin_seconds": trailing,
        "end_overrun_seconds": end_overrun,
        "aligned_span_seconds": span,
        "aligned_span_ratio": span_ratio,
        "word_ctm": str(ctm_raw) if ctm_raw else None,
        "alignment_reference": aligned_row.get("text"),
        "normalization_changed": aligned_row.get("text") != original.get(reference_field),
        "provenance": run_metadata,
    }
    original["quality"] = quality
    return original, leading, trailing, end_overrun, span_ratio, status != "aligned"


def cmd_nfa_summarize(args: argparse.Namespace) -> int:
    originals = _read_jsonl(args.input)
    by_audio = {
        str(Path(str(row[args.audio_field])).expanduser().resolve()): row for row in originals
    }
    if args.rejected_input is not None:
        for row in _read_jsonl(args.rejected_input):
            audio = str(Path(str(row[args.audio_field])).expanduser().resolve())
            by_audio[audio] = row
    aligned = _read_jsonl(args.aligned_manifest)
    run_metadata = (
        json.loads(args.run_metadata.read_text(encoding="utf-8")) if args.run_metadata else None
    )
    output_rows: list[dict[str, Any]] = []
    leading_values: list[float] = []
    trailing_values: list[float] = []
    overrun_values: list[float] = []
    span_ratios: list[float] = []
    missing = 0
    seen_audio: set[str] = set()
    for aligned_row in aligned:
        audio = str(Path(str(aligned_row["audio_filepath"])).expanduser().resolve())
        seen_audio.add(audio)
        original, leading, trailing, end_overrun, span_ratio, alignment_missing = (
            _summarize_aligned_row(
                aligned_row,
                by_audio[audio],
                duration_field=args.duration_field,
                reference_field=args.reference_field,
                run_metadata=run_metadata,
            )
        )
        missing += int(alignment_missing)
        output_rows.append(original)
        if leading is not None:
            leading_values.append(leading)
        if trailing is not None:
            trailing_values.append(trailing)
        if end_overrun is not None:
            overrun_values.append(end_overrun)
        if span_ratio is not None:
            span_ratios.append(span_ratio)
    for audio, source_row in by_audio.items():
        if audio in seen_audio:
            continue
        missing += 1
        output_rows.append(
            _missing_alignment_row(
                source_row,
                reference_field=args.reference_field,
                run_metadata=run_metadata,
            )
        )
    _write_jsonl(args.output, output_rows)
    summary = {
        "input_rows": len(originals),
        "aligned_rows": len(aligned),
        "missing_or_empty_ctm": missing,
        "leading_margin_seconds": _distribution(leading_values),
        "trailing_margin_seconds": _distribution(trailing_values),
        "end_overrun_seconds": _distribution(overrun_values),
        "aligned_span_ratio": _distribution(span_ratios),
        "provenance": run_metadata,
    }
    _write_json(args.summary, summary)
    print(f"summarized {len(output_rows)} NFA rows -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="draw a deterministic bounded JSONL pilot")
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--limit", type=int, required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--audio-field", default="audio_filepath")
    sample.set_defaults(func=cmd_sample)

    edge = subparsers.add_parser("edge", help="score ASR/reference boundary mismatches")
    edge.add_argument("--input", type=Path, required=True)
    edge.add_argument("--output", type=Path, required=True)
    edge.add_argument("--summary", type=Path, required=True)
    edge.add_argument("--reference-field", default="text")
    edge.add_argument("--hypothesis-field", default="hypothesis")
    edge.add_argument("--model-id")
    edge.add_argument("--model-sha256")
    edge.add_argument("--beginning-threshold", type=int)
    edge.add_argument("--end-threshold", type=int)
    edge.set_defaults(func=cmd_edge)

    prepare = subparsers.add_parser(
        "nfa-prepare", help="normalize a JSONL manifest onto the language/model text surface"
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--summary", type=Path, required=True)
    prepare.add_argument("--language", required=True)
    prepare.add_argument("--reference-field", default="text")
    prepare.add_argument("--tokenizer-model", type=Path)
    prepare.add_argument("--rejected-output", type=Path)
    prepare.set_defaults(func=cmd_nfa_prepare)

    nfa_run = subparsers.add_parser("nfa-run", help="run a version-matched NeMo Forced Aligner")
    nfa_run.add_argument("--input", type=Path, required=True)
    nfa_run.add_argument("--output-dir", type=Path, required=True)
    nfa_run.add_argument("--model", type=Path, required=True)
    nfa_run.add_argument("--nemo-root", type=Path, required=True)
    nfa_run.add_argument("--batch-size", type=int, default=4)
    nfa_run.add_argument("--device", default="cuda")
    nfa_run.add_argument("--viterbi-device", default="cpu")
    nfa_run.set_defaults(func=cmd_nfa_run)

    summarize = subparsers.add_parser("nfa-summarize", help="turn NFA CTMs into row metrics")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--aligned-manifest", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--summary", type=Path, required=True)
    summarize.add_argument("--audio-field", default="audio_filepath")
    summarize.add_argument("--reference-field", default="text")
    summarize.add_argument("--duration-field", default="duration")
    summarize.add_argument("--run-metadata", type=Path)
    summarize.add_argument("--rejected-input", type=Path)
    summarize.set_defaults(func=cmd_nfa_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
