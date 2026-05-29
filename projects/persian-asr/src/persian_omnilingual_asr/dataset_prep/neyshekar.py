from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


DEFAULT_SOURCE_URL = "https://zenodo.org/records/19186714"
MIN_TOKEN_LENGTH = 2
RARE_TOKEN_LIMIT = 8
RARE_NGRAM_LIMIT = 12
MAX_TOKEN_REFERENCES = 500
MAX_NGRAM_REFERENCES = 1_000
AGGREGATE_DISTANCE_COUNT = 2


@dataclass(frozen=True)
class PredictionSource:
    name: str
    path: Path


@dataclass(frozen=True)
class ReferenceGroup:
    group_id: int
    text: str
    normalized_text: str
    source_ids: list[int]


@dataclass(frozen=True)
class CandidateIndexes:
    token_index: dict[str, set[int]]
    token_counts: Counter[str]
    ngram_index: dict[str, set[int]]
    ngram_counts: Counter[str]


@dataclass(frozen=True)
class CandidateScore:
    group: ReferenceGroup
    distances: dict[str, float]
    best_distance: float
    vote_count: int
    aggregate_distance: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair Neyshekar v3 by aligning audio hypotheses to transcript text."
    )
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--audio-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Model prediction JSONL from persian-benchmark-asr. Repeat for consensus.",
    )
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--vote-cer", type=float, default=0.20)
    parser.add_argument("--max-best-cer", type=float, default=0.10)
    parser.add_argument("--min-margin", type=float, default=0.04)
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument("--write-audio-zip", action="store_true")
    parser.add_argument("--repo-id", default="Peacockery/neyshekar-v3-asr-aligned")
    return parser


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [dict(row) for row in payload]


def parse_prediction_sources(values: Sequence[str]) -> list[PredictionSource]:
    sources: list[PredictionSource] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"prediction must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name:
            raise ValueError(f"prediction name is empty in {value!r}")
        sources.append(PredictionSource(name=name, path=Path(path)))
    if not sources:
        raise ValueError("at least one --prediction is required")
    return sources


def read_predictions(source: PredictionSource) -> dict[int, str]:
    predictions: dict[int, str] = {}
    with source.path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            predictions[int(row["id"])] = str(row["hypothesis"])
    return predictions


def group_references(rows: Sequence[dict[str, Any]]) -> list[ReferenceGroup]:
    ids_by_text: dict[str, list[int]] = defaultdict(list)
    display_text: dict[str, str] = {}
    for row in rows:
        text = str(row["text"])
        normalized = maybe_normalize(text) or ""
        if not normalized:
            continue
        ids_by_text[normalized].append(int(row["id"]))
        display_text.setdefault(normalized, text)
    return [
        ReferenceGroup(
            group_id=group_id,
            text=display_text[normalized],
            normalized_text=normalized,
            source_ids=ids,
        )
        for group_id, (normalized, ids) in enumerate(ids_by_text.items())
    ]


def word_tokens(text: str) -> set[str]:
    return {token for token in text.split() if len(token) >= MIN_TOKEN_LENGTH}


def char_ngrams(text: str, width: int = 4) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) <= width:
        return {compact} if compact else set()
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def build_indexes(
    references: Sequence[ReferenceGroup],
) -> CandidateIndexes:
    token_index: dict[str, set[int]] = defaultdict(set)
    ngram_index: dict[str, set[int]] = defaultdict(set)
    token_counts: Counter[str] = Counter()
    ngram_counts: Counter[str] = Counter()
    for group in references:
        tokens = word_tokens(group.normalized_text)
        ngrams = char_ngrams(group.normalized_text)
        token_counts.update(tokens)
        ngram_counts.update(ngrams)
        for token in tokens:
            token_index[token].add(group.group_id)
        for ngram in ngrams:
            ngram_index[ngram].add(group.group_id)
    return CandidateIndexes(
        token_index=token_index,
        token_counts=token_counts,
        ngram_index=ngram_index,
        ngram_counts=ngram_counts,
    )


def rare_items(items: Iterable[str], counts: Counter[str], limit: int) -> list[str]:
    return sorted(items, key=lambda item: (counts[item], item))[:limit]


def candidate_ids_for_hypothesis(
    hypothesis: str,
    indexes: CandidateIndexes,
    max_candidates: int,
) -> Counter[int]:
    candidates: Counter[int] = Counter()
    for token in rare_items(word_tokens(hypothesis), indexes.token_counts, RARE_TOKEN_LIMIT):
        if indexes.token_counts[token] <= MAX_TOKEN_REFERENCES:
            candidates.update(indexes.token_index[token])
    for ngram in rare_items(char_ngrams(hypothesis), indexes.ngram_counts, RARE_NGRAM_LIMIT):
        if indexes.ngram_counts[ngram] <= MAX_NGRAM_REFERENCES:
            candidates.update(indexes.ngram_index[ngram])
    return Counter(dict(candidates.most_common(max_candidates)))


def normalized_distance(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return float(Levenshtein.normalized_distance(left, right))


def score_candidates(
    hypotheses: dict[str, str],
    references: Sequence[ReferenceGroup],
    candidate_ids: Iterable[int],
    vote_cer: float,
) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for group_id in candidate_ids:
        group = references[group_id]
        distances = {
            name: normalized_distance(group.normalized_text, hypothesis)
            for name, hypothesis in hypotheses.items()
        }
        ordered_distances = sorted(distances.values())
        vote_count = sum(distance <= vote_cer for distance in ordered_distances)
        aggregate_count = min(AGGREGATE_DISTANCE_COUNT, len(ordered_distances))
        aggregate = sum(ordered_distances[:aggregate_count]) / aggregate_count
        scores.append(
            CandidateScore(
                group=group,
                distances=distances,
                best_distance=ordered_distances[0],
                vote_count=vote_count,
                aggregate_distance=aggregate,
            )
        )
    return sorted(scores, key=lambda score: (score.aggregate_distance, score.best_distance))


def should_accept(
    scores: Sequence[CandidateScore],
    min_votes: int,
    max_best_cer: float,
    min_margin: float,
) -> bool:
    if not scores:
        return False
    best = scores[0]
    if best.vote_count < min_votes or best.best_distance > max_best_cer:
        return False
    if len(scores) == 1:
        return True
    margin = scores[1].aggregate_distance - best.aggregate_distance
    return margin >= min_margin


def build_zip_member_map(archive: zipfile.ZipFile) -> dict[str, str]:
    return {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}


def copy_selected_audio(
    source_zip: Path,
    target_zip: Path,
    selected_audio: Sequence[str],
) -> None:
    selected = {Path(audio).name for audio in selected_audio}
    with zipfile.ZipFile(source_zip) as source, zipfile.ZipFile(
        target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        members = build_zip_member_map(source)
        for audio_name in tqdm(sorted(selected), desc="audio zip", unit="file"):
            member = members[audio_name]
            target.writestr(f"audio/{audio_name}", source.read(member))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(
    path: Path,
    repo_id: str,
    source_rows: int,
    accepted_rows: int,
    summary: dict[str, Any],
) -> None:
    body = f"""---
license: cc0-1.0
language:
- fa
task_categories:
- automatic-speech-recognition
pretty_name: Neyshekar v3 ASR-Aligned
---

# Neyshekar v3 ASR-Aligned

This is a repaired subset of Neyshekar v3 for Persian ASR work. The public v3
archive contains real audio and real transcripts, but the downloaded
`dataset.json` filename-to-text mapping does not align for the checked samples.

This export keeps only audio clips whose transcript could be recovered by
matching multiple ASR hypotheses back to the original Neyshekar transcript pool.
It is useful as a curated ASR training/evaluation candidate set, with the
alignment ledger included for audit.

- Hub repo: `{repo_id}`
- Source: {DEFAULT_SOURCE_URL}
- Source rows checked: {source_rows}
- Accepted aligned rows: {accepted_rows}
- Acceptance rate: {summary["accepted_fraction"]:.2%}

## Files

- `dataset.json`: list of accepted rows with `id`, `audio`, `text`, `duration`.
- `audio.zip`: selected WAV files under `audio/`.
- `alignment.jsonl`: accepted rows with model hypotheses and alignment scores.
- `rejected.jsonl`: rows that did not pass the high-confidence alignment gate.
- `summary.json`: thresholds and aggregate counts.
- `verification.json`: SHA-256 checksums.

## Alignment Gate

Rows are accepted when at least `{summary["thresholds"]["min_votes"]}` model
hypotheses match the same normalized transcript within CER
`{summary["thresholds"]["vote_cer"]}`, the best CER is at most
`{summary["thresholds"]["max_best_cer"]}`, and the candidate margin is at least
`{summary["thresholds"]["min_margin"]}`.
"""
    path.write_text(body, encoding="utf-8")


def repair(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_json_rows(args.metadata_json)
    prediction_sources = parse_prediction_sources(args.prediction)
    predictions = {source.name: read_predictions(source) for source in prediction_sources}
    references = group_references(rows)
    indexes = build_indexes(references)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="align", unit="utt"):
        row_id = int(row["id"])
        hypotheses = {
            name: maybe_normalize(source_predictions[row_id]) or ""
            for name, source_predictions in predictions.items()
            if row_id in source_predictions
        }
        combined_candidates: Counter[int] = Counter()
        for hypothesis in hypotheses.values():
            combined_candidates.update(
                candidate_ids_for_hypothesis(
                    hypothesis,
                    indexes,
                    args.max_candidates,
                )
            )
        candidate_ids = [
            group_id for group_id, _ in combined_candidates.most_common(args.max_candidates)
        ]
        scores = score_candidates(hypotheses, references, candidate_ids, args.vote_cer)
        accepted = should_accept(scores, args.min_votes, args.max_best_cer, args.min_margin)
        best = scores[0] if scores else None
        output_row = {
            "id": row_id,
            "audio": row["audio"],
            "text": best.group.text if accepted and best is not None else row["text"],
            "duration": row["duration"],
        }
        audit_row = {
            **output_row,
            "accepted": accepted,
            "source_text": row["text"],
            "matched_reference_ids": best.group.source_ids if best is not None else [],
            "normalized_matched_text": best.group.normalized_text if best is not None else "",
            "hypotheses": hypotheses,
            "distances": best.distances if best is not None else {},
            "best_distance": best.best_distance if best is not None else None,
            "vote_count": best.vote_count if best is not None else 0,
            "aggregate_distance": best.aggregate_distance if best is not None else None,
            "runner_up_aggregate_distance": (
                scores[1].aggregate_distance if len(scores) > 1 else None
            ),
        }
        if accepted:
            dataset_rows.append(output_row)
            alignment_rows.append(audit_row)
        else:
            rejected_rows.append(audit_row)

    dataset_path = args.output_dir / "dataset.json"
    alignment_path = args.output_dir / "alignment.jsonl"
    rejected_path = args.output_dir / "rejected.jsonl"
    summary_path = args.output_dir / "summary.json"
    audio_zip_path = args.output_dir / "audio.zip"
    readme_path = args.output_dir / "README.md"
    verification_path = args.output_dir / "verification.json"

    write_json(dataset_path, dataset_rows)
    write_jsonl(alignment_path, alignment_rows)
    write_jsonl(rejected_path, rejected_rows)

    if args.write_audio_zip:
        copy_selected_audio(
            args.audio_zip,
            audio_zip_path,
            [str(row["audio"]) for row in dataset_rows],
        )
    elif not audio_zip_path.exists():
        shutil.copy2(args.audio_zip, audio_zip_path)

    total_duration = sum(float(row["duration"]) for row in dataset_rows)
    summary = {
        "source": DEFAULT_SOURCE_URL,
        "source_rows": len(rows),
        "accepted_rows": len(dataset_rows),
        "rejected_rows": len(rejected_rows),
        "accepted_fraction": len(dataset_rows) / len(rows) if rows else 0.0,
        "accepted_hours": total_duration / 3600,
        "thresholds": {
            "min_votes": args.min_votes,
            "vote_cer": args.vote_cer,
            "max_best_cer": args.max_best_cer,
            "min_margin": args.min_margin,
            "max_candidates": args.max_candidates,
        },
        "prediction_sources": [
            {"name": source.name, "path": str(source.path)} for source in prediction_sources
        ],
    }
    write_json(summary_path, summary)
    write_readme(readme_path, args.repo_id, len(rows), len(dataset_rows), summary)

    verification = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [
            dataset_path,
            alignment_path,
            rejected_path,
            summary_path,
            audio_zip_path,
            readme_path,
        ]
    }
    write_json(verification_path, verification)
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = repair(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
