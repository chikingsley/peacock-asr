"""Export a matched random-versus-cleaned Parakeet screen from the V4 quality ledger."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import multiprocessing
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from farsi_asr.quality_v4 import _audio_bytes

if TYPE_CHECKING:
    import argparse

DURATION_BINS = (5.0, 10.0, 20.0)
POLICY_VERSION = 1
CONTROL_CANDIDATE_WINDOW = 32


@dataclass
class RiskRow:
    hub_path: str
    hub_row_index: int
    source: str
    text: str
    audio_sha256: str
    duration: float
    duration_bin: str
    has_digit: bool
    aligned: bool
    agreement_risk: float | None = None
    edge_risk: float | None = None
    alignment_risk: float = 1.0
    risk: float = 1.0

    @property
    def identity(self) -> tuple[str, int]:
        return self.hub_path, self.hub_row_index


@dataclass(frozen=True)
class SelectedRow:
    arm: str
    row: RiskRow


def _stable_key(seed: int, namespace: str, row: RiskRow) -> int:
    payload = f"{seed}\0{namespace}\0{row.hub_path}\0{row.hub_row_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _duration_bin(duration: float) -> str:
    if duration < DURATION_BINS[0]:
        return "00-05"
    if duration < DURATION_BINS[1]:
        return "05-10"
    if duration < DURATION_BINS[2]:
        return "10-20"
    return "20-plus"


def _normalized_value_ranks(values: list[float]) -> dict[float, float]:
    unique = sorted(set(values))
    if len(unique) <= 1:
        return dict.fromkeys(unique, 0.0)
    denominator = len(unique) - 1
    return {value: index / denominator for index, value in enumerate(unique)}


def _mean(values: list[float | None]) -> float:
    available = [value for value in values if value is not None]
    if not available:
        return 1.0
    return sum(available) / len(available)


def _score_group(raw_rows: list[sqlite3.Row]) -> list[RiskRow]:
    rows = [
        RiskRow(
            hub_path=str(raw["hub_path"]),
            hub_row_index=int(raw["hub_row_index"]),
            source=str(raw["source"]),
            text=str(raw["text"]),
            audio_sha256=str(raw["audio_sha256"]),
            duration=float(raw["duration"]),
            duration_bin=_duration_bin(float(raw["duration"])),
            has_digit=any(character.isdigit() for character in str(raw["text"])),
            aligned=str(raw["alignment_status"]) == "aligned",
        )
        for raw in raw_rows
    ]
    wer_ranks = _normalized_value_ranks([float(raw["wer"]) for raw in raw_rows])
    cer_ranks = _normalized_value_ranks([float(raw["cer"]) for raw in raw_rows])
    edge_ranks = _normalized_value_ranks([float(raw["edge_chars"]) for raw in raw_rows])
    coverage_ranks = _normalized_value_ranks([float(raw["coverage_bad"]) for raw in raw_rows])
    span_ranks = _normalized_value_ranks([float(raw["span_bad"]) for raw in raw_rows])
    margin_ranks = _normalized_value_ranks([float(raw["margin_ratio"]) for raw in raw_rows])
    overrun_ranks = _normalized_value_ranks([float(raw["overrun_ratio"]) for raw in raw_rows])

    for row, raw in zip(rows, raw_rows, strict=True):
        if not row.has_digit:
            row.agreement_risk = _mean([wer_ranks[float(raw["wer"])], cer_ranks[float(raw["cer"])]])
        if row.duration < DURATION_BINS[-1]:
            row.edge_risk = edge_ranks[float(raw["edge_chars"])]
        row.alignment_risk = (
            _mean(
                [
                    coverage_ranks[float(raw["coverage_bad"])],
                    span_ranks[float(raw["span_bad"])],
                    margin_ranks[float(raw["margin_ratio"])],
                    overrun_ranks[float(raw["overrun_ratio"])],
                ]
            )
            if row.aligned
            else 1.0
        )
        row.risk = _mean([row.agreement_risk, row.edge_risk, row.alignment_risk])
    return rows


def _take_to_target(rows: list[RiskRow], target_seconds: float, *, seed: int) -> list[RiskRow]:
    ordered = sorted(rows, key=lambda row: _stable_key(seed, "cleaned", row))
    selected: list[RiskRow] = []
    total = 0.0
    for row in ordered:
        before = abs(target_seconds - total)
        after = abs(target_seconds - (total + row.duration))
        if after >= before:
            break
        selected.append(row)
        total += row.duration
        if total >= target_seconds:
            break
    return selected


def _nearest_unused(
    candidates: list[RiskRow],
    durations: list[float],
    used: set[tuple[str, int]],
    target: RiskRow,
    *,
    seed: int,
) -> RiskRow:
    insertion = bisect.bisect_left(durations, target.duration)
    choices: list[RiskRow] = []
    left = insertion - 1
    right = insertion
    while len(choices) < CONTROL_CANDIDATE_WINDOW and (left >= 0 or right < len(candidates)):
        take_left = right >= len(candidates) or (
            left >= 0
            and abs(candidates[left].duration - target.duration)
            <= abs(candidates[right].duration - target.duration)
        )
        index = left if take_left else right
        candidate = candidates[index]
        if take_left:
            left -= 1
        else:
            right += 1
        if candidate.identity in used or candidate.identity == target.identity:
            continue
        choices.append(candidate)
    if not choices:
        raise RuntimeError(f"unable to duration-match control row for {target.identity}")
    return min(
        choices,
        key=lambda row: (
            abs(row.duration - target.duration),
            _stable_key(seed, "control-tie", row),
        ),
    )


def _match_control(group: list[RiskRow], cleaned: list[RiskRow], *, seed: int) -> list[RiskRow]:
    candidates = sorted(
        group,
        key=lambda row: (row.duration, _stable_key(seed, "control-order", row)),
    )
    durations = [row.duration for row in candidates]
    used = {row.identity for row in cleaned}
    selected: list[RiskRow] = []
    for target in sorted(cleaned, key=lambda row: _stable_key(seed, "control-target", row)):
        candidate = _nearest_unused(candidates, durations, used, target, seed=seed)
        used.add(candidate.identity)
        selected.append(candidate)
    return selected


def _query_rows(connection: sqlite3.Connection) -> sqlite3.Cursor:
    return connection.execute(
        """
        SELECT
            q.hub_path,
            q.hub_row_index,
            q.source,
            q.text,
            q.audio_sha256,
            q.duration,
            json_extract(q.asr_agreement_json, '$.wer') AS wer,
            json_extract(q.asr_agreement_json, '$.cer') AS cer,
            coalesce(json_extract(q.asr_edge_json, '$.beginning_error_chars'), 0)
                + coalesce(json_extract(q.asr_edge_json, '$.end_error_chars'), 0) AS edge_chars,
            c.status AS alignment_status,
            1.0 - coalesce(json_extract(c.alignment_json, '$.word_coverage'), 0.0)
                AS coverage_bad,
            1.0 - coalesce(json_extract(c.alignment_json, '$.aligned_span_ratio'), 0.0)
                AS span_bad,
            max(
                coalesce(json_extract(c.alignment_json, '$.leading_margin_seconds'), 0.0),
                coalesce(json_extract(c.alignment_json, '$.trailing_margin_seconds'), 0.0)
            ) / q.duration AS margin_ratio,
            coalesce(json_extract(c.alignment_json, '$.end_overrun_seconds'), 0.0)
                / q.duration AS overrun_ratio,
            CASE
                WHEN q.duration < 5.0 THEN '00-05'
                WHEN q.duration < 10.0 THEN '05-10'
                WHEN q.duration < 20.0 THEN '10-20'
                ELSE '20-plus'
            END AS duration_bin
        FROM quality_rows AS q
        JOIN ctc_alignments AS c USING (hub_path, hub_row_index)
        ORDER BY q.source, duration_bin, q.hub_path, q.hub_row_index
        """
    )


def _validate_ledger(connection: sqlite3.Connection, expected_rows: int) -> None:
    quality_rows = int(connection.execute("SELECT count(*) FROM quality_rows").fetchone()[0])
    ctc_rows = int(connection.execute("SELECT count(*) FROM ctc_alignments").fetchone()[0])
    asr_errors = int(
        connection.execute("SELECT count(*) FROM quality_rows WHERE error IS NOT NULL").fetchone()[
            0
        ]
    )
    matched_rows = int(
        connection.execute(
            """
            SELECT count(*)
            FROM quality_rows AS q
            JOIN ctc_alignments AS c USING (hub_path, hub_row_index)
            """
        ).fetchone()[0]
    )
    if (
        quality_rows != expected_rows
        or ctc_rows != expected_rows
        or matched_rows != expected_rows
        or asr_errors
    ):
        raise SystemExit(
            f"quality gate failed: quality_rows={quality_rows} ctc_rows={ctc_rows} "
            f"matched_rows={matched_rows} asr_errors={asr_errors} expected={expected_rows}"
        )


def _duration_bin_seconds(connection: sqlite3.Connection) -> dict[tuple[str, str], float]:
    return {
        (str(source), str(duration_bin)): float(seconds)
        for source, duration_bin, seconds in connection.execute(
            """
            SELECT
                source,
                CASE
                    WHEN duration < 5.0 THEN '00-05'
                    WHEN duration < 10.0 THEN '05-10'
                    WHEN duration < 20.0 THEN '10-20'
                    ELSE '20-plus'
                END AS duration_bin,
                sum(duration)
            FROM quality_rows
            GROUP BY source, duration_bin
            """
        )
    }


def _select_rows(
    connection: sqlite3.Connection,
    *,
    target_hours: float,
    seed: int,
    source_balance: str,
) -> tuple[list[SelectedRow], dict[str, Any]]:
    source_seconds = {
        str(source): float(seconds)
        for source, seconds in connection.execute(
            "SELECT source, sum(duration) FROM quality_rows GROUP BY source ORDER BY source"
        )
    }
    total_seconds = sum(source_seconds.values())
    if source_balance == "equal":
        target_by_source = {
            source: target_hours * 3600.0 / len(source_seconds) for source in source_seconds
        }
    else:
        target_by_source = {
            source: target_hours * 3600.0 * seconds / total_seconds
            for source, seconds in source_seconds.items()
        }

    bin_seconds = _duration_bin_seconds(connection)

    selected: list[SelectedRow] = []
    group_summaries: list[dict[str, Any]] = []
    cursor = _query_rows(connection)
    for (source, duration_bin), raw_group in groupby(
        cursor, key=lambda raw: (str(raw["source"]), str(raw["duration_bin"]))
    ):
        raw_rows = list(raw_group)
        rows = _score_group(raw_rows)
        clean_eligible = [row for row in rows if row.aligned]
        clean_eligible.sort(key=lambda row: (row.risk, _stable_key(seed, "risk-tie", row)))
        lower_half = clean_eligible[: math.ceil(len(rows) / 2)]
        source_bin_seconds = bin_seconds[source, duration_bin]
        target_seconds = target_by_source[source] * source_bin_seconds / source_seconds[source]
        cleaned = _take_to_target(lower_half, target_seconds, seed=seed)
        control = _match_control(rows, cleaned, seed=seed)
        selected.extend(SelectedRow("cleaned", row) for row in cleaned)
        selected.extend(SelectedRow("control", row) for row in control)
        group_summaries.append(
            {
                "source": source,
                "duration_bin": duration_bin,
                "pool_rows": len(rows),
                "lower_risk_rows": len(lower_half),
                "target_hours": target_seconds / 3600.0,
                "cleaned_rows": len(cleaned),
                "control_rows": len(control),
                "cleaned_hours": sum(row.duration for row in cleaned) / 3600.0,
                "control_hours": sum(row.duration for row in control) / 3600.0,
                "cleaned_mean_risk": sum(row.risk for row in cleaned) / max(1, len(cleaned)),
                "control_mean_risk": sum(row.risk for row in control) / max(1, len(control)),
            }
        )
        print(
            f"selected {source}/{duration_bin}: "
            f"cleaned={len(cleaned)} control={len(control)} "
            f"target={target_seconds / 3600.0:.3f}h",
            flush=True,
        )
    return selected, {
        "source_balance": source_balance,
        "target_hours": target_hours,
        "target_hours_by_source": {
            source: seconds / 3600.0 for source, seconds in target_by_source.items()
        },
        "groups": group_summaries,
    }


def _audio_path(output_dir: Path, row: RiskRow) -> Path:
    identity = hashlib.blake2b(
        f"{row.hub_path}:{row.hub_row_index}".encode(), digest_size=10
    ).hexdigest()
    return (output_dir / "audio" / row.source / f"{identity}.flac").resolve()


def _materialize_shard(
    dataset_root: Path,
    output_dir: Path,
    hub_path: str,
    wanted: dict[int, RiskRow],
) -> dict[tuple[str, int], Path]:
    parquet = pq.ParquetFile(dataset_root / hub_path)
    paths: dict[tuple[str, int], Path] = {}
    offset = 0
    found: set[int] = set()
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(row_group_index).num_rows
        wanted_in_group = {
            row_index for row_index in wanted if offset <= row_index < offset + row_count
        }
        if not wanted_in_group:
            offset += row_count
            continue
        table = parquet.read_row_group(row_group_index, columns=["text", "audio_bytes"])
        texts = table.column(0).to_pylist()
        audio = table.column(1).to_pylist()
        for group_index, (text_raw, audio_raw) in enumerate(zip(texts, audio, strict=True)):
            row_index = offset + group_index
            if row_index not in wanted_in_group:
                continue
            row = wanted[row_index]
            encoded = _audio_bytes(audio_raw)
            if str(text_raw) != row.text:
                raise RuntimeError(f"V4 text changed for {(hub_path, row_index)}")
            digest = hashlib.sha256(encoded).hexdigest()
            if digest != row.audio_sha256:
                raise RuntimeError(f"V4 audio changed for {(hub_path, row_index)}")
            path = _audio_path(output_dir, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
            paths[row.identity] = path
            found.add(row_index)
        offset += row_count
        if len(found) == len(wanted):
            break
    missing = set(wanted) - found
    if missing:
        raise RuntimeError(f"selected V4 rows missing from {hub_path}: {sorted(missing)[:10]}")
    return paths


def _materialize_audio(
    dataset_root: Path,
    output_dir: Path,
    rows: list[SelectedRow],
    *,
    workers: int,
) -> dict[tuple[str, int], Path]:
    unique = {selected.row.identity: selected.row for selected in rows}
    by_shard: dict[str, dict[int, RiskRow]] = defaultdict(dict)
    for row in unique.values():
        by_shard[row.hub_path][row.hub_row_index] = row

    paths: dict[tuple[str, int], Path] = {}
    shard_count = len(by_shard)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                _materialize_shard, dataset_root, output_dir, hub_path, wanted
            ): hub_path
            for hub_path, wanted in sorted(by_shard.items())
        }
        for index, future in enumerate(as_completed(futures), start=1):
            paths.update(future.result())
            if index == 1 or index % 25 == 0 or index == shard_count:
                print(
                    f"materialized {index}/{shard_count} V4 shards "
                    f"({len(paths)} unique clips, {workers} workers)",
                    flush=True,
                )
    if len(paths) != len(unique):
        raise RuntimeError(f"materialized {len(paths)} unique rows; expected {len(unique)}")
    return paths


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_outputs(
    output_dir: Path,
    selected: list[SelectedRow],
    audio_paths: dict[tuple[str, int], Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_arm: dict[str, list[SelectedRow]] = defaultdict(list)
    for item in selected:
        by_arm[item.arm].append(item)

    arm_summaries: dict[str, Any] = {}
    identities_by_arm: dict[str, set[tuple[str, int]]] = {}
    for arm in ("control", "cleaned"):
        ordered = sorted(
            by_arm[arm],
            key=lambda item: (
                item.row.source,
                item.row.duration_bin,
                item.row.hub_path,
                item.row.hub_row_index,
            ),
        )
        manifest = output_dir / f"{arm}.manifest.jsonl"
        selection = output_dir / f"{arm}.selection.jsonl"
        _write_jsonl(
            manifest,
            [
                {
                    "audio_filepath": str(audio_paths[item.row.identity]),
                    "duration": item.row.duration,
                    "text": item.row.text,
                }
                for item in ordered
            ],
        )
        _write_jsonl(
            selection,
            [
                {
                    **asdict(item.row),
                    "audio_filepath": str(audio_paths[item.row.identity]),
                }
                for item in ordered
            ],
        )
        identities_by_arm[arm] = {item.row.identity for item in ordered}
        arm_summaries[arm] = {
            "rows": len(ordered),
            "hours": sum(item.row.duration for item in ordered) / 3600.0,
            "mean_risk": sum(item.row.risk for item in ordered) / max(1, len(ordered)),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "selection": str(selection.resolve()),
            "selection_sha256": _sha256(selection),
        }

    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    hours: dict[tuple[str, str], float] = defaultdict(float)
    for item in selected:
        counts[item.arm, item.row.source, item.row.duration_bin] += 1
        hours[item.arm, item.row.source] += item.row.duration / 3600.0
    count_mismatches = [
        (source, duration_bin)
        for source, duration_bin in sorted({(key[1], key[2]) for key in counts})
        if counts["control", source, duration_bin] != counts["cleaned", source, duration_bin]
    ]
    if count_mismatches:
        raise RuntimeError(f"matched duration-bin counts diverged: {count_mismatches}")
    sources = sorted({item.row.source for item in selected})
    max_source_hour_delta = max(
        abs(hours["control", source] - hours["cleaned", source]) for source in sources
    )
    summary.update(
        {
            "policy_version": POLICY_VERSION,
            "duration_bins_seconds": [0.0, *DURATION_BINS, None],
            "arms": arm_summaries,
            "overlap_rows": len(identities_by_arm["control"] & identities_by_arm["cleaned"]),
            "max_source_hour_delta": max_source_hour_delta,
            "source_hours": {
                arm: {source: hours[arm, source] for source in sources}
                for arm in ("control", "cleaned")
            },
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _store_selection(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    seed: int,
    summary: dict[str, Any],
    selected: list[SelectedRow],
) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS quality_ablation_runs (
            run_id TEXT PRIMARY KEY,
            seed INTEGER NOT NULL,
            policy_version INTEGER NOT NULL,
            summary_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS quality_ablation_rows (
            run_id TEXT NOT NULL,
            arm TEXT NOT NULL,
            hub_path TEXT NOT NULL,
            hub_row_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            duration_bin TEXT NOT NULL,
            duration REAL NOT NULL,
            risk REAL NOT NULL,
            agreement_risk REAL,
            edge_risk REAL,
            alignment_risk REAL NOT NULL,
            PRIMARY KEY (run_id, arm, hub_path, hub_row_index),
            FOREIGN KEY (run_id) REFERENCES quality_ablation_runs(run_id)
        );
        """
    )
    if connection.execute(
        "SELECT 1 FROM quality_ablation_runs WHERE run_id = ?", (run_id,)
    ).fetchone():
        raise SystemExit(f"quality ablation run already exists: {run_id}")
    connection.execute(
        "INSERT INTO quality_ablation_runs VALUES (?, ?, ?, ?)",
        (run_id, seed, POLICY_VERSION, json.dumps(summary, ensure_ascii=False, sort_keys=True)),
    )
    connection.executemany(
        """
        INSERT INTO quality_ablation_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                item.arm,
                item.row.hub_path,
                item.row.hub_row_index,
                item.row.source,
                item.row.duration_bin,
                item.row.duration,
                item.row.risk,
                item.row.agreement_risk,
                item.row.edge_risk,
                item.row.alignment_risk,
            )
            for item in selected
        ],
    )
    connection.commit()


def export_ablation(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    connection = sqlite3.connect(args.database, timeout=300)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=300000")
    _validate_ledger(connection, args.expected_rows)
    selected, summary = _select_rows(
        connection,
        target_hours=args.target_hours,
        seed=args.seed,
        source_balance=args.source_balance,
    )
    audio_paths = _materialize_audio(
        args.dataset_root,
        args.output_dir,
        selected,
        workers=args.materialize_workers,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "seed": args.seed,
            "database": str(args.database.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
        }
    )
    summary = _write_outputs(args.output_dir, selected, audio_paths, summary)
    if summary["max_source_hour_delta"] * 3600.0 > args.max_source_hour_delta_seconds:
        raise RuntimeError(
            "matched source-hour delta exceeds gate: "
            f"{summary['max_source_hour_delta'] * 3600.0:.3f}s > "
            f"{args.max_source_hour_delta_seconds:.3f}s"
        )
    _store_selection(
        connection,
        run_id=args.run_id,
        seed=args.seed,
        summary=summary,
        selected=selected,
    )
    connection.close()
    print(json.dumps(summary["arms"], indent=2, sort_keys=True), flush=True)
    print(f"matched V4 quality ablation -> {args.output_dir}", flush=True)
    return 0


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "export-ablation",
        help="materialize matched random and lower-risk V4 Parakeet manifests",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-hours", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--source-balance", choices=("equal", "proportional"), default="equal")
    parser.add_argument("--expected-rows", type=int, default=538117)
    parser.add_argument("--max-source-hour-delta-seconds", type=float, default=60.0)
    parser.add_argument("--materialize-workers", type=int, default=8)
    parser.set_defaults(func=export_ablation)
