"""Prepare and score word-preserving punctuation-and-capitalization pilots."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from english_asr.evaluation import pnc_surface, score_pnc_rows

_PNC_SYSTEM_PROMPT = (
    "Restore punctuation and capitalization to English ASR text. Return only the restored text "
    "as one line. Do not add, remove, replace, reorder, expand, contract, or respell any word. "
    "You may only change letter case and add punctuation. Preserve apostrophes exactly."
)
_PNC_HELPER_FIELDS = frozenset({"_pnc_source", "_pnc_split", "lexical_text", "prediction_text"})


@dataclass(frozen=True, kw_only=True)
class OpenAIRestoreConfig:
    """Connection and generation controls for a local OpenAI-compatible restorer."""

    base_url: str
    model: str
    concurrency: int = 16
    request_timeout: float = 120.0
    max_tokens: int = 128


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _sample_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def _require_text(row: dict[str, Any], field: str, source: Path) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} row has no nonempty {field!r}")
    return value


def prepare_pilot(
    lexical_manifest: Path,
    reference_manifest: Path,
    output_manifest: Path,
    *,
    limit: int = 0,
    seed: int = 17,
) -> dict[str, Any]:
    """Build an immutable, matched lexical-input/PnC-reference pilot manifest."""
    if output_manifest.exists():
        raise FileExistsError(f"immutable output already exists: {output_manifest}")
    lexical_rows = _read_manifest(lexical_manifest)
    references: dict[str, dict[str, Any]] = {}
    for row in _read_manifest(reference_manifest):
        sample_id = _require_text(row, "sample_id", reference_manifest)
        if sample_id in references:
            raise ValueError(f"duplicate reference sample_id: {sample_id}")
        references[sample_id] = row

    matched: list[dict[str, str]] = []
    for row in lexical_rows:
        sample_id = _require_text(row, "sample_id", lexical_manifest)
        reference = references.get(sample_id)
        if reference is None:
            raise ValueError(f"reference manifest is missing sample_id: {sample_id}")
        lexical_text = _require_text(row, "text", lexical_manifest)
        reference_text = _require_text(reference, "text", reference_manifest)
        if pnc_surface(lexical_text).words != pnc_surface(reference_text).words:
            raise ValueError(f"lexical/reference mismatch for sample_id: {sample_id}")
        matched.append(
            {
                "sample_id": sample_id,
                "lexical_text": lexical_text,
                "reference_text": reference_text,
                "text": lexical_text,
            }
        )

    matched.sort(key=lambda row: _sample_key(row["sample_id"], seed))
    if limit:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        matched = matched[:limit]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as handle:
        for row in matched:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "rows": len(matched),
        "seed": seed,
        "lexical_manifest": str(lexical_manifest.resolve()),
        "reference_manifest": str(reference_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
    }


def score_manifest(path: Path, *, prediction_field: str = "text") -> dict[str, Any]:
    """Score a prepared pilot after a restorer populates its prediction field."""
    scoring_rows = [
        {
            "lexical_text": _require_text(row, "lexical_text", path),
            "reference_text": _require_text(row, "reference_text", path),
            "prediction_text": _require_text(row, prediction_field, path),
        }
        for row in _read_manifest(path)
    ]
    return score_pnc_rows(scoring_rows)


def _read_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("sources"), list):
        raise TypeError(f"invalid mixture summary: {path}")
    return summary


def prepare_restoration_pool(template_dir: Path, output_manifest: Path) -> dict[str, Any]:
    """Flatten a multi-source mixture into one immutable PnC restoration job."""
    template_dir = template_dir.resolve()
    summary_path = template_dir / "mixture_summary.json"
    summary = _read_summary(summary_path)
    if output_manifest.exists():
        raise FileExistsError(f"immutable output already exists: {output_manifest}")

    rows = 0
    split_counts: dict[str, dict[str, int]] = {}
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as output_handle:
        for source in sorted(summary["sources"], key=lambda item: item["name"]):
            source_name = source["name"]
            split_counts[source_name] = {}
            for split in ("train", "dev"):
                manifest = template_dir / source["output"][split]
                split_rows = 0
                for row in _read_manifest(manifest):
                    collision = _PNC_HELPER_FIELDS.intersection(row)
                    if collision:
                        raise ValueError(
                            f"{manifest} uses reserved PnC fields: {sorted(collision)}"
                        )
                    lexical_text = _require_text(row, "text", manifest)
                    prepared = dict(row)
                    prepared["_pnc_source"] = source_name
                    prepared["_pnc_split"] = split
                    prepared["lexical_text"] = lexical_text
                    output_handle.write(_json_line(prepared))
                    split_rows += 1
                expected = source["output"].get(f"{split}_rows")
                if expected is not None and split_rows != expected:
                    raise ValueError(
                        f"{manifest} contains {split_rows} rows; summary declares {expected}"
                    )
                split_counts[source_name][split] = split_rows
                rows += split_rows
    return {
        "rows": rows,
        "template_dir": str(template_dir),
        "template_summary_sha256": _sha256(summary_path),
        "output_manifest": str(output_manifest.resolve()),
        "output_sha256": _sha256(output_manifest),
        "sources": split_counts,
    }


def _restored_training_row(row: dict[str, Any], source: Path) -> dict[str, Any]:
    lexical_text = _require_text(row, "lexical_text", source)
    prediction_text = _require_text(row, "prediction_text", source)
    if pnc_surface(lexical_text).words != pnc_surface(prediction_text).words:
        sample_id = row.get("sample_id", "<unknown>")
        raise ValueError(f"restorer changed words for sample_id: {sample_id}")
    restored = {key: value for key, value in row.items() if key not in _PNC_HELPER_FIELDS}
    restored["text"] = prediction_text
    return restored


def _load_restored_groups(
    restored_manifest: Path,
    expected_sources: dict[str, dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {
        (name, split): [] for name in expected_sources for split in ("train", "dev")
    }
    restored_by_sample_id: dict[str, dict[str, Any]] = {}
    for row in _read_manifest(restored_manifest):
        source_name = _require_text(row, "_pnc_source", restored_manifest)
        split = _require_text(row, "_pnc_split", restored_manifest)
        key = (source_name, split)
        if key not in grouped:
            raise ValueError(f"unexpected restoration group: {source_name}/{split}")
        sample_id = _require_text(row, "sample_id", restored_manifest)
        if sample_id in restored_by_sample_id:
            raise ValueError(f"duplicate restored sample_id: {sample_id}")
        restored = _restored_training_row(row, restored_manifest)
        grouped[key].append(restored)
        restored_by_sample_id[sample_id] = restored
    return grouped, restored_by_sample_id


def _write_restored_sources(
    temporary: Path,
    template_dir: Path,
    expected_sources: dict[str, dict[str, Any]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output_sources: list[dict[str, Any]] = []
    for source_name in sorted(expected_sources):
        template_source = expected_sources[source_name]
        source_dir = temporary / source_name
        source_dir.mkdir()
        output_record = {
            "name": source_name,
            "sampling_weight": template_source["sampling_weight"],
            "input": {
                "directory": str(template_dir / source_name),
                "train_sha256": template_source["output"]["train_sha256"],
                "dev_sha256": template_source["output"]["dev_sha256"],
            },
            "output": {},
        }
        for split in ("train", "dev"):
            rows = grouped[(source_name, split)]
            expected = template_source["output"][f"{split}_rows"]
            if len(rows) != expected:
                raise ValueError(
                    f"restored {source_name}/{split} has {len(rows)} rows; expected {expected}"
                )
            destination = source_dir / f"{split}.jsonl"
            with destination.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(_json_line(row))
            output_record["output"].update(
                {
                    split: f"{source_name}/{split}.jsonl",
                    f"{split}_rows": len(rows),
                    f"{split}_sha256": _sha256(destination),
                }
            )
        output_sources.append(output_record)
    return output_sources


def _write_balanced_validation(
    temporary: Path,
    template_dir: Path,
    template_summary: dict[str, Any],
    restored_by_sample_id: dict[str, dict[str, Any]],
) -> tuple[Path, int]:
    template_dev = template_dir / template_summary["balanced_validation"]["path"]
    balanced_dev = temporary / "balanced-dev.jsonl"
    rows = 0
    with balanced_dev.open("x", encoding="utf-8") as output_handle:
        for row in _read_manifest(template_dev):
            sample_id = _require_text(row, "sample_id", template_dev)
            restored = restored_by_sample_id.get(sample_id)
            if restored is None:
                raise ValueError(f"restored pool is missing validation sample_id: {sample_id}")
            output_handle.write(_json_line(restored))
            rows += 1
    return balanced_dev, rows


def build_restored_mixture(
    restored_manifest: Path,
    template_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Build an immutable weighted mixture from word-preserving PnC predictions."""
    restored_manifest = restored_manifest.resolve()
    template_dir = template_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    template_summary_path = template_dir / "mixture_summary.json"
    template_summary = _read_summary(template_summary_path)
    expected_sources = {source["name"]: source for source in template_summary["sources"]}
    grouped, restored_by_sample_id = _load_restored_groups(restored_manifest, expected_sources)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp"
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir()
    try:
        output_sources = _write_restored_sources(temporary, template_dir, expected_sources, grouped)
        balanced_dev, balanced_rows = _write_balanced_validation(
            temporary, template_dir, template_summary, restored_by_sample_id
        )

        summary = {
            "schema_version": 1,
            "label_profile": "pnc-nemo-punctuation-en-bert-v1",
            "source_label_profile": template_summary.get("label_profile"),
            "seed": template_summary.get("seed"),
            "restoration": {
                "model": model_name,
                "manifest": str(restored_manifest),
                "manifest_sha256": _sha256(restored_manifest),
                "rows": len(restored_by_sample_id),
                "word_preservation_rate": 1.0,
            },
            "template": {
                "directory": str(template_dir),
                "summary_sha256": _sha256(template_summary_path),
            },
            "sources": output_sources,
            "balanced_validation": {
                "path": "balanced-dev.jsonl",
                "rows_per_source": template_summary["balanced_validation"]["rows_per_source"],
                "rows": balanced_rows,
                "sha256": _sha256(balanced_dev),
            },
        }
        (temporary / "mixture_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    temporary.replace(output_dir)
    return summary


async def _restore_one(
    client: httpx.AsyncClient,
    row: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    lexical_text = _require_text(row, "lexical_text", Path("<request>"))
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _PNC_SYSTEM_PROMPT},
                {"role": "user", "content": lexical_text},
            ],
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    payload = response.json()
    prediction = payload["choices"][0]["message"]["content"]
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("OpenAI-compatible server returned an empty prediction")
    restored = dict(row)
    restored["prediction_text"] = prediction.strip()
    return restored


async def _restore_rows(
    rows: list[dict[str, Any]], config: OpenAIRestoreConfig
) -> tuple[list[dict[str, Any]], float]:
    if config.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(config.concurrency)

    async with httpx.AsyncClient(
        base_url=config.base_url, timeout=config.request_timeout
    ) as client:

        async def bounded(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await _restore_one(
                    client, row, model=config.model, max_tokens=config.max_tokens
                )

        started = time.monotonic()
        restored_rows = await asyncio.gather(*(bounded(row) for row in rows))
        elapsed = time.monotonic() - started
    return restored_rows, elapsed


def restore_openai(
    input_manifest: Path,
    output_manifest: Path,
    config: OpenAIRestoreConfig,
) -> dict[str, Any]:
    """Restore a prepared pilot through an OpenAI-compatible local model server."""
    if output_manifest.exists():
        raise FileExistsError(f"immutable output already exists: {output_manifest}")
    rows = _read_manifest(input_manifest)
    restored_rows, elapsed = asyncio.run(_restore_rows(rows, config))

    invalid_rows = sum(
        pnc_surface(_require_text(row, "lexical_text", input_manifest)).words
        != pnc_surface(_require_text(row, "prediction_text", output_manifest)).words
        for row in restored_rows
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as handle:
        for row in restored_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "model": config.model,
        "rows": len(restored_rows),
        "invalid_word_rows": invalid_rows,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(restored_rows) / elapsed if elapsed else 0.0,
        "input_manifest": str(input_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--lexical-manifest", type=Path, required=True)
    prepare.add_argument("--reference-manifest", type=Path, required=True)
    prepare.add_argument("--output-manifest", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=17)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--prediction-field", default="text")
    score.add_argument("--output-summary", type=Path)

    prepare_pool = subparsers.add_parser("prepare-restoration-pool")
    prepare_pool.add_argument("--template-dir", type=Path, required=True)
    prepare_pool.add_argument("--output-manifest", type=Path, required=True)

    build_mixture = subparsers.add_parser("build-restored-mixture")
    build_mixture.add_argument("--restored-manifest", type=Path, required=True)
    build_mixture.add_argument("--template-dir", type=Path, required=True)
    build_mixture.add_argument("--output-dir", type=Path, required=True)
    build_mixture.add_argument("--model-name", required=True)

    restore = subparsers.add_parser("restore-openai")
    restore.add_argument("--input-manifest", type=Path, required=True)
    restore.add_argument("--output-manifest", type=Path, required=True)
    restore.add_argument("--base-url", default="http://127.0.0.1:8020")
    restore.add_argument("--model", default="qwen3.5-4b")
    restore.add_argument("--concurrency", type=int, default=16)
    restore.add_argument("--timeout", type=float, default=120.0)
    restore.add_argument("--max-tokens", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_pilot(
            args.lexical_manifest,
            args.reference_manifest,
            args.output_manifest,
            limit=args.limit,
            seed=args.seed,
        )
    elif args.command == "score":
        result = score_manifest(args.manifest, prediction_field=args.prediction_field)
        if args.output_summary:
            args.output_summary.parent.mkdir(parents=True, exist_ok=True)
            args.output_summary.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    elif args.command == "prepare-restoration-pool":
        result = prepare_restoration_pool(args.template_dir, args.output_manifest)
    elif args.command == "build-restored-mixture":
        result = build_restored_mixture(
            args.restored_manifest,
            args.template_dir,
            args.output_dir,
            model_name=args.model_name,
        )
    else:
        result = restore_openai(
            args.input_manifest,
            args.output_manifest,
            OpenAIRestoreConfig(
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                request_timeout=args.timeout,
                max_tokens=args.max_tokens,
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
