from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text

DEFAULT_ROOT = Path("/home/simon/github/peacock-asr/projects/tajik-asr/data/raw")
DATASETS_SERVER = "https://datasets-server.huggingface.co"
TEXT_COLUMNS = ("transcript", "sentence", "transcription", "raw_transcription", "text")


@dataclass(frozen=True)
class Candidate:
    repo: str
    config: str = "default"
    useful: bool = True
    note: str = ""


@dataclass(frozen=True)
class ParquetFile:
    dataset: str
    config: str
    split: str
    filename: str
    size: int
    url: str


CANDIDATES = {
    "shunyalabs": Candidate("shunyalabs/tajik-speech-dataset", note="audio + transcript"),
    "muhtasham-augmented": Candidate(
        "muhtasham/tajik-asr-augmented-test", note="audio + sentence, augmented"
    ),
    "sib-fleurs": Candidate("WueNLP/sib-fleurs", config="tgk_Cyrl", note="FLEURS-derived"),
    "belebele-fleurs": Candidate(
        "WueNLP/belebele-fleurs", config="tgk_Cyrl", note="FLEURS/Belebele eval"
    ),
    "2m-belebele": Candidate(
        "facebook/2M-Belebele", config="tgk_Cyrl", note="Belebele/FLORES/FLEURS eval"
    ),
    "abduaziz-fleurs-cleaned": Candidate(
        "abduaziz/fleurs_tajik_cleaned",
        useful=False,
        note="Whisper feature tensors, not raw audio",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and inventory accessible Tajik HF data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured candidates.")
    list_parser.set_defaults(func=run_list)

    download_parser = subparsers.add_parser("download", help="Download candidate parquet files.")
    download_parser.add_argument("names", nargs="*", default=[])
    download_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    download_parser.add_argument("--include-nonuseful", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.set_defaults(func=run_download)

    inventory_parser = subparsers.add_parser(
        "inventory", help="Inventory downloaded parquet files."
    )
    inventory_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    inventory_parser.add_argument("--out", type=Path, default=Path("data/raw/hf_inventory.json"))
    inventory_parser.add_argument("--examples", type=int, default=3)
    inventory_parser.set_defaults(func=run_inventory)

    sample_parser = subparsers.add_parser("sample", help="Print one downloaded parquet row.")
    sample_parser.add_argument("path", type=Path)
    sample_parser.add_argument("--columns", nargs="*", default=[])
    sample_parser.set_defaults(func=run_sample)
    return parser


def api_json(path: str, params: dict[str, str]) -> Any:
    query = urllib.parse.urlencode(params)
    url = f"{DATASETS_SERVER}/{path}?{query}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def dataset_dir(root: Path, repo: str, config: str) -> Path:
    names = {
        "shunyalabs/tajik-speech-dataset": "shunyalabs_tajik_speech_dataset",
        "muhtasham/tajik-asr-augmented-test": "muhtasham_tajik_asr_augmented_test",
        "WueNLP/sib-fleurs": "wuenlp_sib_fleurs_tgk_cyrl",
        "WueNLP/belebele-fleurs": "wuenlp_belebele_fleurs_tgk_cyrl",
        "facebook/2M-Belebele": "facebook_2m_belebele_tgk_cyrl",
        "abduaziz/fleurs_tajik_cleaned": "abduaziz_fleurs_tajik_cleaned",
    }
    return root / names.get(repo, f"{repo.replace('/', '_')}_{config}")


def parquet_files(candidate: Candidate) -> list[ParquetFile]:
    data = api_json("parquet", {"dataset": candidate.repo})
    files: list[ParquetFile] = []
    for row in data.get("parquet_files", []):
        if row["config"] != candidate.config:
            continue
        files.append(
            ParquetFile(
                dataset=row["dataset"],
                config=row["config"],
                split=row["split"],
                filename=row["filename"],
                size=int(row["size"]),
                url=row["url"],
            )
        )
    return files


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def selected_candidates(names: list[str], include_nonuseful: bool) -> dict[str, Candidate]:
    selected = CANDIDATES if not names else {name: CANDIDATES[name] for name in names}
    if include_nonuseful:
        return selected
    return {name: candidate for name, candidate in selected.items() if candidate.useful}


def run_list(_args: argparse.Namespace) -> None:
    for name, candidate in CANDIDATES.items():
        print(
            json.dumps(
                {"name": name, **asdict(candidate)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def run_download(args: argparse.Namespace) -> None:
    candidates = selected_candidates(args.names, args.include_nonuseful)
    for name, candidate in candidates.items():
        try:
            files = parquet_files(candidate)
        except Exception as exc:
            print(f"{name}\t{candidate.repo}\tERROR\t{exc}")
            continue
        total_size = sum(file.size for file in files)
        print(
            f"{name}\t{candidate.repo}\t{candidate.config}\t"
            f"{len(files)} files\t{total_size} bytes"
        )
        if args.dry_run:
            continue
        root = dataset_dir(args.root, candidate.repo, candidate.config)
        manifest = {
            "name": name,
            "candidate": asdict(candidate),
            "files": [asdict(file) for file in files],
        }
        (root / "source.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "source.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for file in files:
            target = root / file.split / file.filename
            print(f"download\t{target}")
            download_file(file.url, target)


def text_value(row: dict[str, Any]) -> str:
    for column in TEXT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def inspect_parquet(
    path: Path, examples_limit: int
) -> tuple[dict[str, Any], Counter[str], list[str]]:
    parquet = pq.ParquetFile(path)
    fields = parquet.schema_arrow.names
    rows = parquet.metadata.num_rows
    text_counter: Counter[str] = Counter()
    examples: list[str] = []
    audio_columns = [name for name in fields if "audio" in name.lower()]
    text_columns = [name for name in fields if name in TEXT_COLUMNS]
    for batch in parquet.iter_batches(batch_size=512, columns=text_columns):
        for row in batch.to_pylist():
            text = text_value(row)
            if text:
                normalized = normalize_text(text)
                text_counter[normalized] += 1
                if len(examples) < examples_limit:
                    examples.append(text)
    return (
        {
            "path": str(path),
            "rows": rows,
            "columns": fields,
            "text_columns": text_columns,
            "audio_columns": audio_columns,
            "unique_normalized_texts": len(text_counter),
            "examples": examples,
        },
        text_counter,
        examples,
    )


def run_inventory(args: argparse.Namespace) -> None:
    summaries: list[dict[str, Any]] = []
    by_dataset_texts: dict[str, Counter[str]] = defaultdict(Counter)
    for source_path in sorted(args.root.glob("*/source.json")):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        repo = source["candidate"]["repo"]
        config = source["candidate"]["config"]
        root = source_path.parent
        dataset_summary: dict[str, Any] = {
            "repo": repo,
            "config": config,
            "note": source["candidate"].get("note", ""),
            "files": [],
        }
        for path in sorted(root.glob("*/*.parquet")):
            file_summary, text_counter, _examples = inspect_parquet(path, args.examples)
            split = path.parent.name
            file_summary["split"] = split
            dataset_summary["files"].append(file_summary)
            by_dataset_texts[f"{repo}/{config}"].update(text_counter)
        dataset_summary["rows"] = sum(file["rows"] for file in dataset_summary["files"])
        dataset_summary["unique_normalized_texts"] = len(by_dataset_texts[f"{repo}/{config}"])
        summaries.append(dataset_summary)

    overlap: list[dict[str, Any]] = []
    keys = sorted(by_dataset_texts)
    for index, left_key in enumerate(keys):
        for right_key in keys[index + 1 :]:
            shared = set(by_dataset_texts[left_key]) & set(by_dataset_texts[right_key])
            if shared:
                overlap.append(
                    {
                        "left": left_key,
                        "right": right_key,
                        "shared_normalized_texts": len(shared),
                    }
                )

    payload = {"datasets": summaries, "overlap": overlap}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


def simplify_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, dict):
        return {key: simplify_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview": [simplify_value(item) for item in value[:5]],
        }
    return value


def run_sample(args: argparse.Namespace) -> None:
    parquet = pq.ParquetFile(args.path)
    columns = args.columns or parquet.schema_arrow.names
    batch = next(parquet.iter_batches(batch_size=1, columns=columns), None)
    if batch is None:
        print("{}")
        return
    row = batch.to_pylist()[0]
    print(json.dumps(simplify_value(row), ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
