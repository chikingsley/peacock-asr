from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# All curation/scoring/selection data lives inside this core package (gitignored).
CORE_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CURATION_ROOT = CORE_DATA_ROOT / "curation"
DEFAULT_POOL_MANIFEST = (
    CORE_DATA_ROOT
    / "datasets/persian-asr-wer35-fastconformer/manifests/train_manifest.jsonl"
)
DEFAULT_POOL_SCORES = CURATION_ROOT / "omni_scores/fleurs-thomcles-best/scores.jsonl"
DEFAULT_CV25_MANIFEST = CURATION_ROOT / "nemo_runs/cv25-score-pool/manifest/manifest.jsonl"
DEFAULT_CV25_SCORES = CURATION_ROOT / "omni_scores/cv25-best/scores.jsonl"
DEFAULT_OUTPUT_ROOT = CORE_DATA_ROOT / "selection/candidate-manifests"


@dataclass(frozen=True)
class Recipe:
    name: str
    train_quotas: dict[str, float]
    dev_hours: float
    min_duration: float
    max_duration: float
    max_omni_wer: float
    max_nvidia_wer: float | None
    min_words: int


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    candidate_split: str
    original_source_split: str
    finetune_omni_wer: float
    finetune_omni_cer: float
    nvidia_wer: float | None

    @property
    def source(self) -> str:
        return str(self.row["source"])

    @property
    def duration_hours(self) -> float:
        return float(self.row["duration"]) / 3600


RECIPES = [
    Recipe(
        name="persian-asr-clean-100h-filter",
        train_quotas={"common_voice_25_0": 96.2, "fleurs": 3.8},
        dev_hours=2.0,
        min_duration=2.0,
        max_duration=15.0,
        max_omni_wer=0.18,
        max_nvidia_wer=35.0,
        min_words=2,
    ),
    Recipe(
        name="persian-asr-balanced-100h-filter",
        train_quotas={
            "common_voice_25_0": 55.0,
            "fleurs": 5.0,
            "thomcles_persian_farsi_speech": 40.0,
        },
        dev_hours=2.0,
        min_duration=1.0,
        max_duration=20.0,
        max_omni_wer=0.25,
        max_nvidia_wer=35.0,
        min_words=2,
    ),
    Recipe(
        name="persian-asr-target-100h-filter",
        train_quotas={
            "thomcles_persian_farsi_speech": 55.0,
            "common_voice_25_0": 40.0,
            "fleurs": 5.0,
        },
        dev_hours=2.0,
        min_duration=1.0,
        max_duration=20.0,
        max_omni_wer=0.35,
        max_nvidia_wer=35.0,
        min_words=1,
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Persian ASR candidate manifests.")
    parser.add_argument("--pool-manifest", type=Path, default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--pool-scores", type=Path, default=DEFAULT_POOL_SCORES)
    parser.add_argument("--cv25-manifest", type=Path, default=DEFAULT_CV25_MANIFEST)
    parser.add_argument("--cv25-scores", type=Path, default=DEFAULT_CV25_SCORES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_scores(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_jsonl(path)}


def candidate_split(row: dict[str, Any]) -> str | None:
    source = str(row["source"])
    split = str(row["source_split"]).lower()
    if source == "common_voice_25_0":
        if split == "validated":
            return "train"
        if split == "dev":
            return "dev"
        return None
    if split == "train":
        return "train"
    if split in {"dev", "validation"}:
        return "dev"
    return None


def word_count(row: dict[str, Any]) -> int:
    return len(str(row["text"]).split())


def load_candidates(
    manifests: list[Path],
    score_paths: list[Path],
) -> list[Candidate]:
    scores: dict[str, dict[str, Any]] = {}
    for path in score_paths:
        scores.update(read_scores(path))

    candidates: list[Candidate] = []
    for manifest in manifests:
        for row in read_jsonl(manifest):
            split = candidate_split(row)
            if split is None:
                continue
            score = scores.get(str(row["sample_id"]))
            if score is None:
                continue
            candidates.append(
                Candidate(
                    row=row,
                    candidate_split=split,
                    original_source_split=str(row["source_split"]),
                    finetune_omni_wer=float(score["wer"]),
                    finetune_omni_cer=float(score["cer"]),
                    nvidia_wer=float(row["wer"]) if row.get("wer") is not None else None,
                )
            )
    return candidates


def passes_recipe(candidate: Candidate, recipe: Recipe) -> bool:
    duration = float(candidate.row["duration"])
    if duration < recipe.min_duration or duration > recipe.max_duration:
        return False
    if word_count(candidate.row) < recipe.min_words:
        return False
    if candidate.finetune_omni_wer > recipe.max_omni_wer:
        return False
    return not (
        recipe.max_nvidia_wer is not None
        and candidate.nvidia_wer is not None
        and candidate.nvidia_wer > recipe.max_nvidia_wer
    )


def quality_key(candidate: Candidate) -> tuple[float, float, float, float]:
    nvidia_wer = candidate.nvidia_wer if candidate.nvidia_wer is not None else 999.0
    return (
        candidate.finetune_omni_wer,
        candidate.finetune_omni_cer,
        nvidia_wer,
        candidate.duration_hours,
    )


def select_hours(candidates: list[Candidate], target_hours: float) -> list[Candidate]:
    selected = []
    hours = 0.0
    seen_audio = set()
    for candidate in sorted(candidates, key=quality_key):
        audio_path = str(candidate.row["audio_filepath"])
        if audio_path in seen_audio:
            continue
        selected.append(candidate)
        seen_audio.add(audio_path)
        hours += candidate.duration_hours
        if hours >= target_hours:
            break
    return selected


def select_train(candidates: list[Candidate], recipe: Recipe) -> list[Candidate]:
    selected = []
    for source, target_hours in recipe.train_quotas.items():
        source_pool = [
            candidate
            for candidate in candidates
            if candidate.candidate_split == "train"
            and candidate.source == source
            and passes_recipe(candidate, recipe)
        ]
        selected.extend(select_hours(source_pool, target_hours))
    return selected


def select_dev(candidates: list[Candidate], recipe: Recipe) -> list[Candidate]:
    dev_pool = [
        candidate
        for candidate in candidates
        if candidate.candidate_split == "dev"
        and candidate.source in recipe.train_quotas
        and passes_recipe(candidate, recipe)
    ]
    return select_hours(dev_pool, recipe.dev_hours)


def manifest_payload(candidate: Candidate) -> dict[str, Any]:
    row = dict(candidate.row)
    row["source_split"] = candidate.candidate_split
    row["original_source_split"] = candidate.original_source_split
    row["finetune_omni_wer"] = candidate.finetune_omni_wer
    row["finetune_omni_cer"] = candidate.finetune_omni_cer
    row["nvidia_wer"] = candidate.nvidia_wer
    return row


def summarize(selected: list[Candidate]) -> dict[str, Any]:
    rows_by_split: dict[str, int] = defaultdict(int)
    hours_by_split: dict[str, float] = defaultdict(float)
    rows_by_source: dict[str, int] = defaultdict(int)
    hours_by_source: dict[str, float] = defaultdict(float)
    max_omni_wer_by_source: dict[str, float] = defaultdict(float)
    max_nvidia_wer_by_source: dict[str, float] = defaultdict(float)
    for candidate in selected:
        split = candidate.candidate_split
        source = candidate.source
        rows_by_split[split] += 1
        hours_by_split[split] += candidate.duration_hours
        rows_by_source[source] += 1
        hours_by_source[source] += candidate.duration_hours
        max_omni_wer_by_source[source] = max(
            max_omni_wer_by_source[source],
            candidate.finetune_omni_wer,
        )
        if candidate.nvidia_wer is not None:
            max_nvidia_wer_by_source[source] = max(
                max_nvidia_wer_by_source[source],
                candidate.nvidia_wer,
            )
    return {
        "rows": len(selected),
        "hours": sum(candidate.duration_hours for candidate in selected),
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "hours_by_split": dict(sorted(hours_by_split.items())),
        "rows_by_source": dict(sorted(rows_by_source.items())),
        "hours_by_source": dict(sorted(hours_by_source.items())),
        "max_finetune_omni_wer_by_source": dict(
            sorted(max_omni_wer_by_source.items())
        ),
        "max_nvidia_wer_by_source": dict(sorted(max_nvidia_wer_by_source.items())),
    }


def write_candidate(output_root: Path, recipe: Recipe, selected: list[Candidate]) -> None:
    out_dir = output_root / recipe.name
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "train_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            handle.write(json.dumps(manifest_payload(candidate), ensure_ascii=False) + "\n")
    summary = {
        "dataset_name": recipe.name,
        "manifest": str(manifest_path),
        "recipe": {
            "train_quotas": recipe.train_quotas,
            "dev_hours": recipe.dev_hours,
            "min_duration": recipe.min_duration,
            "max_duration": recipe.max_duration,
            "max_omni_wer": recipe.max_omni_wer,
            "max_nvidia_wer": recipe.max_nvidia_wer,
            "min_words": recipe.min_words,
        },
        **summarize(selected),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {out_dir / 'summary.json'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = load_candidates(
        [args.pool_manifest, args.cv25_manifest],
        [args.pool_scores, args.cv25_scores],
    )
    for recipe in RECIPES:
        selected = [*select_train(candidates, recipe), *select_dev(candidates, recipe)]
        write_candidate(args.output_root, recipe, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
