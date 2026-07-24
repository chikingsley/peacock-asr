"""Run a pinned model-by-exam matrix and score it with the official Open ASR stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from parakeet_finetune_core import eval as shared_eval


@dataclass(frozen=True)
class Binding:
    """One stable name bound to one existing path."""

    name: str
    path: Path


def parse_binding(value: str) -> Binding:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"invalid binding {value!r}; expected NAME=PATH")
    if not all(character.isalnum() or character in {"-", "_"} for character in name):
        raise ValueError(f"invalid binding name {name!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return Binding(name, path)


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required executable is not on PATH: {name}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def _valid_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict)


def _prepare_output(output_dir: Path, matrix: dict[str, object]) -> None:
    matrix_path = output_dir / "matrix.json"
    if output_dir.exists():
        if not _valid_json(matrix_path):
            raise RuntimeError(f"partial or invalid matrix output exists: {output_dir}")
        existing = json.loads(matrix_path.read_text(encoding="utf-8"))
        if existing != matrix:
            raise RuntimeError(f"matrix configuration drift under immutable output: {output_dir}")
        return
    output_dir.mkdir(parents=True)
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_partial(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _aggregate_matrix(
    output_dir: Path, models: list[Binding], exams: list[Binding]
) -> dict[str, object]:
    model_results: dict[str, object] = {}
    aggregate: dict[str, object] = {"schema_version": 1, "models": model_results}
    for model in models:
        domains: dict[str, dict[str, float | int]] = {}
        errors = 0
        reference_words = 0
        audio_seconds = 0.0
        elapsed_seconds = 0.0
        for exam in exams:
            model_dir = output_dir / model.name
            score = json.loads(
                (model_dir / f"{exam.name}.summary.json").read_text(encoding="utf-8")
            )
            runtime = json.loads(
                (model_dir / f"{exam.name}.runtime.json").read_text(encoding="utf-8")
            )
            domains[exam.name] = {
                "wer_percent": score["wer_percent"],
                "rtfx": runtime["rtfx"],
                "rows": score["rows"],
            }
            errors += score["deletions"] + score["insertions"] + score["substitutions"]
            reference_words += score["reference_words"]
            audio_seconds += runtime["audio_seconds"]
            elapsed_seconds += runtime["elapsed_seconds"]
        wer_values = [domain["wer_percent"] for domain in domains.values()]
        model_results[model.name] = {
            "domains": domains,
            "macro_wer_percent": sum(wer_values) / len(wer_values),
            "pooled_wer_percent": 100.0 * errors / reference_words,
            "aggregate_rtfx": audio_seconds / elapsed_seconds,
        }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def run_matrix(  # noqa: PLR0913
    *,
    models: list[Binding],
    exams: list[Binding],
    output_dir: Path,
    scorer: Path,
    normalizer_root: Path,
    normalizer_revision: str,
    batch_size: int,
    warmup_count: int,
    inference_dtype: str,
    longform_attention_context: int,
    load_model_on_cpu: bool,
    disable_cuda_graph_decoder: bool,
    memory_efficient_subsampling: bool,
    score_workers: int,
) -> None:
    """Run every model/exam pair in a fresh evaluator process and persist completion markers."""
    if not scorer.is_file():
        raise FileNotFoundError(scorer)
    if not (normalizer_root / "normalizer" / "normalizer.py").is_file():
        raise FileNotFoundError(f"Open ASR normalizer package missing under {normalizer_root}")
    evaluator = _executable("english-parakeet-eval")
    uv = _executable("uv")
    evaluator_source = Path(shared_eval.__file__).resolve()
    runner_source = Path(__file__).resolve()
    matrix = {
        "models": [{**asdict(item), "path": str(item.path)} for item in models],
        "exams": [{**asdict(item), "path": str(item.path)} for item in exams],
        "batch_size": batch_size,
        "warmup_count": warmup_count,
        "inference_dtype": inference_dtype,
        "longform_attention_context": longform_attention_context,
        "load_model_on_cpu": load_model_on_cpu,
        "disable_cuda_graph_decoder": disable_cuda_graph_decoder,
        "memory_efficient_subsampling": memory_efficient_subsampling,
        "evaluator_source": str(evaluator_source),
        "evaluator_source_sha256": _sha256(evaluator_source),
        "runner_source": str(runner_source),
        "runner_source_sha256": _sha256(runner_source),
        "normalizer_root": str(normalizer_root),
        "normalizer_revision": normalizer_revision,
        "scorer": str(scorer),
        "scorer_sha256": _sha256(scorer),
        "score_workers": score_workers,
    }
    _prepare_output(output_dir, matrix)

    for model in models:
        model_dir = output_dir / model.name
        model_dir.mkdir(exist_ok=True)
        log = model_dir / "eval.log"
        for exam in exams:
            predictions = model_dir / f"{exam.name}.predictions.jsonl"
            runtime = model_dir / f"{exam.name}.runtime.json"
            summary = model_dir / f"{exam.name}.summary.json"
            inference_complete = predictions.is_file() and predictions.stat().st_size > 0
            inference_complete = inference_complete and _valid_json(runtime)
            if not inference_complete:
                _remove_partial(predictions, runtime, summary)
                evaluator_command = [
                    evaluator,
                    "--kind",
                    "tdt",
                    "--model-name",
                    str(model.path),
                    "--manifest",
                    str(exam.path),
                    "--device",
                    "cuda",
                    "--batch-size",
                    str(batch_size),
                    "--warmup-count",
                    str(warmup_count),
                    "--inference-dtype",
                    inference_dtype,
                    "--longform-attention-context",
                    str(longform_attention_context),
                    "--output-jsonl",
                    str(predictions),
                    "--output-summary-json",
                    str(runtime),
                ]
                if load_model_on_cpu:
                    evaluator_command.append("--load-model-on-cpu")
                if disable_cuda_graph_decoder:
                    evaluator_command.append("--disable-cuda-graph-decoder")
                if memory_efficient_subsampling:
                    evaluator_command.append("--memory-efficient-subsampling")
                _run(evaluator_command, log)
            if not _valid_json(summary):
                summary.unlink(missing_ok=True)
                _run(
                    [
                        uv,
                        "run",
                        str(scorer),
                        "--predictions",
                        str(predictions),
                        "--output",
                        str(summary),
                        "--normalizer-root",
                        str(normalizer_root),
                        "--normalizer-revision",
                        normalizer_revision,
                        "--workers",
                        str(score_workers),
                    ],
                    log,
                )
        (model_dir / ".complete").touch()
    _aggregate_matrix(output_dir, models, exams)
    (output_dir / ".complete").touch()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--exam", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--normalizer-root", type=Path, required=True)
    parser.add_argument("--normalizer-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-count", type=int, default=8)
    parser.add_argument("--inference-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--longform-attention-context", type=int, default=0)
    parser.add_argument("--load-model-on-cpu", action="store_true")
    parser.add_argument("--disable-cuda-graph-decoder", action="store_true")
    parser.add_argument("--memory-efficient-subsampling", action="store_true")
    parser.add_argument("--score-workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_matrix(
        models=[parse_binding(value) for value in args.model],
        exams=[parse_binding(value) for value in args.exam],
        output_dir=args.output_dir.expanduser().resolve(),
        scorer=args.scorer.expanduser().resolve(),
        normalizer_root=args.normalizer_root.expanduser().resolve(),
        normalizer_revision=args.normalizer_revision,
        batch_size=args.batch_size,
        warmup_count=args.warmup_count,
        inference_dtype=args.inference_dtype,
        longform_attention_context=args.longform_attention_context,
        load_model_on_cpu=args.load_model_on_cpu,
        disable_cuda_graph_decoder=args.disable_cuda_graph_decoder,
        memory_efficient_subsampling=args.memory_efficient_subsampling,
        score_workers=args.score_workers,
    )
    return 0
