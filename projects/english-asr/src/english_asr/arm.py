"""Run one resumable English 110M data-replacement arm end to end."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from english_asr import ROOT
from english_asr.mixture import Source, parse_source, parse_source_weights
from english_asr.parakeet import PROJECT

SHORT_GATE_STEPS = 2_000


@dataclass(frozen=True)
class Evaluation:
    """One stable evaluation name and manifest."""

    name: str
    manifest: Path


def parse_evaluation(value: str) -> Evaluation:
    """Parse and validate ``NAME=MANIFEST``."""
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"invalid evaluation {value!r}; expected NAME=MANIFEST")
    if not all(character.isalnum() or character in {"-", "_"} for character in name):
        raise ValueError(f"invalid evaluation name {name!r}")
    manifest = Path(raw_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return Evaluation(name, manifest)


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"required executable is not on PATH: {name}")
    return executable


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def wait_for(paths: list[Path], poll_seconds: float) -> None:
    """Wait for explicit upstream completion markers."""
    while missing := [path for path in paths if not path.exists()]:
        print(f"waiting for: {', '.join(str(path) for path in missing)}", flush=True)
        time.sleep(poll_seconds)


def _ensure_lexical_mixture(
    *,
    sources: list[Source],
    sampling_weights: dict[str, float] | None,
    lexical_dir: Path,
    validation_per_source: int,
    log: Path,
) -> None:
    if (lexical_dir / "mixture_summary.json").is_file():
        return
    if lexical_dir.exists():
        raise RuntimeError(f"partial immutable lexical mixture exists: {lexical_dir}")
    command = [_executable("english-mixture")]
    for source in sources:
        command.extend(["--source", f"{source.name}={source.directory}"])
    if sampling_weights is not None:
        for name, weight in sorted(sampling_weights.items()):
            command.extend(["--source-weight", f"{name}={weight:.12g}"])
    command.extend(
        [
            "--output-dir",
            str(lexical_dir),
            "--validation-per-source",
            str(validation_per_source),
            "--seed",
            "0",
        ]
    )
    _run(command, log)


def _ensure_restoration_pool(*, lexical_dir: Path, restoration_dir: Path, log: Path) -> Path:
    input_manifest = restoration_dir / "input.jsonl"
    if input_manifest.is_file():
        return input_manifest
    restoration_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _executable("english-pnc"),
            "prepare-restoration-pool",
            "--template-dir",
            str(lexical_dir),
            "--output-manifest",
            str(input_manifest),
        ],
        log,
    )
    return input_manifest


def _ensure_restored_labels(*, input_manifest: Path, restoration_dir: Path, log: Path) -> Path:
    restored = restoration_dir / "nemo-punctuation-en-bert.jsonl"
    summary = restored.with_suffix(".summary.json")
    if restored.is_file() and summary.is_file():
        return restored
    if restored.exists() or summary.exists():
        raise RuntimeError(f"partial immutable PnC restoration exists under {restoration_dir}")
    _run(
        [
            _executable("uv"),
            "run",
            str(ROOT / "scripts" / "restore_pnc_nemo.py"),
            "--input-manifest",
            str(input_manifest),
            "--output-manifest",
            str(restored),
        ],
        log,
    )
    return restored


def _ensure_pnc_mixture(
    *, restored: Path, lexical_dir: Path, pnc_dir: Path, log: Path
) -> dict[str, Any]:
    summary_path = pnc_dir / "mixture_summary.json"
    if not summary_path.is_file():
        if pnc_dir.exists():
            raise RuntimeError(f"partial immutable PnC mixture exists: {pnc_dir}")
        _run(
            [
                _executable("english-pnc"),
                "build-restored-mixture",
                "--restored-manifest",
                str(restored),
                "--template-dir",
                str(lexical_dir),
                "--output-dir",
                str(pnc_dir),
                "--model-name",
                "punctuation_en_bert",
            ],
            log,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("sources"), list):
        raise TypeError(f"invalid PnC mixture summary: {summary_path}")
    return summary


def _ensure_training(  # noqa: PLR0913
    *,
    name: str,
    pnc_dir: Path,
    mixture: dict[str, Any],
    max_steps: int,
    warmup_steps: int,
    seed: int,
    l2sp_weight: float,
    log: Path,
) -> Path:
    run_dir = PROJECT.runs / name
    best_model = run_dir / f"{name}_best-valloss.nemo"
    if best_model.is_file():
        return best_model
    command = [
        _executable("english-parakeet-train-tdt"),
        "--name",
        name,
    ]
    for source in sorted(mixture["sources"], key=lambda item: item["name"]):
        manifest = pnc_dir / source["output"]["train"]
        command.extend(["--train-source", f"{manifest}={float(source['sampling_weight']):.12g}"])
    command.extend(
        [
            "--validation-manifest",
            str(pnc_dir / mixture["balanced_validation"]["path"]),
            "--max-steps",
            str(max_steps),
            "--val-every",
            "500" if max_steps <= SHORT_GATE_STEPS else "2000",
            "--warmup",
            str(warmup_steps),
            "--lr",
            "1e-4",
            "--seed",
            str(seed),
            "--num-workers",
            "0",
            "--l2sp-weight",
            str(l2sp_weight),
            "--resume",
        ]
    )
    _run(command, log)
    if not best_model.is_file():
        raise FileNotFoundError(best_model)
    return best_model


def _ensure_interpolation(*, name: str, candidate: Path, alpha: float, log: Path) -> Path:
    output = PROJECT.runs / name / f"interpolate-base-alpha{alpha:.2f}.nemo"
    if not output.is_file():
        _run(
            [
                _executable("english-interpolate"),
                "--base",
                str(PROJECT.default_tdt_model),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--alpha",
                str(alpha),
            ],
            log,
        )
    return output


def _evaluate(
    *, name: str, model: Path, evaluations: list[Evaluation], alpha: float, log: Path
) -> dict[str, Any]:
    run_dir = PROJECT.runs / name
    results: dict[str, Any] = {}
    for evaluation in evaluations:
        summary_path = run_dir / f"eval-interp-alpha{alpha:.2f}-{evaluation.name}.summary.json"
        if not summary_path.is_file():
            _run(
                [
                    _executable("english-parakeet-eval"),
                    "--kind",
                    "tdt",
                    "--model-name",
                    str(model),
                    "--manifest",
                    str(evaluation.manifest),
                    "--device",
                    "cuda",
                    "--batch-size",
                    "16",
                    "--warmup-count",
                    "8",
                    "--output-summary-json",
                    str(summary_path),
                ],
                log,
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results[evaluation.name] = {
            "manifest": str(evaluation.manifest),
            "wer_percent": summary["normalized"]["wer_percent"],
            "cer_percent": summary["normalized"]["cer_percent"],
            "rtfx": summary["rtfx"],
            "summary": str(summary_path),
        }
    wer_values = [result["wer_percent"] for result in results.values()]
    return {
        "domains": results,
        "macro_wer_percent": sum(wer_values) / len(wer_values),
    }


def run_arm(  # noqa: PLR0913
    *,
    name: str,
    sources: list[Source],
    sampling_weights: dict[str, float] | None,
    evaluations: list[Evaluation],
    lexical_dir: Path,
    restoration_dir: Path,
    pnc_dir: Path,
    waits: list[Path],
    validation_per_source: int,
    max_steps: int,
    warmup_steps: int,
    seed: int,
    l2sp_weight: float,
    alpha: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Run or resume one arm and write its measured completion summary."""
    run_dir = PROJECT.runs / name
    completion = run_dir / ".complete"
    arm_summary = run_dir / "arm_summary.json"
    if completion.is_file() and arm_summary.is_file():
        return json.loads(arm_summary.read_text(encoding="utf-8"))
    wait_for(waits, poll_seconds)
    log = run_dir / "arm.log"
    _ensure_lexical_mixture(
        sources=sources,
        sampling_weights=sampling_weights,
        lexical_dir=lexical_dir,
        validation_per_source=validation_per_source,
        log=log,
    )
    input_manifest = _ensure_restoration_pool(
        lexical_dir=lexical_dir, restoration_dir=restoration_dir, log=log
    )
    restored = _ensure_restored_labels(
        input_manifest=input_manifest, restoration_dir=restoration_dir, log=log
    )
    mixture = _ensure_pnc_mixture(
        restored=restored, lexical_dir=lexical_dir, pnc_dir=pnc_dir, log=log
    )
    candidate = _ensure_training(
        name=name,
        pnc_dir=pnc_dir,
        mixture=mixture,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        l2sp_weight=l2sp_weight,
        log=log,
    )
    model = _ensure_interpolation(name=name, candidate=candidate, alpha=alpha, log=log)
    result = {
        "name": name,
        "seed": seed,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "l2sp_weight": l2sp_weight,
        "interpolation_alpha": alpha,
        "lexical_dir": str(lexical_dir),
        "pnc_dir": str(pnc_dir),
        "candidate": str(candidate),
        "model": str(model),
        "evaluation": _evaluate(
            name=name, model=model, evaluations=evaluations, alpha=alpha, log=log
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    arm_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completion.touch()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--source-weight", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--eval", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--lexical-dir", type=Path, required=True)
    parser.add_argument("--restoration-dir", type=Path, required=True)
    parser.add_argument("--pnc-dir", type=Path, required=True)
    parser.add_argument("--wait-for", action="append", default=[], type=Path)
    parser.add_argument("--validation-per-source", type=int, default=129)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2sp-weight", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_arm(
        name=args.name,
        sources=[parse_source(value) for value in args.source],
        sampling_weights=parse_source_weights(args.source_weight),
        evaluations=[parse_evaluation(value) for value in args.eval],
        lexical_dir=args.lexical_dir.expanduser().resolve(),
        restoration_dir=args.restoration_dir.expanduser().resolve(),
        pnc_dir=args.pnc_dir.expanduser().resolve(),
        waits=[path.expanduser().resolve() for path in args.wait_for],
        validation_per_source=args.validation_per_source,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        l2sp_weight=args.l2sp_weight,
        alpha=args.alpha,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
