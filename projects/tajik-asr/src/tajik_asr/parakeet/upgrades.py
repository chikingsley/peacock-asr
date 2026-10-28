"""Measured runtime trials and controlled gold-data refinement after a clean baseline."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from parakeet_finetune_core.provenance import fingerprint
from parakeet_finetune_core.splits import validate_splits

from tajik_asr.parakeet import PROJECT

BASELINE_NAME = "tajik-parakeet-tdt-110m-clean-20261028"
BASELINE_UNIT = "peacock-tajik-clean-20261028"
BASELINE_JOB = PROJECT.runs / "clean-retrain-20261028"
DEFAULT_OUTPUT = PROJECT.runs / "upgrades-20261028"
DEFAULT_RUNTIME = {"loss_mode": "fused", "precision": "bf16-mixed"}
PILOT_MIN_DURATION = 2
PILOT_MAX_DURATION = 6
MIN_CHECKPOINTS = 2
REFINEMENT_ARMS = ("control", "balanced")
SHARD_ROWS = 200
GOLD_SHARE = 0.12
SAMPLING_TOLERANCE = 0.02


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_shards(path: Path, rows: list[dict]) -> None:
    for index, start in enumerate(range(0, len(rows), SHARD_ROWS)):
        write_manifest(path / f"shard_{index:05d}.jsonl", rows[start : start + SHARD_ROWS])


def prepare(output: Path) -> dict:
    rows = records(PROJECT.default_train_manifest)
    pilot = [row for row in rows if PILOT_MIN_DURATION <= row["duration"] <= PILOT_MAX_DURATION]
    random.Random(20261028).shuffle(pilot)  # noqa: S311 - fixed runtime comparison input
    random.Random(20261028).shuffle(rows)  # noqa: S311 - deterministic corpus mixing
    gold = [row for row in rows if row["label_origin"] == "fleurs_reference"]
    pseudo = [row for row in rows if row["label_origin"] == "scribe_ensemble_llm_fusion"]
    if not gold or not pseudo or len(gold) + len(pseudo) != len(rows):
        raise ValueError("Training records require explicit gold or pseudo-label provenance")
    for name, subset in (("gold", gold), ("pseudo", pseudo)):
        write_manifest(output / "manifests" / f"{name}.jsonl", subset)
        write_shards(output / "manifests" / name, subset)
    write_manifest(output / "manifests/control.jsonl", rows)
    write_shards(output / "manifests/control", rows)
    write_manifest(output / "manifests/pilot-train.jsonl", pilot[:512])
    validation = [
        row
        for row in records(PROJECT.default_validation_manifest)
        if row["duration"] <= PILOT_MAX_DURATION
    ]
    write_manifest(output / "manifests/pilot-dev.jsonl", validation[:32])
    validate_splits(
        [output / "manifests/gold", output / "manifests/pseudo"],
        [PROJECT.default_validation_manifest, *PROJECT.evaluation_manifests],
        heldout_videos=frozenset(json.loads(PROJECT.heldout_manifest.read_text())["video_ids"]),
        require_video_ids=True,
    )
    plan = {
        "input": fingerprint(PROJECT.default_train_manifest),
        "gold_rows": len(gold),
        "pseudo_rows": len(pseudo),
        "original_gold_clip_fraction": len(gold) / len(rows),
        "trial_gold_clip_fraction": 0.12,
        "refinement_steps_per_arm": 2000,
        "refinement_lr": 1e-5,
        "shard_rows": SHARD_ROWS,
        "arms": ["original mixture control", "12 percent gold clip mixture"],
        "selection": "lowest normalized fp32 development WER; test sets used after selection",
        "benchmark_scope": (
            "same short clips and seed; baseline GPU process suspended with its memory resident"
        ),
    }
    write_json(output / "plan.json", plan)
    return plan


def command(output: Path, name: str, arguments: list[str]) -> int:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be available on PATH")
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}.log").open("w") as log:
        return subprocess.run(  # noqa: S603 - explicit local project command, argument array
            [uv, "run", "--project", str(PROJECT.root), "--locked", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def verify_sampling(output: Path) -> dict:
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
    from omegaconf import OmegaConf
    from parakeet_finetune_core.tdt import train_ds

    config = OmegaConf.create(
        train_ds(
            [
                (output / "manifests/gold", GOLD_SHARE),
                (output / "manifests/pseudo", 1 - GOLD_SHARE),
            ],
            30,
            120,
            4,
            0,
        )
    )
    config.metadata_only = True
    cuts, _ = read_cutset_from_config(config)
    counts = Counter(cut.custom["corpus"] for cut in itertools.islice(cuts, 5000))
    total = sum(counts.values())
    fraction = counts["fleurs"] / total
    if abs(fraction - GOLD_SHARE) > SAMPLING_TOLERANCE:
        raise ValueError(
            f"Observed gold sampling fraction differs from its configuration: {fraction}"
        )
    report = {"draws": total, "gold_fraction": fraction, "sources": dict(counts)}
    write_json(output / "sampling-validation.json", report)
    return report


def training_result(name: str, steps: int) -> dict:
    directory = PROJECT.runs / name
    train = records(directory / "train_log.jsonl")
    validation = records(directory / "val_log.jsonl")
    if train[-1]["step"] != steps or not validation:
        raise ValueError("Training must reach its requested step and validation")
    if any(not math.isfinite(row["loss"]) for row in train):
        raise ValueError("Training produced nonfinite loss")
    if any(row["val_loss"] is None or not math.isfinite(row["val_loss"]) for row in validation):
        raise ValueError("Validation produced nonfinite loss")
    start = train[len(train) // 2]
    end = train[-1]
    elapsed = end["t"] - start["t"]
    return {
        "steps": steps,
        "validation": validation[-1],
        "steady_seconds_per_step": elapsed / (end["step"] - start["step"]),
        "steady_audio_hours_per_second": (end["audio_hours_seen"] - start["audio_hours_seen"])
        / elapsed,
        "peak_cuda_bytes": max(row.get("peak_cuda_bytes", 0) for row in train),
    }


def benchmark(output: Path) -> dict:
    results = []
    for loss_mode in ("fused", "unfused"):
        for precision in ("bf16-mixed", "bf16-true"):
            name = f"tajik-runtime-{loss_mode}-{precision}-20261028"
            if (PROJECT.runs / name).exists():
                raise FileExistsError(f"Preserve the previous trial before rerunning: {name}")
            status = command(
                output,
                name,
                [
                    "tajik-parakeet-train-tdt",
                    "--name",
                    name,
                    "--train-manifest",
                    str(output / "manifests/pilot-train.jsonl"),
                    "--validation-manifest",
                    str(output / "manifests/pilot-dev.jsonl"),
                    "--max-steps",
                    "200",
                    "--warmup",
                    "20",
                    "--val-every",
                    "200",
                    "--log-every",
                    "20",
                    "--num-workers",
                    "2",
                    "--batch-dur",
                    "24",
                    "--max-dur",
                    "6",
                    "--precision",
                    precision,
                    "--loss-mode",
                    loss_mode,
                ],
            )
            result = {
                "name": name,
                "loss_mode": loss_mode,
                "precision": precision,
                "exit_code": status,
            }
            if status:
                result["error"] = (output / f"{name}.log").read_text().splitlines()[-1]
            if status == 0:
                try:
                    result.update(training_result(name, 200))
                except (ValueError, OSError) as error:
                    result.update(exit_code=1, error=str(error))
            results.append(result)
            write_json(output / "runtime-trials.json", {"trials": results})
    successful = [row for row in results if row["exit_code"] == 0]
    if not successful:
        raise RuntimeError("All runtime trials failed; inspect per-trial logs")
    winner = min(successful, key=lambda row: row["steady_seconds_per_step"])
    report = {
        "trials": results,
        "selected": winner,
        "scope": (
            "short-clip runtime comparison; full-duration refinement validates memory and quality"
        ),
    }
    write_json(output / "runtime-trials.json", report)
    return report


def evaluate(output: Path, name: str, model: Path, manifest: Path) -> dict:
    summary = output / "evaluation" / f"{name}.json"
    summary.parent.mkdir(exist_ok=True)
    status = command(
        output,
        name,
        [
            "tajik-parakeet-eval",
            "--model-name",
            str(model),
            "--manifest",
            str(manifest),
            "--device",
            "cuda",
            "--inference-dtype",
            "fp32",
            "--batch-size",
            "8",
            "--warmup-count",
            "8",
            "--output-summary-json",
            str(summary),
            "--output-jsonl",
            str(summary.with_suffix(".predictions.jsonl")),
        ],
    )
    if status:
        raise RuntimeError(f"Evaluation failed: {name}")
    return json.loads(summary.read_text())


def candidates(run: Path, output: Path) -> dict[str, Path]:
    from parakeet_finetune_core.averaging import average_checkpoints

    best = run / f"{run.name}_best-valloss.nemo"
    paths = sorted(
        run.glob("checkpoints/best-valloss-*.ckpt"),
        key=lambda path: float(path.stem.rsplit("-", 1)[1]),
    )
    result = {f"{run.name}-best": best, f"{run.name}-final": run / f"{run.name}_final.nemo"}
    if len(paths) >= MIN_CHECKPOINTS:
        averaged = output / "candidates" / f"{run.name}-averaged.nemo"
        average_checkpoints(best, paths, averaged)
        result[f"{run.name}-averaged"] = averaged
    return result


def refine(output: Path) -> dict:
    prepared = json.loads((output / "plan.json").read_text())
    if fingerprint(PROJECT.default_train_manifest)["sha256"] != prepared["input"]["sha256"]:
        raise ValueError("Training manifest changed after refinement preparation")
    baseline = PROJECT.runs / BASELINE_NAME
    initial = baseline / f"{BASELINE_NAME}_best-valloss.nemo"
    selected_runtime = json.loads((output / "runtime-trials.json").read_text())["selected"]
    runtime = {key: selected_runtime[key] for key in DEFAULT_RUNTIME}
    variants = candidates(baseline, output)
    attempts = [runtime] if runtime == DEFAULT_RUNTIME else [runtime, DEFAULT_RUNTIME]
    for attempt, runtime in enumerate(attempts):
        completed = []
        for arm in REFINEMENT_ARMS:
            name = f"tajik-refine-{arm}-a{attempt}-20261028"
            if (PROJECT.runs / name).exists():
                raise FileExistsError(f"Preserve the previous refinement before rerunning: {name}")
            args = [
                "tajik-parakeet-train-tdt",
                "--name",
                name,
                "--model-name",
                str(initial),
                "--train-manifest",
                str(output / "manifests/control"),
                "--keep-tokenizer",
                "--max-steps",
                "2000",
                "--warmup",
                "100",
                "--lr",
                "0.00001",
                "--val-every",
                "500",
                "--log-every",
                "50",
                "--save-top-k",
                "5",
                "--batch-dur",
                "120",
                "--max-dur",
                "30",
                "--num-workers",
                "4",
                "--loss-mode",
                runtime["loss_mode"],
                "--precision",
                runtime["precision"],
            ]
            if arm == "balanced":
                args += [
                    "--train-source",
                    f"{output / 'manifests/gold'}=0.12",
                    "--train-source",
                    f"{output / 'manifests/pseudo'}=0.88",
                ]
            if command(output, name, args):
                break
            try:
                training_result(name, 2000)
            except (ValueError, OSError) as error:
                write_json(output / f"{name}-failure.json", {"error": str(error)})
                break
            completed.append(PROJECT.runs / name)
        if len(completed) == len(REFINEMENT_ARMS):
            for run in completed:
                variants.update(candidates(run, output))
            break
    else:
        raise RuntimeError("Refinement failed with both runtime configurations")
    # Development scores are the only input to candidate selection.
    scores = {
        name: evaluate(output, f"dev-{name}", model, PROJECT.default_validation_manifest)
        for name, model in variants.items()
    }
    winner = min(scores, key=lambda name: scores[name]["normalized"]["wer_percent"])
    report = {
        "status": "selected_on_development",
        "selected": winner,
        "model": str(variants[winner]),
        "runtime": runtime,
        "development": {name: score["normalized"] for name, score in scores.items()},
        "promotion": "candidate artifact; held-out results follow selection",
    }
    write_json(output / "selection.json", report)
    report["heldout"] = {
        dataset: evaluate(
            output, f"selected-{dataset}", variants[winner], BASELINE_JOB / f"{dataset}.jsonl"
        )
        for dataset in ("fleurs600", "conversational")
    }
    report["status"] = "complete"
    write_json(output / "selection.json", report)
    return report


def wait_and_refine(output: Path) -> dict:
    write_json(output / "status.json", {"status": "waiting_for_clean_baseline"})
    while True:
        state = subprocess.run(  # noqa: S603 - fixed workflow service name
            ["/usr/bin/systemctl", "--user", "is-active", BASELINE_UNIT],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if state not in {"active", "activating", "deactivating"}:
            if not (BASELINE_JOB / "completed-at.txt").exists():
                raise RuntimeError("Clean baseline ended before evaluation completed")
            break
        time.sleep(30)
    (output / "gpu-lease").touch()
    try:
        subprocess.run(["/usr/bin/docker", "stop", "--time", "60", "vllm-asr"], check=True)
        write_json(output / "status.json", {"status": "refining_and_evaluating"})
        result = refine(output)
        write_json(output / "status.json", {"status": "complete", "selected": result["selected"]})
        return result
    finally:
        subprocess.run(  # noqa: S603 - verified local service restoration script
            [str(BASELINE_JOB / "restore-asr.sh")], check=True
        )
        (output / "gpu-lease").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "verify-sampling", "benchmark", "refine", "wait-and-refine"]
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    PROJECT.configure_environment()
    os.environ.setdefault("CUDA_PATH", "/opt/cuda")
    actions = {
        "prepare": prepare,
        "verify-sampling": verify_sampling,
        "benchmark": benchmark,
        "refine": refine,
        "wait-and-refine": wait_and_refine,
    }
    try:
        result = actions[args.action](args.output_root.resolve())
    except Exception as error:
        write_json(args.output_root / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
