from __future__ import annotations

from pathlib import Path
from pprint import pformat

import click

from .superb import TASK_DATASET_KEYS, get_problem


@click.group()
def cli() -> None:
    """Run original-HConv benchmark reproductions on top of s3prl SUPERB tasks."""


@cli.command("show-config")
@click.option("--task", type=click.Choice(["pr", "er", "ic", "asv"], case_sensitive=False), required=True)
def show_config(task: str) -> None:
    problem = get_problem(task)
    click.echo(pformat(problem.default_config(), sort_dicts=False))


@cli.command("run-superb")
@click.option("--task", type=click.Choice(["pr", "er", "ic", "asv"], case_sensitive=False), required=True)
@click.option("--dataset-root", type=click.Path(path_type=Path), required=True)
@click.option("--target-dir", type=click.Path(path_type=Path), required=True)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
@click.option("--upstream", "upstream_name", required=True)
@click.option(
    "--interface",
    type=click.Choice(["hconv", "weighted_sum"], case_sensitive=False),
    default="hconv",
    show_default=True,
)
@click.option("--device", default="cuda", show_default=True)
@click.option("--normalize/--no-normalize", default=False, show_default=True)
@click.option("--output-dim", type=int, default=None)
@click.option("--conv-kernel-size", type=int, default=5, show_default=True)
@click.option("--conv-kernel-stride", type=int, default=3, show_default=True)
@click.option("--total-steps", type=int, default=None)
@click.option("--eval-batch", type=int, default=None)
@click.option("--gradient-accumulate", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--test-fold", type=int, default=0, show_default=True)
@click.option("--dry-run", is_flag=True, help="Print the resolved config and exit.")
def run_superb(
    task: str,
    dataset_root: Path,
    target_dir: Path,
    cache_dir: Path | None,
    upstream_name: str,
    interface: str,
    device: str,
    normalize: bool,
    output_dim: int | None,
    conv_kernel_size: int,
    conv_kernel_stride: int,
    total_steps: int | None,
    eval_batch: int | None,
    gradient_accumulate: int | None,
    seed: int | None,
    test_fold: int,
    dry_run: bool,
) -> None:
    problem = get_problem(task)
    config = problem.default_config()

    config["target_dir"] = str(target_dir)
    config["cache_dir"] = str(cache_dir) if cache_dir else None
    config["device"] = device
    config["build_upstream"]["name"] = upstream_name
    config["build_featurizer"]["interface"] = interface
    config["build_featurizer"]["normalize"] = normalize
    config["build_featurizer"]["conv_kernel_size"] = conv_kernel_size
    config["build_featurizer"]["conv_kernel_stride"] = conv_kernel_stride
    config["build_featurizer"]["output_dim"] = output_dim

    dataset_key = TASK_DATASET_KEYS[task.lower()]
    config["prepare_data"][dataset_key] = str(dataset_root)

    if task.lower() == "er":
        config["prepare_data"]["test_fold"] = test_fold

    if total_steps is not None:
        config["train"]["total_steps"] = total_steps
    if eval_batch is not None:
        config["eval_batch"] = eval_batch
    if gradient_accumulate is not None:
        config["train"]["gradient_accumulate"] = gradient_accumulate
    if seed is not None:
        config["train"]["seed"] = seed

    if dry_run:
        click.echo(pformat(config, sort_dicts=False))
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    problem.run(**config)
