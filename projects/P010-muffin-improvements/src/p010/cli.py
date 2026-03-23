"""CLI entry points for P010.

Usage:
    uv run p010 download
    uv run p010 pretrain
    uv run p010 train --seed 22 --use-conpco --use-mdd --pretrained checkpoints/pretrained/best_pretrain.pth
    uv run p010 sweep --seeds 22,33,44,55,66 --use-conpco --use-mdd \
        --pretrained checkpoints/pretrained/best_pretrain.pth
    uv run p010 eval --checkpoint path/to/best_model.pth
"""

from __future__ import annotations

import statistics
from pathlib import Path

import click
import torch

from p010.data import download_features, make_loaders
from p010.models.hiercb import HierCB
from p010.settings import Settings
from p010.trainer import train_one_config


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_model(settings: Settings) -> HierCB:
    return HierCB(
        embed_dim=settings.embed_dim,
        num_heads=settings.num_heads,
        p_depth=settings.p_depth,
        w_depth=settings.w_depth,
        u_depth=settings.u_depth,
        ssl_drop=settings.ssl_drop,
        use_mdd=settings.use_mdd,
    )


@click.group()
def cli() -> None:
    """P010: MuFFIN replication + CHConv improvements."""


@cli.command()
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
def download(features_dir: str | None) -> None:
    """Download pre-extracted SpeechOcean762 features from HuggingFace Hub."""
    settings = Settings() if features_dir is None else Settings(features_dir=Path(features_dir))  # type: ignore[missing-argument]
    download_features(settings.features_dir)


@cli.command()
@click.option("--checkpoint-dir", type=click.Path(), default="checkpoints/pretrained")
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
@click.option("--n-epochs", type=int, default=None, help="Override pretrain_epochs")
def pretrain(checkpoint_dir: str, features_dir: str | None, n_epochs: int | None) -> None:
    """Run self-supervised pretraining (MuFFIN §V.B, ref [41] HierTFR)."""
    from p010.pretrain import HierCBPretrain, pretrain_one_config

    settings = (
        Settings(features_dir=Path(features_dir))
        if features_dir
        else Settings()  # type: ignore[missing-argument]
    )
    if n_epochs is not None:
        settings = settings.model_copy(update={"pretrain_epochs": n_epochs})

    _set_seed(settings.seed)
    train_loader, _ = make_loaders(settings.features_dir, settings.batch_size)

    model = HierCBPretrain(
        embed_dim=settings.embed_dim,
        num_heads=settings.num_heads,
        p_depth=settings.p_depth,
        w_depth=settings.w_depth,
        u_depth=settings.u_depth,
        ssl_drop=settings.ssl_drop,
    )

    best_path = pretrain_one_config(settings, model, train_loader, Path(checkpoint_dir))
    click.echo(f"Pretrained checkpoint: {best_path}")


@cli.command()
@click.option("--seed", type=int, default=22, show_default=True)
@click.option("--use-conpco", is_flag=True, default=False)
@click.option("--use-mdd", is_flag=True, default=False)
@click.option("--use-phnvar", is_flag=True, default=False)
@click.option("--checkpoint-dir", type=click.Path(), default=None)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
@click.option("--n-epochs", type=int, default=None, help="Override n_epochs (e.g. 1 for smoke test)")
@click.option("--pretrained", type=click.Path(exists=True), default=None, help="Pretrained checkpoint path")
def train(
    seed: int, use_conpco: bool, use_mdd: bool, use_phnvar: bool,
    checkpoint_dir: str | None, features_dir: str | None, n_epochs: int | None,
    pretrained: str | None,
) -> None:
    """Train HierCB for a single seed."""
    if features_dir:
        settings = Settings(seed=seed, use_conpco=use_conpco, use_mdd=use_mdd, use_phnvar=use_phnvar,
                            features_dir=Path(features_dir))
    else:
        settings = Settings(seed=seed, use_conpco=use_conpco, use_mdd=use_mdd, use_phnvar=use_phnvar)  # type: ignore[missing-argument]
    if n_epochs is not None:
        settings = settings.model_copy(update={"n_epochs": n_epochs})
    _set_seed(seed)

    train_loader, test_loader = make_loaders(settings.features_dir, settings.batch_size)
    model = _build_model(settings)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    pre = Path(pretrained) if pretrained else None
    pcc = train_one_config(settings, model, train_loader, test_loader, checkpoint_dir=ckpt_dir, pretrained=pre)
    click.echo(f"Best phone PCC: {pcc:.4f}")


@cli.command()
@click.option("--seeds", default="22,33,44,55,66", show_default=True)
@click.option("--use-conpco", is_flag=True, default=False)
@click.option("--use-mdd", is_flag=True, default=False)
@click.option("--use-phnvar", is_flag=True, default=False)
@click.option("--checkpoint-dir", type=click.Path(), default="checkpoints")
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
@click.option("--pretrained", type=click.Path(exists=True), default=None, help="Pretrained checkpoint path")
def sweep(
    seeds: str, use_conpco: bool, use_mdd: bool, use_phnvar: bool,
    checkpoint_dir: str, features_dir: str | None, pretrained: str | None,
) -> None:
    """Run multi-seed sweep, report mean +/- std PCC."""
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    results = []
    pre = Path(pretrained) if pretrained else None
    for seed in seed_list:
        click.echo(f"\n── Seed {seed} {'─'*40}")
        if features_dir:
            settings = Settings(seed=seed, use_conpco=use_conpco, use_mdd=use_mdd, use_phnvar=use_phnvar,
                                features_dir=Path(features_dir))
        else:
            settings = Settings(seed=seed, use_conpco=use_conpco, use_mdd=use_mdd, use_phnvar=use_phnvar)  # type: ignore[missing-argument]
        _set_seed(seed)
        train_loader, test_loader = make_loaders(settings.features_dir, settings.batch_size)
        model = _build_model(settings)
        ckpt_dir = Path(checkpoint_dir) / f"seed{seed}"
        pcc = train_one_config(settings, model, train_loader, test_loader, checkpoint_dir=ckpt_dir, pretrained=pre)
        results.append(pcc)
        click.echo(f"Seed {seed} → PCC {pcc:.4f}")

    mean_pcc = statistics.mean(results)
    std_pcc = statistics.stdev(results) if len(results) > 1 else 0.0
    click.echo(f"\n{'─'*50}")
    click.echo(f"Sweep ({len(seed_list)} seeds): PCC = {mean_pcc:.4f} ± {std_pcc:.4f}")
    click.echo(f"Individual: {[f'{r:.4f}' for r in results]}")


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--use-mdd", is_flag=True, default=False)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
def eval_cmd(checkpoint: str, use_mdd: bool, features_dir: str | None) -> None:
    """Evaluate a saved checkpoint on the test set, with MDD threshold grid search."""
    from p010.eval import grid_search_mdd_threshold
    from p010.trainer import _evaluate

    settings = (
        Settings(use_mdd=use_mdd, features_dir=Path(features_dir))
        if features_dir
        else Settings(use_mdd=use_mdd)  # type: ignore[missing-argument]
    )
    train_loader, test_loader = make_loaders(settings.features_dir, settings.batch_size)

    model = _build_model(settings)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # MDD threshold grid search on training set (MuFFIN §V.B)
    mdd_threshold = 0.5
    if use_mdd:
        click.echo("Running MDD threshold grid search on training set...")
        all_logit, all_label = [], []
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                tensors = [t.to(device, non_blocking=True) for t in batch]
                word_label_b = tensors[7]
                out = model(tensors[0], tensors[2], tensors[3], tensors[1], tensors[5],
                            word_label_b[:, :, 3], tensors[8])
                all_logit.append(out[9].cpu())   # mdd_logit
                all_label.append(tensors[9].cpu())  # mdd_label
        mdd_threshold = grid_search_mdd_threshold(torch.cat(all_logit), torch.cat(all_label))
        click.echo(f"Best MDD threshold: {mdd_threshold:.1f}")

    metrics = _evaluate(model, test_loader, device, settings, mdd_threshold=mdd_threshold)

    click.echo("\n── Evaluation results ────────────────────────────────")
    click.echo(f"Phone:  MSE {metrics['phn_mse']:.4f}  PCC {metrics['phn_pcc']:.4f}")
    utt_aspects = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    for asp in utt_aspects:
        click.echo(f"Utt {asp:<13}: PCC {metrics[f'utt_pcc_{asp}']:.4f}")
    word_aspects = ["accuracy", "stress", "total"]
    for asp in word_aspects:
        click.echo(f"Word {asp:<12}: PCC {metrics[f'word_pcc_{asp}']:.4f}")
    if use_mdd:
        click.echo(
            f"MDD:    F1 {metrics['mdd_f1']:.4f}  "
            f"P {metrics['mdd_precision']:.4f}  R {metrics['mdd_recall']:.4f}"
        )
        if "mdd_far" in metrics:
            click.echo(
                f"Diag:   FAR {metrics['mdd_far']:.4f}  FRR {metrics['mdd_frr']:.4f}  "
                f"DER {metrics['mdd_der']:.4f}  PER {metrics['mdd_per']:.4f}"
            )


# Register eval command under the name 'eval' in the CLI
cli.add_command(eval_cmd, name="eval")
