"""CLI entry points for P010."""

from __future__ import annotations

import statistics
from pathlib import Path

import click
import torch

from p010.data import download_features, make_loaders
from p010.models import HierCB, HierCBFrameInterfaceModel
from p010.settings import Settings, SSLInterfaceMode
from p010.trainer import _evaluate, _forward_model, _move_batch_to_device, train_one_config

_SSL_INTERFACE_CHOICES = click.Choice([mode.value for mode in SSLInterfaceMode], case_sensitive=False)


def _parse_ssl_interface(value: str) -> SSLInterfaceMode:
    """Parse the Click string value into the typed enum used by Settings."""
    return SSLInterfaceMode(value.lower())


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_model(settings: Settings) -> torch.nn.Module:
    if settings.ssl_interface is SSLInterfaceMode.NONE:
        return HierCB(
            embed_dim=settings.embed_dim,
            num_heads=settings.num_heads,
            p_depth=settings.p_depth,
            w_depth=settings.w_depth,
            u_depth=settings.u_depth,
            ssl_drop=settings.ssl_drop,
            use_mdd=settings.use_mdd,
        )

    # ssl_output_dim=3072 matches the Phase 1 pretrained checkpoint's input projection
    # shape (92 GOP + 3072 SSL = 3164). HConv/CHConv project to exactly this dimension
    # so pretrained weights transfer without shape mismatch.
    return HierCBFrameInterfaceModel(
        ssl_interface=settings.ssl_interface,
        ssl_output_dim=3072,
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
    """Download pre-extracted SpeechOcean762 phone-level features."""
    settings = Settings() if features_dir is None else Settings(features_dir=Path(features_dir))
    download_features(settings.features_dir)


@cli.command()
@click.option("--audio-dir", type=click.Path(exists=True), required=True, help="Path to SpeechOcean762 WAVE/ directory")
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
@click.option("--device", type=str, default="cuda")
def extract(audio_dir: str, features_dir: str | None, device: str) -> None:
    """Extract frame-level SSL feature shards for HConv/CHConv."""
    from p010.extract import extract_split

    settings = (
        Settings(features_dir=Path(features_dir))
        if features_dir
        else Settings()
    )
    audio = Path(audio_dir)
    for split in ("train", "test"):
        click.echo(f"\n── Extracting {split} ────────────────────────────────")
        extract_split(split, settings.features_dir, audio, device=device)
    click.echo("Done.")


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
        else Settings()
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
@click.option("--ssl-interface", type=_SSL_INTERFACE_CHOICES, default=SSLInterfaceMode.NONE.value, show_default=True)
@click.option("--checkpoint-dir", type=click.Path(), default=None)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
@click.option("--n-epochs", type=int, default=None, help="Override n_epochs (e.g. 1 for smoke test)")
@click.option("--pretrained", type=click.Path(exists=True), default=None, help="Pretrained checkpoint path")
def train(
    seed: int,
    use_conpco: bool,
    use_mdd: bool,
    use_phnvar: bool,
    ssl_interface: str,
    checkpoint_dir: str | None,
    features_dir: str | None,
    n_epochs: int | None,
    pretrained: str | None,
) -> None:
    """Train HierCB for a single seed."""
    interface_mode = _parse_ssl_interface(ssl_interface)
    if features_dir:
        settings = Settings(
            seed=seed,
            use_conpco=use_conpco,
            use_mdd=use_mdd,
            use_phnvar=use_phnvar,
            ssl_interface=interface_mode,
            features_dir=Path(features_dir),
        )
    else:
        settings = Settings(
            seed=seed,
            use_conpco=use_conpco,
            use_mdd=use_mdd,
            use_phnvar=use_phnvar,
            ssl_interface=interface_mode,
        )
    if n_epochs is not None:
        settings = settings.model_copy(update={"n_epochs": n_epochs})

    _set_seed(seed)
    train_loader, test_loader = make_loaders(
        settings.features_dir,
        settings.batch_size,
        ssl_interface=settings.ssl_interface,
    )
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
@click.option("--ssl-interface", type=_SSL_INTERFACE_CHOICES, default=SSLInterfaceMode.NONE.value, show_default=True)
@click.option("--checkpoint-dir", type=click.Path(), default="checkpoints")
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
@click.option("--pretrained", type=click.Path(exists=True), default=None, help="Pretrained checkpoint path")
def sweep(
    seeds: str,
    use_conpco: bool,
    use_mdd: bool,
    use_phnvar: bool,
    ssl_interface: str,
    checkpoint_dir: str,
    features_dir: str | None,
    pretrained: str | None,
) -> None:
    """Run multi-seed sweep and report mean/std PCC."""
    seed_list = [int(seed.strip()) for seed in seeds.split(",")]
    interface_mode = _parse_ssl_interface(ssl_interface)
    results: list[float] = []
    pre = Path(pretrained) if pretrained else None

    for seed in seed_list:
        click.echo(f"\n── Seed {seed} {'─' * 40}")
        if features_dir:
            settings = Settings(
                seed=seed,
                use_conpco=use_conpco,
                use_mdd=use_mdd,
                use_phnvar=use_phnvar,
                ssl_interface=interface_mode,
                features_dir=Path(features_dir),
            )
        else:
            settings = Settings(
                seed=seed,
                use_conpco=use_conpco,
                use_mdd=use_mdd,
                use_phnvar=use_phnvar,
                ssl_interface=interface_mode,
            )
        _set_seed(seed)
        train_loader, test_loader = make_loaders(
            settings.features_dir,
            settings.batch_size,
            ssl_interface=settings.ssl_interface,
        )
        model = _build_model(settings)
        ckpt_dir = Path(checkpoint_dir) / f"seed{seed}"
        pcc = train_one_config(settings, model, train_loader, test_loader, checkpoint_dir=ckpt_dir, pretrained=pre)
        results.append(pcc)
        click.echo(f"Seed {seed} → PCC {pcc:.4f}")

    mean_pcc = statistics.mean(results)
    std_pcc = statistics.stdev(results) if len(results) > 1 else 0.0
    click.echo(f"\n{'─' * 50}")
    click.echo(f"Sweep ({len(seed_list)} seeds): PCC = {mean_pcc:.4f} ± {std_pcc:.4f}")
    click.echo(f"Individual: {[f'{value:.4f}' for value in results]}")


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--use-mdd", is_flag=True, default=False)
@click.option("--ssl-interface", type=_SSL_INTERFACE_CHOICES, default=SSLInterfaceMode.NONE.value, show_default=True)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P010_FEATURES_DIR")
def eval_cmd(checkpoint: str, use_mdd: bool, ssl_interface: str, features_dir: str | None) -> None:
    """Evaluate a checkpoint on the test set, with optional MDD threshold search."""
    from p010.eval import grid_search_mdd_threshold

    interface_mode = _parse_ssl_interface(ssl_interface)
    settings = (
        Settings(use_mdd=use_mdd, ssl_interface=interface_mode, features_dir=Path(features_dir))
        if features_dir
        else Settings(use_mdd=use_mdd, ssl_interface=interface_mode)
    )
    train_loader, test_loader = make_loaders(
        settings.features_dir,
        settings.batch_size,
        ssl_interface=settings.ssl_interface,
    )

    model = _build_model(settings)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    mdd_threshold = 0.5
    if use_mdd:
        click.echo("Running MDD threshold grid search on training set...")
        all_logit: list[torch.Tensor] = []
        all_label: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                device_batch = _move_batch_to_device(batch, device)
                outputs = _forward_model(model, device_batch, settings)
                all_logit.append(outputs[9].cpu())
                all_label.append(device_batch.mdd_label.cpu())
        mdd_threshold = grid_search_mdd_threshold(torch.cat(all_logit), torch.cat(all_label))
        click.echo(f"Best MDD threshold: {mdd_threshold:.1f}")

    metrics = _evaluate(model, test_loader, device, settings, mdd_threshold=mdd_threshold)

    click.echo("\n── Evaluation results ────────────────────────────────")
    click.echo(f"Phone:  MSE {metrics['phn_mse']:.4f}  PCC {metrics['phn_pcc']:.4f}")
    utt_aspects = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    for aspect in utt_aspects:
        click.echo(f"Utt {aspect:<13}: PCC {metrics[f'utt_pcc_{aspect}']:.4f}")
    word_aspects = ["accuracy", "stress", "total"]
    for aspect in word_aspects:
        click.echo(f"Word {aspect:<12}: PCC {metrics[f'word_pcc_{aspect}']:.4f}")
    if use_mdd:
        click.echo(
            "MDD: "
            f"F1 {metrics['mdd_f1']:.4f}  "
            f"Prec {metrics['mdd_precision']:.4f}  "
            f"Rec {metrics['mdd_recall']:.4f}"
        )
