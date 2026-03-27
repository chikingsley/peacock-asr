"""CLI entry points for P011."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import click
import torch

from p011.data import download_features, make_loaders
from p011.models import AllLayerInterfaceModel, HierCB
from p011.settings import PROJECT_ROOT, Settings, SSLInterfaceMode
from p011.ssl_features import SSL_MODEL_KEYS
from p011.trainer import _evaluate, _forward_model, _move_batch_to_device, train_one_config

_SSL_INTERFACE_CHOICES = click.Choice([mode.value for mode in SSLInterfaceMode], case_sensitive=False)
_SSL_MODEL_HELP = f"Comma-separated subset of SSL models in {{{', '.join(SSL_MODEL_KEYS)}}}"


def _parse_ssl_interface(value: str) -> SSLInterfaceMode:
    """Parse the Click string value into the typed enum used by Settings."""
    return SSLInterfaceMode(value.lower())


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ssl_model_tag(settings: Settings) -> str:
    return "+".join(settings.ssl_models)


def _experiment_slug(settings: Settings) -> str:
    return f"{settings.ssl_interface.value}/{_ssl_model_tag(settings)}"


def _default_train_dir(settings: Settings) -> Path:
    return PROJECT_ROOT / "checkpoints" / "train" / _experiment_slug(settings) / _timestamp_tag() / f"seed{settings.seed}"


def _default_sweep_dir(settings: Settings) -> Path:
    return PROJECT_ROOT / "checkpoints" / "sweeps" / _experiment_slug(settings) / _timestamp_tag()


def _default_pretrain_dir(settings: Settings) -> Path:
    return PROJECT_ROOT / "checkpoints" / "pretrain" / _ssl_model_tag(settings) / _timestamp_tag()


def _default_run_name(settings: Settings) -> str:
    return f"{settings.ssl_interface.value}-{_ssl_model_tag(settings)}-seed{settings.seed}-{_timestamp_tag()}"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT.parent.parent,
            text=True,
        ).strip()
    except Exception:
        return None


def _write_run_manifest(path: Path, settings: Settings, *, run_name: str, extra: dict[str, object] | None = None) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "git_commit": _git_commit(),
        "settings": settings.model_dump(mode="json"),
    }
    if extra:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_wandb_dir() -> None:
    os.environ.setdefault("WANDB_DIR", str(PROJECT_ROOT / "wandb"))


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
            ssl_dim=settings.selected_ssl_dim,
            use_mdd=settings.use_mdd,
        )

    return AllLayerInterfaceModel(
        ssl_interface=settings.ssl_interface,
        ssl_output_dim=settings.resolved_ssl_output_dim,
        ssl_model_keys=settings.ssl_models,
        embed_dim=settings.embed_dim,
        num_heads=settings.num_heads,
        p_depth=settings.p_depth,
        w_depth=settings.w_depth,
        u_depth=settings.u_depth,
        ssl_drop=settings.ssl_drop,
        use_mdd=settings.use_mdd,
    )


def _make_settings(
    *,
    features_dir: str | None = None,
    seed: int | None = None,
    use_conpco: bool | None = None,
    use_mdd: bool | None = None,
    use_phnvar: bool | None = None,
    ssl_interface: SSLInterfaceMode | None = None,
    ssl_models: str | None = None,
    ssl_output_dim: int | None = None,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
) -> Settings:
    """Build Settings while only overriding fields explicitly requested by the CLI."""
    kwargs: dict[str, object] = {}
    if features_dir is not None:
        kwargs["features_dir"] = Path(features_dir)
    if seed is not None:
        kwargs["seed"] = seed
    if use_conpco is not None:
        kwargs["use_conpco"] = use_conpco
    if use_mdd is not None:
        kwargs["use_mdd"] = use_mdd
    if use_phnvar is not None:
        kwargs["use_phnvar"] = use_phnvar
    if ssl_interface is not None:
        kwargs["ssl_interface"] = ssl_interface
    if ssl_models is not None:
        kwargs["ssl_models"] = ssl_models
    if ssl_output_dim is not None:
        kwargs["ssl_output_dim"] = ssl_output_dim
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if grad_accum_steps is not None:
        kwargs["grad_accum_steps"] = grad_accum_steps
    return Settings(**kwargs)


@click.group()
def cli() -> None:
    """P011: paper-faithful frame-level HConv on top of MuFFIN."""


@cli.command()
@click.option("--features-dir", type=click.Path(), default=None, help="Override P011_FEATURES_DIR")
def download(features_dir: str | None) -> None:
    """Download pre-extracted SpeechOcean762 phone-level features."""
    settings = Settings() if features_dir is None else Settings(features_dir=Path(features_dir))
    download_features(settings.features_dir)


@cli.command()
@click.option("--checkpoint-dir", type=click.Path(), default=None)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P011_FEATURES_DIR")
@click.option("--ssl-models", type=str, default=None, help=_SSL_MODEL_HELP)
@click.option("--batch-size", type=int, default=None, help="Override batch_size")
@click.option("--grad-accum-steps", type=int, default=None, help="Override grad_accum_steps")
@click.option("--n-epochs", type=int, default=None, help="Override pretrain_epochs")
def pretrain(
    checkpoint_dir: str,
    features_dir: str | None,
    ssl_models: str | None,
    batch_size: int | None,
    grad_accum_steps: int | None,
    n_epochs: int | None,
) -> None:
    """Run self-supervised pretraining (MuFFIN §V.B, ref [41] HierTFR)."""
    from p011.pretrain import HierCBPretrain, pretrain_one_config

    settings = _make_settings(
        features_dir=features_dir,
        ssl_models=ssl_models,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
    )
    if n_epochs is not None:
        settings = settings.model_copy(update={"pretrain_epochs": n_epochs})

    _set_seed(settings.seed)
    _ensure_wandb_dir()
    train_loader, _ = make_loaders(
        settings.features_dir,
        settings.batch_size,
        ssl_model_keys=settings.ssl_models,
    )

    model = HierCBPretrain(
        embed_dim=settings.embed_dim,
        num_heads=settings.num_heads,
        p_depth=settings.p_depth,
        w_depth=settings.w_depth,
        u_depth=settings.u_depth,
        ssl_drop=settings.ssl_drop,
        ssl_dim=settings.selected_ssl_dim,
    )

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else _default_pretrain_dir(settings)
    _write_run_manifest(ckpt_dir / "run_manifest.json", settings, run_name="pretrain", extra={"kind": "pretrain"})
    best_path = pretrain_one_config(settings, model, train_loader, ckpt_dir)
    click.echo(f"Pretrained checkpoint: {best_path}")


@cli.command()
@click.option("--seed", type=int, default=22, show_default=True)
@click.option("--use-conpco", is_flag=True, default=False)
@click.option("--use-mdd", is_flag=True, default=False)
@click.option("--use-phnvar", is_flag=True, default=False)
@click.option("--ssl-interface", type=_SSL_INTERFACE_CHOICES, default=SSLInterfaceMode.HCONV.value, show_default=True)
@click.option("--ssl-models", type=str, default=None, help=_SSL_MODEL_HELP)
@click.option("--ssl-output-dim", type=int, default=None, help="Override projected SSL width after HConv/CHConv")
@click.option("--batch-size", type=int, default=None, help="Override batch_size")
@click.option("--grad-accum-steps", type=int, default=None, help="Override grad_accum_steps")
@click.option("--checkpoint-dir", type=click.Path(), default=None)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P011_FEATURES_DIR")
@click.option("--n-epochs", type=int, default=None, help="Override n_epochs (e.g. 1 for smoke test)")
@click.option("--pretrained", type=click.Path(exists=True), default=None, help="Pretrained checkpoint path")
def train(
    seed: int,
    use_conpco: bool,
    use_mdd: bool,
    use_phnvar: bool,
    ssl_interface: str,
    ssl_models: str | None,
    ssl_output_dim: int | None,
    batch_size: int | None,
    grad_accum_steps: int | None,
    checkpoint_dir: str | None,
    features_dir: str | None,
    n_epochs: int | None,
    pretrained: str | None,
) -> None:
    """Train HierCB for a single seed."""
    interface_mode = _parse_ssl_interface(ssl_interface)
    settings = _make_settings(
        seed=seed,
        use_conpco=use_conpco,
        use_mdd=use_mdd,
        use_phnvar=use_phnvar,
        ssl_interface=interface_mode,
        ssl_models=ssl_models,
        ssl_output_dim=ssl_output_dim,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        features_dir=features_dir,
    )
    if n_epochs is not None:
        settings = settings.model_copy(update={"n_epochs": n_epochs})

    _set_seed(seed)
    _ensure_wandb_dir()
    train_loader, test_loader = make_loaders(
        settings.features_dir,
        settings.batch_size,
        ssl_interface=settings.ssl_interface,
        ssl_model_keys=settings.ssl_models,
    )
    model = _build_model(settings)

    run_name = _default_run_name(settings)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else _default_train_dir(settings)
    _write_run_manifest(
        ckpt_dir / "run_manifest.json",
        settings,
        run_name=run_name,
        extra={"kind": "train", "pretrained": pretrained},
    )
    pre = Path(pretrained) if pretrained else None
    pcc = train_one_config(
        settings,
        model,
        train_loader,
        test_loader,
        run_name=run_name,
        checkpoint_dir=ckpt_dir,
        pretrained=pre,
    )
    click.echo(f"Best phone PCC: {pcc:.4f}")


@cli.command()
@click.option("--seeds", default="22,33,44,55,66", show_default=True)
@click.option("--use-conpco", is_flag=True, default=False)
@click.option("--use-mdd", is_flag=True, default=False)
@click.option("--use-phnvar", is_flag=True, default=False)
@click.option("--ssl-interface", type=_SSL_INTERFACE_CHOICES, default=SSLInterfaceMode.HCONV.value, show_default=True)
@click.option("--ssl-models", type=str, default=None, help=_SSL_MODEL_HELP)
@click.option("--ssl-output-dim", type=int, default=None, help="Override projected SSL width after HConv/CHConv")
@click.option("--batch-size", type=int, default=None, help="Override batch_size")
@click.option("--grad-accum-steps", type=int, default=None, help="Override grad_accum_steps")
@click.option("--checkpoint-dir", type=click.Path(), default=None)
@click.option("--features-dir", type=click.Path(), default=None, help="Override P011_FEATURES_DIR")
@click.option("--pretrained", type=click.Path(exists=True), default=None, help="Pretrained checkpoint path")
def sweep(
    seeds: str,
    use_conpco: bool,
    use_mdd: bool,
    use_phnvar: bool,
    ssl_interface: str,
    ssl_models: str | None,
    ssl_output_dim: int | None,
    batch_size: int | None,
    grad_accum_steps: int | None,
    checkpoint_dir: str | None,
    features_dir: str | None,
    pretrained: str | None,
) -> None:
    """Run multi-seed sweep and report mean/std PCC."""
    seed_list = [int(seed.strip()) for seed in seeds.split(",")]
    interface_mode = _parse_ssl_interface(ssl_interface)
    results: list[float] = []
    pre = Path(pretrained) if pretrained else None
    _ensure_wandb_dir()

    template_settings = _make_settings(
        seed=seed_list[0],
        use_conpco=use_conpco,
        use_mdd=use_mdd,
        use_phnvar=use_phnvar,
        ssl_interface=interface_mode,
        ssl_models=ssl_models,
        ssl_output_dim=ssl_output_dim,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        features_dir=features_dir,
    )
    sweep_root = Path(checkpoint_dir) if checkpoint_dir else _default_sweep_dir(template_settings)
    _write_run_manifest(
        sweep_root / "sweep_manifest.json",
        template_settings,
        run_name="sweep",
        extra={"kind": "sweep", "seeds": seed_list, "pretrained": pretrained},
    )

    for seed in seed_list:
        click.echo(f"\n── Seed {seed} {'─' * 40}")
        settings = _make_settings(
            seed=seed,
            use_conpco=use_conpco,
            use_mdd=use_mdd,
            use_phnvar=use_phnvar,
            ssl_interface=interface_mode,
            ssl_models=ssl_models,
            ssl_output_dim=ssl_output_dim,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            features_dir=features_dir,
        )
        _set_seed(seed)
        train_loader, test_loader = make_loaders(
            settings.features_dir,
            settings.batch_size,
            ssl_interface=settings.ssl_interface,
            ssl_model_keys=settings.ssl_models,
        )
        model = _build_model(settings)
        run_name = _default_run_name(settings)
        ckpt_dir = sweep_root / f"seed{seed}"
        _write_run_manifest(
            ckpt_dir / "run_manifest.json",
            settings,
            run_name=run_name,
            extra={"kind": "sweep-seed", "pretrained": pretrained},
        )
        pcc = train_one_config(
            settings,
            model,
            train_loader,
            test_loader,
            run_name=run_name,
            checkpoint_dir=ckpt_dir,
            pretrained=pre,
        )
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
@click.option("--ssl-models", type=str, default=None, help=_SSL_MODEL_HELP)
@click.option("--ssl-output-dim", type=int, default=None, help="Override projected SSL width after HConv/CHConv")
@click.option("--batch-size", type=int, default=None, help="Override batch_size")
@click.option("--features-dir", type=click.Path(), default=None, help="Override P011_FEATURES_DIR")
def eval_cmd(
    checkpoint: str,
    use_mdd: bool,
    ssl_interface: str,
    ssl_models: str | None,
    ssl_output_dim: int | None,
    batch_size: int | None,
    features_dir: str | None,
) -> None:
    """Evaluate a checkpoint on the test set, with optional MDD threshold search."""
    from p011.eval import grid_search_mdd_threshold

    interface_mode = _parse_ssl_interface(ssl_interface)
    settings = _make_settings(
        use_mdd=use_mdd,
        ssl_interface=interface_mode,
        ssl_models=ssl_models,
        ssl_output_dim=ssl_output_dim,
        batch_size=batch_size,
        features_dir=features_dir,
    )
    train_loader, test_loader = make_loaders(
        settings.features_dir,
        settings.batch_size,
        ssl_interface=settings.ssl_interface,
        ssl_model_keys=settings.ssl_models,
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
