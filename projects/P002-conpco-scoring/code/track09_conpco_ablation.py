"""
Track 09 Phase 1: ConPCO loss ablation on GOPT + GOP-SF pipeline.

Tests whether ConPCO loss terms improve our GOPT phone-level scorer,
keeping architecture and features (42-dim GOP-SF) unchanged.

Ablation matrix:
    p1a: GOPT + MSE only (baseline)
    p1b: GOPT + MSE + Ordinal Entropy (diversity + tightness)
    p1c: GOPT + MSE + OE + CLAP contrastive (full ConPCO)

Requires cached GOP-SF features from a prior
`uv run --project projects/P001-gop-baselines peacock-asr run --backend original --gopt`
run.

Usage:
    uv run --project projects/P002-conpco-scoring python \
        projects/P002-conpco-scoring/code/track09_conpco_ablation.py \
        --ablation p1b --seed 42
    uv run --project projects/P002-conpco-scoring python \
        projects/P002-conpco-scoring/code/track09_conpco_ablation.py \
        --ablation p1c --seed 42 --n-epochs 5
"""

import argparse
import logging
import os
import random
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

from p002_conpco.gopt_track09 import train_and_evaluate_gopt_conpco
from p002_conpco.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("track09")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track 09 Phase 1: ConPCO loss ablation on GOPT",
    )
    parser.add_argument(
        "--ablation", type=str, default="p1b",
        choices=["p1a", "p1b", "p1c"],
        help="p1a=MSE only, p1b=MSE+OE, p1c=MSE+OE+CLAP",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3)
    # ConPCO hyperparams (paper defaults)
    parser.add_argument("--lambda-d", type=float, default=0.5)
    parser.add_argument("--lambda-t", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda-clap-t2a", type=float, default=0.5)
    parser.add_argument("--w-mse", type=float, default=1.0)
    parser.add_argument("--w-oe", type=float, default=1.0)
    parser.add_argument("--w-clap", type=float, default=1.0)
    parser.add_argument("--tag", type=str, default="track09-p1")
    return parser.parse_args()


def load_cached_features() -> tuple[list, list, int]:
    """Load GOP-SF features from the standard cache location.

    Returns (train_utts, test_utts, input_dim).
    """
    features_dir = settings.features_dir

    # Search for original backend cache with gop_sf features
    # Backend dir names include model qualifier, e.g. "original__checkpoint-8000_"
    train_path = test_path = None
    for backend_dir in sorted(features_dir.iterdir()):
        if not backend_dir.is_dir() or not backend_dir.name.startswith("original"):
            continue
        candidate = backend_dir / "gop_sf_a0p5000"
        if (candidate / "train.pt").exists() and (candidate / "test.pt").exists():
            train_path = candidate / "train.pt"
            test_path = candidate / "test.pt"
            logger.info("Found cached features at %s", candidate)
            break
        # Also check old-format cache (before score variants were added)
        if (backend_dir / "train.pt").exists() and (backend_dir / "test.pt").exists():
            train_path = backend_dir / "train.pt"
            test_path = backend_dir / "test.pt"
            logger.info("Found cached features (old format) at %s", backend_dir)
            break

    if train_path is None or test_path is None:
        logger.error(
            "Cached GOP-SF features not found under %s/original*/. "
            "Run `uv run --project projects/P001-gop-baselines "
            "peacock-asr run --backend original --gopt` first.",
            features_dir,
        )
        raise SystemExit(1)

    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    train_utts = train_data["utt_feats"]
    test_utts = test_data["utt_feats"]

    # Infer feature dimension from first utterance
    input_dim = len(train_utts[0].feat_vecs[0])

    logger.info(
        "Loaded %d train, %d test utterances (feat_dim=%d)",
        len(train_utts), len(test_utts), input_dim,
    )
    return train_utts, test_utts, input_dim


def main() -> None:
    args = get_args()

    # Seed everything
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Ablation config
    use_oe = args.ablation in ("p1b", "p1c")
    use_clap = args.ablation == "p1c"
    ablation_label = {
        "p1a": "MSE-only",
        "p1b": "MSE+OE",
        "p1c": "MSE+OE+CLAP",
    }[args.ablation]

    logger.info("Ablation: %s (%s), seed=%d", args.ablation, ablation_label, args.seed)

    # Load cached features
    train_utts, test_utts, input_dim = load_cached_features()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    run_name = f"track09-{args.ablation}-seed{args.seed}"
    run_prefix = os.environ.get("PEACOCK_WANDB_RUN_PREFIX", "p002")
    run_name = f"{run_prefix} | {run_name}" if run_prefix else run_name
    group = os.environ.get("PEACOCK_WANDB_GROUP", "p002-track09-p1-ablation")
    job_type = os.environ.get("PEACOCK_WANDB_JOB_TYPE", "ablation")
    extra_tags_raw = os.environ.get("PEACOCK_WANDB_TAGS", "")
    extra_tags = [tag.strip() for tag in extra_tags_raw.split(",") if tag.strip()]

    wandb_config: dict[str, object] = {
        "track": "09",
        "project_id": "P002",
        "phase": "p1",
        "ablation": args.ablation,
        "ablation_label": ablation_label,
        "use_ordinal_entropy": use_oe,
        "use_clap": use_clap,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "input_dim": input_dim,
        "features": "GOP-SF (42-dim)",
        "architecture": "GOPT",
        "lambda_d": args.lambda_d,
        "lambda_t": args.lambda_t,
        "margin": args.margin,
        "lambda_clap_t2a": args.lambda_clap_t2a,
        "w_mse": args.w_mse,
        "w_oe": args.w_oe,
        "w_clap": args.w_clap,
        "wandb_group": group,
        "wandb_job_type": job_type,
    }
    wandb_project = None
    wandb_entity = None
    if not os.environ.get("WANDB_SWEEP_ID"):
        wandb_project = os.environ.get(
            "PEACOCK_WANDB_PROJECT",
            "peacock-asr-p002-conpco-scoring",
        )
        wandb_entity = os.environ.get("PEACOCK_WANDB_ENTITY", "peacockery")
    wandb.init(
        name=run_name,
        group=group,
        job_type=job_type,
        config=wandb_config,
        tags=["p002", "track09", "p1", args.ablation, args.tag, *extra_tags],
        project=wandb_project,
        entity=wandb_entity,
    )

    history: list[dict[str, float]] = []

    def on_epoch_end(
        epoch: int,
        total: int,
        loss: float,
        lr: float,
        time_sec: float,
    ) -> None:
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": loss,
                "train/loss_mse": history[-1]["loss_mse"],
                "train/loss_oe": history[-1]["loss_oe"],
                "train/loss_clap": history[-1]["loss_clap"],
                "lr": lr,
                "epoch_time_sec": time_sec,
            }
        )
        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d  loss=%.4f  lr=%.2e  (%.1fs)",
                epoch,
                total,
                loss,
                lr,
                time_sec,
            )

    result = train_and_evaluate_gopt_conpco(
        train_utts,
        test_utts,
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        use_ordinal_entropy=use_oe,
        use_clap=use_clap,
        w_mse=args.w_mse,
        w_oe=args.w_oe,
        w_clap=args.w_clap,
        lambda_d=args.lambda_d,
        lambda_t=args.lambda_t,
        margin=args.margin,
        lambda_clap_t2a=args.lambda_clap_t2a,
        history_out=history,
        on_epoch_end=on_epoch_end,
    )

    # Log final result
    wandb.log(
        {
            "final/phn_pcc": result.pcc,
            "final/phn_mse": result.mse,
            "final/n_phones": result.n_phones,
        }
    )
    wandb.summary["phn_pcc"] = result.pcc
    wandb.summary["phn_mse"] = result.mse

    logger.info(
        "Done. PCC=%.4f  MSE=%.4f  n_phones=%d",
        result.pcc, result.mse, result.n_phones,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
