"""CLI entry points for peacock-asr."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("peacock_asr")


def cmd_download(_args: argparse.Namespace) -> None:
    """Download the SpeechOcean762 dataset."""
    from peacock_asr.dataset import load_speechocean762  # noqa: PLC0415

    data = load_speechocean762()
    print(f"Train: {len(data.train)} utterances")  # noqa: T201
    print(f"Test:  {len(data.test)} utterances")  # noqa: T201
    print("Dataset cached by HuggingFace datasets.")  # noqa: T201


def cmd_run(args: argparse.Namespace) -> None:
    """Run GOP-SF with a specified backend and evaluate."""
    from tqdm import tqdm  # noqa: PLC0415

    from peacock_asr.backends import get_backend  # noqa: PLC0415
    from peacock_asr.dataset import load_speechocean762  # noqa: PLC0415
    from peacock_asr.gop import compute_gop  # noqa: PLC0415
    from peacock_asr.gopt_model import UtteranceFeats  # noqa: PLC0415
    from peacock_asr.settings import settings  # noqa: PLC0415

    use_feats = getattr(args, "feats", False)
    use_gopt = getattr(args, "gopt", False)
    extract_features = use_feats or use_gopt
    if getattr(args, "device", None) is not None:
        settings.device = args.device
    device = settings.torch_device

    backend = get_backend(args.backend)()

    logger.info("Backend: %s", backend.name)

    backend.load()
    logger.info("Vocab size: %d  Device: %s", len(backend.vocab), device)

    if extract_features:
        logger.info(
            "Feature extraction: ON (batched ctc_loss, %d tokens)",
            len(backend.vocab),
        )

    data = load_speechocean762(limit=args.limit)

    def process_split(
        utterances: list, split_name: str,
    ) -> tuple[list, list[UtteranceFeats]]:
        """Process a split, returning (scalar_results, utt_feats).

        scalar_results: always populated (phone, gop_score, human_score)
        utt_feats: populated only when extract_features is True
        """
        scalar_results: list[tuple[str, float, float]] = []
        utt_feats: list[UtteranceFeats] = []
        skipped = 0

        for utt in tqdm(utterances, desc=split_name, unit="utt"):
            phone_indices: list[int] = []
            valid_phones: list[str] = []
            valid_scores: list[float] = []

            for phone, score in zip(
                utt.phones, utt.phone_scores, strict=True,
            ):
                indices = backend.map_phone(phone)
                if indices is not None:
                    phone_indices.extend(indices)
                    valid_phones.append(phone)
                    valid_scores.append(score)

            if len(phone_indices) < 2:  # noqa: PLR2004
                skipped += 1
                continue

            gop_result = compute_gop(
                posteriors=backend.get_posteriors(
                    utt.audio, utt.sample_rate,
                ),
                phone_indices=phone_indices,
                blank=backend.blank_index,
                extract_features=extract_features,
                device=device if extract_features else None,
            )

            for phone, gop_score, human_score in zip(
                valid_phones, gop_result.scores,
                valid_scores, strict=True,
            ):
                scalar_results.append(
                    (phone, gop_score, human_score),
                )

            if extract_features and gop_result.features is not None:
                feat_vecs = [
                    [
                        *gop_result.features[i].tolist(),
                        gop_result.occupancies[i],
                    ]
                    for i in range(len(valid_phones))
                ]
                utt_feats.append(UtteranceFeats(
                    phones=valid_phones,
                    feat_vecs=feat_vecs,
                    scores=valid_scores,
                ))

        logger.info(
            "[%s] %d utts, %d phones, %d skipped",
            split_name, len(utterances),
            len(scalar_results), skipped,
        )
        return scalar_results, utt_feats

    train_scalar, train_utts = process_split(data.train, "train")
    test_scalar, test_utts = process_split(data.test, "test")

    feat_dim = len(backend.vocab) + 2  # LPP + VxLPR + occupancy

    _run_evaluation(
        use_gopt=use_gopt,
        use_feats=use_feats,
        backend_name=backend.name,
        feat_dim=feat_dim,
        device=device,
        train_scalar=train_scalar,
        test_scalar=test_scalar,
        train_utts=train_utts,
        test_utts=test_utts,
        verbose=args.verbose,
    )


def _run_evaluation(  # noqa: PLR0913
    *,
    use_gopt: bool,
    use_feats: bool,
    backend_name: str,
    feat_dim: int,
    device: object,
    train_scalar: list,
    test_scalar: list,
    train_utts: list,
    test_utts: list,
    verbose: bool,
) -> None:
    """Dispatch to the appropriate evaluation method."""
    from peacock_asr.evaluate import evaluate_gop, evaluate_gop_feats  # noqa: PLC0415

    if use_gopt:
        from peacock_asr.gopt_model import (  # noqa: PLC0415
            train_and_evaluate_gopt,
        )

        logger.info(
            "Training GOPT transformer (feat_dim=%d)...", feat_dim,
        )
        eval_result = train_and_evaluate_gopt(
            train_utts, test_utts,
            input_dim=feat_dim,
            device=device,
        )
        _print_results(
            backend_name + " (GOPT)", eval_result, verbose,
        )
    elif use_feats:
        # Flatten utterance records into per-phone tuples for SVR
        train_flat = [
            (p, f, s)
            for u in train_utts
            for p, f, s in zip(
                u.phones, u.feat_vecs, u.scores, strict=True,
            )
        ]
        test_flat = [
            (p, f, s)
            for u in test_utts
            for p, f, s in zip(
                u.phones, u.feat_vecs, u.scores, strict=True,
            )
        ]
        logger.info("Evaluating with SVR on feature vectors...")
        eval_result = evaluate_gop_feats(train_flat, test_flat)
        _print_results(
            backend_name + " (SVR+feats)", eval_result, verbose,
        )
    else:
        logger.info(
            "Evaluating with poly regression on scalar scores...",
        )
        eval_result = evaluate_gop(train_scalar, test_scalar)
        _print_results(backend_name, eval_result, verbose)


def _print_results(
    name: str,
    result: object,
    verbose: bool,  # noqa: FBT001
) -> None:
    from peacock_asr.evaluate import EvalResult  # noqa: PLC0415

    if not isinstance(result, EvalResult):
        return
    sep = "=" * 50
    print(f"\n{sep}")  # noqa: T201
    print(f"Backend:    {name}")  # noqa: T201
    print(f"Phones:     {result.n_phones}")  # noqa: T201
    print(f"PCC:        {result.pcc:.4f}")  # noqa: T201
    lo, hi = result.pcc_low, result.pcc_high
    print(f"PCC 95% CI: [{lo:.4f}, {hi:.4f}]")  # noqa: T201
    print(f"MSE:        {result.mse:.4f}")  # noqa: T201
    print(sep)  # noqa: T201

    if verbose:
        print("\nPer-phone PCC:")  # noqa: T201
        items = sorted(result.per_phone_pcc.items(), key=lambda x: -x[1])
        for phone, pcc in items:
            print(f"  {phone:4s}  {pcc:.4f}")  # noqa: T201


def cmd_compare(args: argparse.Namespace) -> None:
    """Run all available backends and compare."""
    from peacock_asr.backends import BACKEND_REGISTRY  # noqa: PLC0415

    available = list(BACKEND_REGISTRY.keys())
    print(f"Available backends: {', '.join(available)}")  # noqa: T201
    print()  # noqa: T201

    for name in available:
        sep = "=" * 50
        print(sep)  # noqa: T201
        print(f"Running: {name}")  # noqa: T201
        print(sep)  # noqa: T201
        run_args = argparse.Namespace(backend=name, limit=args.limit, verbose=False)
        try:
            cmd_run(run_args)
        except Exception as e:  # noqa: BLE001
            print(f"  Skipped {name}: {type(e).__name__}: {e}")  # noqa: T201
        print()  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="peacock-asr",
        description="Pronunciation assessment benchmark",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download SpeechOcean762 dataset")

    run_p = sub.add_parser("run", help="Run GOP-SF with a backend")
    run_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend: original, xlsr-espeak, zipa",
    )
    run_p.add_argument(
        "--feats",
        action="store_true",
        help="Extract full feature vectors (LPP+LPR) and evaluate with SVR",
    )
    run_p.add_argument(
        "--gopt",
        action="store_true",
        help="Train GOPT transformer on feature vectors and evaluate",
    )
    run_p.add_argument(
        "--device",
        default=None,
        help="Compute device: auto, cpu, cuda (default: auto)",
    )
    run_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )

    cmp_p = sub.add_parser("compare", help="Run all backends and compare")
    cmp_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    match args.command:
        case "download":
            cmd_download(args)
        case "run":
            cmd_run(args)
        case "compare":
            cmd_compare(args)
        case _:
            parser.print_help()
            sys.exit(1)
