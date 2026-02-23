"""CLI entry points for gopt-bench."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("gopt_bench")


def cmd_download(_args: argparse.Namespace) -> None:
    """Download the SpeechOcean762 dataset."""
    from gopt_bench.dataset import load_speechocean762  # noqa: PLC0415

    data = load_speechocean762()
    print(f"Train: {len(data.train)} utterances")  # noqa: T201
    print(f"Test:  {len(data.test)} utterances")  # noqa: T201
    print("Dataset cached by HuggingFace datasets.")  # noqa: T201


def cmd_run(args: argparse.Namespace) -> None:
    """Run GOP-SF with a specified backend and evaluate."""
    from tqdm import tqdm  # noqa: PLC0415

    from gopt_bench.backends import get_backend  # noqa: PLC0415
    from gopt_bench.dataset import load_speechocean762  # noqa: PLC0415
    from gopt_bench.evaluate import evaluate_gop  # noqa: PLC0415
    from gopt_bench.gop import compute_gop  # noqa: PLC0415

    backend_cls = get_backend(args.backend)
    backend = backend_cls()

    logger.info("Backend: %s", backend.name)

    backend.load()
    logger.info("Vocab size: %d", len(backend.vocab))

    data = load_speechocean762(limit=args.limit)
    train_utts = data.train
    test_utts = data.test

    def process_split(
        utterances: list, split_name: str
    ) -> list[tuple[str, float, float]]:
        results: list[tuple[str, float, float]] = []
        skipped = 0

        for utt in tqdm(utterances, desc=split_name, unit="utt"):
            phone_indices = []
            valid_phones = []
            valid_scores = []

            for phone, score in zip(utt.phones, utt.phone_scores, strict=True):
                idx = backend.map_phone(phone)
                if idx is not None:
                    phone_indices.append(idx)
                    valid_phones.append(phone)
                    valid_scores.append(score)

            if len(phone_indices) < 2:  # noqa: PLR2004
                skipped += 1
                continue

            gop_result = compute_gop(
                posteriors=backend.get_posteriors(utt.audio, utt.sample_rate),
                phone_indices=phone_indices,
                blank=backend.blank_index,
            )

            for phone, gop_score, human_score in zip(
                valid_phones,
                gop_result.scores,
                valid_scores,
                strict=True,
            ):
                results.append((phone, gop_score, human_score))

        logger.info(
            "[%s] %d utts, %d phones, %d skipped",
            split_name, len(utterances), len(results), skipped,
        )
        return results

    train_results = process_split(train_utts, "train")
    test_results = process_split(test_utts, "test")

    logger.info("Evaluating...")
    eval_result = evaluate_gop(train_results, test_results)

    _print_results(backend.name, eval_result, args.verbose)


def _print_results(
    name: str,
    result: object,
    verbose: bool,  # noqa: FBT001
) -> None:
    from gopt_bench.evaluate import EvalResult  # noqa: PLC0415

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
    from gopt_bench.backends import BACKEND_REGISTRY  # noqa: PLC0415

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
        prog="gopt-bench",
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
