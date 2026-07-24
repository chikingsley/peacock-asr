"""``cv26`` command line: download | process | queue | card | status.

``process`` is the steady-state loop (convert -> upload -> verify -> delete). ``download`` fetches
archives from the manifest. The two run concurrently in ``--watch`` mode: the downloader lands
archives while the processor drains them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from cv26 import config, status
from cv26.hub import hf_card
from cv26.hub.upload import hf_api, upload_file
from cv26.manifest import load_manifest, select_rows
from cv26.mdc.download import (
    DEFAULT_MAX_SLEEP_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_WORKERS,
    IDLE_SLEEP_SECONDS,
    MIN_FREE_BYTES,
    DownloadConfig,
    free_bytes,
    pending_rows,
    run_batch,
)
from cv26.mdc.queue import add_from_text
from cv26.pipeline import RunConfig, load_state, run

_GB = 1024**3
# Don't re-download datasets already fully handled — their archives were uploaded then deleted.
_DONE_STATES = frozenset({"complete", "no_shards"})


def _download_pending(args: argparse.Namespace, workers: int, failed: set[str]) -> list:
    """Select downloadable rows not on disk, not done, and not known-failed; capped per cycle."""
    rows = select_rows(load_manifest(config.MANIFEST), dataset_ids=args.dataset_id)
    done = {
        ds for ds, status in load_state(config.PROCESSING_STATE).items() if status in _DONE_STATES
    }
    candidates = [
        row
        for row in pending_rows(rows, config.ARCHIVES, force=args.force)
        if row.dataset_id not in failed and row.dataset_id not in done
    ]
    if args.max_downloads:
        candidates = candidates[: args.max_downloads]
    return candidates[:workers] if args.watch else candidates


def _cmd_download(args: argparse.Namespace) -> None:
    cfg = DownloadConfig(
        token="" if args.dry_run else config.require_mdc_key(),
        out_dir=config.ARCHIVES,
        report_path=config.DOWNLOAD_REPORT,
        retries=args.retries,
        max_sleep=args.max_sleep,
        workers=max(1, args.workers),
        force=args.force,
    )
    if args.dry_run:
        for row in select_rows(load_manifest(config.MANIFEST), dataset_ids=args.dataset_id):
            print(f"{row.dataset_id} -> {row.filename}")
        return
    failed: set[str] = set()  # datasets that errored this session (e.g. access not granted) — skip
    while True:
        candidates = _download_pending(args, cfg.workers, failed)
        if not candidates:
            if not args.watch:
                return
            time.sleep(IDLE_SLEEP_SECONDS)
            continue
        if free_bytes(config.DATA) < MIN_FREE_BYTES:
            msg = f"low disk (< {MIN_FREE_BYTES // _GB} GB free); let `cv26 process` drain first"
            if not args.watch:
                raise SystemExit(msg)
            logging.getLogger("cv26.cli").warning("%s; waiting", msg)
            time.sleep(IDLE_SLEEP_SECONDS)
            continue
        failed |= run_batch(candidates, cfg)
        if not args.watch:
            return


def _cmd_process(args: argparse.Namespace) -> None:
    if args.delete_after_upload and not args.upload:
        raise SystemExit("--delete-after-upload requires --upload")
    cfg = RunConfig(
        archive_dir=config.ARCHIVES,
        parquet_root=config.PARQUET_ROOT,
        state_path=config.PROCESSING_STATE,
        repo_id=args.repo_id,
        upload=args.upload,
        create_repo=args.create_repo,
        delete_after_upload=args.delete_after_upload,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        dataset_ids=tuple(args.dataset_id),
    )
    run(cfg, watch=args.watch)


def _cmd_queue(args: argparse.Namespace) -> None:
    text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    add_from_text(text)
    if args.refresh_card:
        _cmd_card(argparse.Namespace(repo_id=args.repo_id, upload=True))


def _cmd_status(_: argparse.Namespace) -> None:
    status.report()


def _cmd_card(args: argparse.Namespace) -> None:
    readme, license_path = hf_card.write_card()
    if not args.upload:
        return
    api = hf_api(config.require_hf_token())
    upload_file(api, args.repo_id, readme, "README.md", "Refresh dataset card")
    upload_file(api, args.repo_id, license_path, "LICENSE", "Refresh license")
    upload_file(api, args.repo_id, config.MANIFEST, "manifests/datasets.jsonl", "Refresh manifest")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``cv26`` argument parser."""
    parser = argparse.ArgumentParser(prog="cv26", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download MDC archives from the manifest")
    download.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    download.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    download.add_argument("--max-sleep", type=int, default=DEFAULT_MAX_SLEEP_SECONDS)
    download.add_argument("--max-downloads", type=int, default=0, help="0 means no limit")
    download.add_argument("--dataset-id", action="append", default=[])
    download.add_argument("--force", action="store_true", help="re-download archives on disk")
    download.add_argument("--watch", action="store_true", help="poll the manifest for new rows")
    download.add_argument("--dry-run", action="store_true")
    download.set_defaults(func=_cmd_download)

    process = sub.add_parser("process", help="convert downloaded archives, upload, verify, delete")
    process.add_argument("--repo-id", default=config.REPO_ID)
    process.add_argument("--dataset-id", action="append", default=[])
    process.add_argument("--limit", type=int, default=0)
    process.add_argument("--upload", action="store_true")
    process.add_argument("--create-repo", action="store_true", help="create the repo if missing")
    process.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="delete archive+shards after verify",
    )
    process.add_argument("--overwrite", action="store_true", help="rewrite existing parquet shards")
    process.add_argument("--watch", action="store_true", help="keep processing as archives land")
    process.add_argument("--dry-run", action="store_true")
    process.set_defaults(func=_cmd_process)

    queue = sub.add_parser("queue", help="add languages from pasted MDC curl snippets")
    queue.add_argument("input", nargs="?", type=Path, help="snippet file (default: stdin)")
    queue.add_argument("--repo-id", default=config.REPO_ID)
    queue.add_argument("--refresh-card", action="store_true", help="regenerate + upload card")
    queue.set_defaults(func=_cmd_queue)

    card_cmd = sub.add_parser("card", help="regenerate the dataset card (README + LICENSE)")
    card_cmd.add_argument("--repo-id", default=config.REPO_ID)
    card_cmd.add_argument("--upload", action="store_true", help="push README/LICENSE/manifest")
    card_cmd.set_defaults(func=_cmd_card)

    status_cmd = sub.add_parser("status", help="progress per collection + which need MDC terms")
    status_cmd.set_defaults(func=_cmd_status)

    return parser


def main() -> None:
    """Entry point for the ``cv26`` console script."""
    config.configure_logging()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
