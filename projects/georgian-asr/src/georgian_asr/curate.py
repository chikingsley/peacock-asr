"""Curate Georgian ASR data through omni-curator. Thin wiring of sources.py into the curator stages.

  georgian-curate ingest fleurs|commonvoice   # existing-labeled datasets -> the curator store
  georgian-curate download [--tier clean]      # YouTube channel audio -> data/create/<slug>
  georgian-curate enqueue && segment & labelq  # split create pipeline: VAD producers + Scribe pool
  georgian-curate harvest && merge             # labeled queue clips -> per-channel stores -> master
  georgian-curate verify [--source fleurs]     # Scribe-score every un-scored clip (idempotent)
  georgian-curate export v0 [--max-wer 0.25]   # store -> omni-parquet ablation (coverage-gated)

The curator store (``data/curator.sqlite``) is the master pool; every source funnels through it.
All curation LOGIC is in omni-curator — this only supplies the Georgian sources (``sources.py``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.quality import OMNI_MAX_DURATION_S

from georgian_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore

CREATE = DATA / "create"  # raw channel audio (full videos, 16 kHz FLAC) awaiting labeling
CHANNELS = DATA / "channels"  # per-channel curator stores (labeled clips), merged before export
CANONICAL = DATA / "canonical_audio"  # resampled ingest clips
RAW = DATA / "raw"  # transient dataset downloads
DATASETS = DATA / "datasets"  # exported ablations (datasets/vN)
QUEUE = DATA / "queue.sqlite"  # split-pipeline work queue (segment producer -> labelq consumer)
CLIPS = DATA / "clips"  # split-pipeline cut clips (segment output)
TOKENIZER = Path(__file__).resolve().parent / "models" / "omniASR_tokenizer_written_v2.model"
COOKIES = DATA / "youtube_cookies.txt"  # optional Netscape cookies.txt; used if present (anti-bot)
_BATCH = 200


def _load_root_env() -> None:
    """Load KEY=VALUE lines from the monorepo-root .env (MDC_API_KEY, Scribe key, HF_TOKEN, ...)."""
    env_path = ROOT.parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _selected_channels(args: argparse.Namespace) -> list[sources.Channel]:
    if args.channel:
        return [sources.CHANNELS_BY_SLUG[args.channel]]
    return [c for c in sources.YOUTUBE_CHANNELS if args.tier is None or c.tier == args.tier]


def _store_batched(store: CuratorStore, samples: Iterable[Sample]) -> int:
    """Upsert a stream of samples into the store in batches; return the count written."""
    count = 0
    batch: list[Sample] = []
    for sample in samples:
        batch.append(sample)
        count += 1
        if len(batch) >= _BATCH:
            store.upsert(batch)
            batch = []
    if batch:
        store.upsert(batch)
    return count


def cmd_list(args: argparse.Namespace) -> int:
    """Size each selected YouTube channel (video counts, no download)."""
    from omni_curator.create.youtube import list_channel_videos

    for ch in _selected_channels(args):
        ids = list_channel_videos(ch.url, limit=args.limit)
        print(f"{ch.slug:20s} {ch.tier:6s} {len(ids):>5} videos  {ch.url}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download each selected channel's audio as 16 kHz FLAC; report hours landed."""
    from omni_curator.create.youtube import download_channel

    cookies = COOKIES if COOKIES.exists() else None
    channels = _selected_channels(args)
    total_hours = 0.0
    for ch in channels:
        print(f"== {ch.slug} ({ch.tier}): {ch.url}")
        result = download_channel(
            ch.url, out_dir=CREATE / ch.slug, limit=args.limit, cookies=cookies
        )
        total_hours += result.hours
        print(f"   {result.flac_count} files, {result.hours:.2f} h -> {CREATE / ch.slug}")
    print(f"TOTAL: {total_hours:.2f} h across {len(channels)} channel(s) under {CREATE}")
    return 0


def cmd_cookies(args: argparse.Namespace) -> int:
    """Refresh ``data/youtube_cookies.txt`` from the logged-in browser profile (KasmVNC Chrome)."""
    _load_root_env()
    from omni_curator.create.youtube import refresh_cookies_from_browser

    profile = args.profile or os.environ.get("YT_COOKIES_PROFILE")
    if not profile:
        msg = "set YT_COOKIES_PROFILE in the root .env (path to a logged-in Chrome profile)"
        raise SystemExit(msg)
    count = refresh_cookies_from_browser(Path(profile), COOKIES)
    print(f"refreshed {count} youtube.com cookies -> {COOKIES}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest an existing-labeled dataset (FLEURS / Common Voice) into the curator store."""
    _load_root_env()
    from omni_curator.store import CuratorStore

    store = CuratorStore(DB)
    if args.dataset == "fleurs":
        os.environ.setdefault("HF_HOME", str(RAW / "hf-cache"))
        from omni_curator.ingest.fleurs import load_fleurs

        count = _store_batched(
            store,
            load_fleurs(
                sources.FLEURS_CONFIG, language=LANGUAGE,
                audio_dir=CANONICAL / "fleurs", streaming=True,
            ),
        )
    else:  # commonvoice
        if not sources.COMMONVOICE:
            store.close()
            raise SystemExit("no Common Voice dataset ids in sources.COMMONVOICE")
        api_key = os.environ.get("MDC_API_KEY")
        if not api_key:
            store.close()
            raise SystemExit("set MDC_API_KEY in the root .env (Mozilla Data Collective API key)")
        from omni_curator.ingest.commonvoice import download_commonvoice, load_commonvoice
        from omni_curator.process import resample_samples

        count = 0
        for name, dataset_id in sources.COMMONVOICE.items():
            cv_dir = download_commonvoice(
                dataset_id, dest=RAW / "commonvoice" / name, api_key=api_key
            )
            loaded = load_commonvoice(cv_dir, language=LANGUAGE, source=f"commonvoice-{name}")
            count += store.upsert(resample_samples(loaded, CANONICAL / "commonvoice" / name))

    print(f"ingested {count} {args.dataset} samples -> {DB}")
    print(f"store now: {store.counts()}  ({store.hours():.2f} h)")
    store.close()
    return 0


def _labeled_video_ids(slug: str) -> set[str]:
    """Video ids already labeled in a channel's store (``<slug>_<stem>``) — for incremental skip."""
    from omni_curator.store import CuratorStore

    db = CHANNELS / slug / "store.sqlite"
    if not db.exists():
        return set()
    store = CuratorStore(db)
    done = {s.id.rsplit("_", 1)[0] for s in store.iter_samples()}
    store.close()
    return done


def cmd_enqueue(args: argparse.Namespace) -> int:
    """Seed the split-pipeline queue with not-yet-labeled channel videos (segment stage input)."""
    from omni_curator.create.queue import QueueStore, QVideo

    videos: list[QVideo] = []
    for ch in _selected_channels(args):
        flacs = sorted((CREATE / ch.slug).glob("*.flac"))
        done = set() if args.all else _labeled_video_ids(ch.slug)
        for flac in flacs[: args.limit] if args.limit else flacs:
            video_id = f"{ch.slug}_{flac.stem}"
            if video_id in done:
                continue
            videos.append(QVideo(video_id, ch.slug, str(flac), ch.tier, ch.url))
    queue = QueueStore(QUEUE)
    inserted = queue.enqueue_videos(videos)
    counts = queue.status_counts()
    queue.close()
    print(f"enqueued {inserted} new videos ({len(videos)} candidates) -> {QUEUE}")
    print(f"  queue now: videos={counts['videos']} clips={counts['clips']}")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    """Segment stage: resident-model VAD producers cut queued videos into clips (CPU-bound)."""
    from omni_curator.create.segment import run_segmenters

    run_segmenters(
        QUEUE, procs=args.procs, clips_root=CLIPS, language=LANGUAGE, script=SCRIPT,
        max_dur=args.max_duration, pending_hwm=args.hwm,
    )
    from omni_curator.create.queue import QueueStore

    queue = QueueStore(QUEUE)
    print(f"segment done. queue: {queue.status_counts()}")
    queue.close()
    return 0


def cmd_labelq(args: argparse.Namespace) -> int:
    """Label stage: drain the clip queue with ~200-250 concurrent Scribe workers (I/O-bound)."""
    _load_root_env()
    from omni_curator.create.labelq import run_labeler

    labeled = run_labeler(
        QUEUE, workers=args.workers, batch=args.batch, runs=args.runs,
        idle_rounds=args.idle_rounds,
        on_progress=lambda n: print(f"  labeled {n}", flush=True) if n % 1000 == 0 else None,
        on_event=lambda msg: print(msg, flush=True),
    )
    print(f"labeled {labeled} clips")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    """Fold labeled queue clips into the per-channel stores (idempotent insert-if-absent)."""
    import json

    from omni_curator.create.queue import QueueStore
    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore

    queue = QueueStore(QUEUE)
    stores: dict[str, CuratorStore] = {}
    written = skipped = 0
    while True:
        clips = queue.harvestable(args.batch)
        if not clips:
            break
        by_channel: dict[str, list[Sample]] = {}
        for c in clips:
            if not c.label.strip():  # empty label: nothing to train on, but still mark harvested
                skipped += 1
                continue
            by_channel.setdefault(c.channel, []).append(
                Sample(
                    id=c.clip_id, source=f"youtube-{c.channel}", language=c.language, text=c.label,
                    audio_path=c.clip_path, duration=round(c.end - c.start, 3), sample_rate=16_000,
                    citation=c.citation,
                    meta={"variants": json.loads(c.variants)} if c.variants else {},
                )
            )
        for slug, samples in by_channel.items():
            if slug not in stores:
                stores[slug] = CuratorStore(CHANNELS / slug / "store.sqlite")
            written += stores[slug].insert_if_absent(samples)
        queue.mark_harvested([c.clip_id for c in clips])
    for store in stores.values():
        store.close()
    queue.close()
    print(f"harvested {written} clips into {len(stores)} channel stores ({skipped} empty skipped)")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Merge the per-channel stores into the master ``curator.sqlite`` (FLEURS already there)."""
    from omni_curator.store import CuratorStore

    master = CuratorStore(DB)
    merged = 0
    for sub in sorted(CHANNELS.glob("*/store.sqlite")):
        src = CuratorStore(sub)
        samples = list(src.iter_samples())
        merged += master.upsert(samples)
        src.close()
        print(f"  +{len(samples):>6} from {sub.parent.name}")
    print(f"merged {merged} clips -> {DB}  (store now {master.counts()}, {master.hours():.1f} h)")
    master.close()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Scribe-score every un-scored clip in the store (idempotent); print the spread."""
    _load_root_env()
    from omni_curator.store import CuratorStore
    from omni_curator.verify import scribe_summary, verify_store

    store = CuratorStore(DB)
    stats = verify_store(
        store, key=args.source, scribe_language=args.scribe_language,
        workers=args.workers, force=args.force,
    )
    renew_note = f", key renewals {stats.renewals}" if stats.renewals else ""
    print(f"scored {stats.scored}, skipped {stats.skipped}, failed {stats.failed}{renew_note}")
    for msg, n in stats.top_failures():
        print(f"  {n:>6}x {msg[:140]}")
    if stats.scored:
        print(f"  WER {stats.wer}  CER {stats.cer}")
    for source, summ in scribe_summary(store).items():
        print(f"  {source}: {summ}")
    store.close()
    return 0


def _coverage_check(texts: list[str]) -> int:
    """Count rows whose normalized text produces a tokenizer ``<unk>`` (the export gate)."""
    from fairseq2.data.tokenizers.char import load_char_tokenizer
    from omni_finetune_core.tokenizer_audit import audit_texts

    return audit_texts(texts, load_char_tokenizer(TOKENIZER, None)).unk_rows


def cmd_export(args: argparse.Namespace) -> int:
    """Materialize a dataset ablation: store -> omni-parquet under data/datasets/<name>."""
    from omni_curator.export import Selection, export_dataset
    from omni_curator.store import CuratorStore

    selection = Selection(max_duration_seconds=args.max_duration, max_scribe_wer=args.max_wer)
    store = CuratorStore(DB)
    stats = export_dataset(
        store, DATASETS / args.name, version=0, selection=selection,
        coverage_check=_coverage_check, strict=not args.no_strict,
    )
    store.close()
    print(f"exported {stats.rows} rows ({stats.hours:.2f} h) -> {DATASETS / args.name}")
    print(f"  by corpus: {stats.rows_by_corpus}")
    if stats.dropped_quality_total:
        print(f"  dropped by quality filter: {stats.dropped_by_quality}")
    print(f"  coverage gate <unk> rows: {stats.unk_rows}")
    return 0


def _add_channel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--channel", choices=sorted(sources.CHANNELS_BY_SLUG), help="one channel by slug"
    )
    parser.add_argument("--tier", choices=("clean", "noisy"), help="only this tier")
    parser.add_argument("--limit", type=int, help="cap to the first N videos per channel")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate Georgian ASR data via omni-curator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="size channels (video counts, no download)")
    _add_channel_args(p_list)
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download channel audio -> data/create/<slug>")
    _add_channel_args(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_ck = sub.add_parser("cookies", help="refresh youtube_cookies.txt from the browser profile")
    p_ck.add_argument("--profile", help="Chrome profile dir (default: $YT_COOKIES_PROFILE)")
    p_ck.set_defaults(func=cmd_cookies)

    p_eq = sub.add_parser("enqueue", help="seed the queue with not-yet-labeled videos")
    _add_channel_args(p_eq)
    p_eq.add_argument("--all", action="store_true", help="ignore the already-labeled skip")
    p_eq.set_defaults(func=cmd_enqueue)

    p_sg = sub.add_parser("segment", help="VAD-segment queued videos into clips (CPU)")
    p_sg.add_argument("--procs", type=int, default=6, help="resident-model segment processes")
    p_sg.add_argument("--max-duration", type=float, default=30.0, help="hard cap per VAD span (s)")
    p_sg.add_argument("--hwm", type=int, default=50_000, help="pending-clip backpressure ceiling")
    p_sg.set_defaults(func=cmd_segment)

    p_lq = sub.add_parser("labelq", help="drain the clip queue with Scribe workers (I/O)")
    p_lq.add_argument("--workers", type=int, default=200, help="concurrent Scribe calls")
    p_lq.add_argument("--batch", type=int, default=None, help="clips per claim (default 2x worker)")
    p_lq.add_argument("--runs", type=int, default=1, help="Scribe ensemble runs per clip")
    p_lq.add_argument("--idle-rounds", type=int, default=3, help="empty polls before exit")
    p_lq.set_defaults(func=cmd_labelq)

    p_hv = sub.add_parser("harvest", help="fold labeled clips into per-channel stores")
    p_hv.add_argument("--batch", type=int, default=2000)
    p_hv.set_defaults(func=cmd_harvest)

    p_mg = sub.add_parser("merge", help="merge per-channel stores into the master curator.sqlite")
    p_mg.set_defaults(func=cmd_merge)

    p_in = sub.add_parser("ingest", help="ingest an existing-labeled dataset into the store")
    p_in.add_argument("dataset", choices=("fleurs", "commonvoice"))
    p_in.set_defaults(func=cmd_ingest)

    p_vf = sub.add_parser("verify", help="Scribe-score un-scored clips in the store")
    p_vf.add_argument("--source", help="restrict to one source (e.g. fleurs)")
    p_vf.add_argument("--scribe-language", help="Scribe language (default auto)")
    p_vf.add_argument("--workers", type=int, default=100)
    p_vf.add_argument("--force", action="store_true", help="re-score already-scored clips")
    p_vf.set_defaults(func=cmd_verify)

    p_ex = sub.add_parser("export", help="store -> omni-parquet ablation (coverage-gated)")
    p_ex.add_argument("name", help="ablation dir under data/datasets/ (e.g. v0)")
    p_ex.add_argument("--max-wer", type=float, default=None, help="drop clips above this WER")
    p_ex.add_argument("--max-duration", type=float, default=OMNI_MAX_DURATION_S)
    p_ex.add_argument("--no-strict", action="store_true", help="warn instead of fail on <unk>")
    p_ex.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
