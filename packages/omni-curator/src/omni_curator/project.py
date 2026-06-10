"""The per-language project CLI: every curate stage, parameterized by a :class:`CuratorProject`.

A language project is pure config — a ``sources.py`` (channel registry + dataset ids) and a
~20-line ``curate.py`` that builds a :class:`CuratorProject` and calls :func:`main`. ALL curation
logic and the full command set live here, so every language gets every command (and every fix)
the day it lands, instead of drifting copy-paste CLIs per project.

Commands, in pipeline order::

    list | download | cookies        # source: size + pull YouTube channel audio
    enqueue | segment | labelq       # split create pipeline (queue -> VAD -> Scribe)
    harvest | merge                  # labeled clips -> per-channel stores -> master store
    ingest                           # existing-labeled datasets (the project's registry)
    verify | rescore                 # Scribe-score the store (script-aware)
    export                           # store -> omni-parquet ablation (gated, coverage-checked)

The data layout under ``project.data`` is owned here too (``create/``, ``channels/``,
``canonical_audio/``, ``raw/``, ``datasets/``, ``clips/``, ``queue.sqlite``) — identical across
languages by construction.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.quality import OMNI_MAX_DURATION_S

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from omni_curator.create.youtube import Channel
    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore

    #: An ingest source: pulls already-labeled samples for a project (FLEURS, Common Voice, any
    #: HF dataset, a bespoke corpus loader). Registered per-project in ``CuratorProject.ingests``;
    #: the ``ingest`` subcommand's choices derive from the registry, so a new source is a new
    #: entry — never a new dataclass field or parser branch.
    IngestFn = Callable[["CuratorProject"], Iterable[Sample]]

_BATCH = 200


_TIERS = ("clean", "noisy")


@dataclass(frozen=True, kw_only=True)
class CuratorProject:
    """Everything language-specific, in one frozen config object.

    ``ingests`` maps source name -> loader (see :data:`IngestFn`; :func:`fleurs_source` and
    :func:`commonvoice_source` are the ready-made factories). ``coverage_check`` is the export
    coverage gate, injected (build one with :func:`omni_curator.coverage.char_tokenizer_coverage`);
    ``None`` disables the gate. ``heldout_manifest`` is the frozen held-out test-video manifest
    (``None`` = no carve; a configured-but-missing path fails fast). ``mixture_weights`` is the
    default sampling-weight recipe for the weighted mixture TSV.
    """

    name: str  # short project name for messages, e.g. "tajik"
    language: str  # curator language code, e.g. "tgk_Cyrl"
    script: str  # the script the create-pipeline LLM keeps labels in, e.g. "Cyrillic"
    data: Path  # project data dir; the layout below hangs off it
    db: Path  # the master CuratorStore (usually data/curator.sqlite)
    channels: Sequence[Channel] = ()
    ingests: Mapping[str, IngestFn] = field(default_factory=dict)
    env_file: Path | None = None  # KEY=VALUE file loaded into os.environ (API keys)
    coverage_check: Callable[[list[str]], int] | None = None
    heldout_manifest: Path | None = None
    mixture_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the collections and reject config typos at construction time."""
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "ingests", dict(self.ingests))
        object.__setattr__(self, "mixture_weights", dict(self.mixture_weights))
        slugs = [c.slug for c in self.channels]
        if len(slugs) != len(set(slugs)):
            dupes = sorted({s for s in slugs if slugs.count(s) > 1})
            msg = f"duplicate channel slugs: {dupes}"
            raise ValueError(msg)
        bad_tiers = sorted({c.tier for c in self.channels} - set(_TIERS))
        if bad_tiers:
            msg = f"unknown channel tiers {bad_tiers}; expected one of {_TIERS}"
            raise ValueError(msg)

    # -- the project-owned data layout ---------------------------------------------------------

    @property
    def create_dir(self) -> Path:
        """Raw channel audio (full videos, 16 kHz FLAC) awaiting labeling."""
        return self.data / "create"

    @property
    def channels_dir(self) -> Path:
        """Per-channel curator stores (labeled clips), merged into the master before export."""
        return self.data / "channels"

    @property
    def canonical_dir(self) -> Path:
        """Resampled ingest clips."""
        return self.data / "canonical_audio"

    @property
    def raw_dir(self) -> Path:
        """Transient dataset downloads (HF cache, Common Voice)."""
        return self.data / "raw"

    @property
    def datasets_dir(self) -> Path:
        """Exported omni-parquet ablations (``datasets/vN``)."""
        return self.data / "datasets"

    @property
    def queue_path(self) -> Path:
        """The split-pipeline work queue."""
        return self.data / "queue.sqlite"

    @property
    def clips_dir(self) -> Path:
        """Split-pipeline cut clips (segment output)."""
        return self.data / "clips"

    @property
    def cookies_path(self) -> Path:
        """Netscape cookies.txt, used by downloads when present (anti-bot)."""
        return self.data / "youtube_cookies.txt"

    @property
    def channels_by_slug(self) -> dict[str, Channel]:
        return {c.slug: c for c in self.channels}

    # -- shared helpers -------------------------------------------------------------------------

    def load_env(self) -> None:
        """Load KEY=VALUE lines from ``env_file`` into ``os.environ`` (existing vars win)."""
        if self.env_file is None or not self.env_file.exists():
            return
        for line in self.env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def selected_channels(self, args: argparse.Namespace) -> list[Channel]:
        if args.channel:
            return [self.channels_by_slug[args.channel]]
        return [c for c in self.channels if args.tier is None or c.tier == args.tier]

    def heldout_videos(self) -> frozenset[str]:
        """The frozen held-out test-video ids (``None`` manifest = no carve).

        A configured-but-missing manifest raises — silently exporting without the held-out
        carve would put benchmark videos into training.
        """
        if self.heldout_manifest is None:
            return frozenset()
        if not self.heldout_manifest.exists():
            msg = f"held-out manifest configured but missing: {self.heldout_manifest}"
            raise FileNotFoundError(msg)
        return frozenset(json.loads(self.heldout_manifest.read_text())["video_ids"])


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


def _labeled_video_ids(project: CuratorProject, slug: str) -> set[str]:
    """Video ids already labeled in a channel's store (``<slug>_<stem>``) — incremental skip."""
    from omni_curator.store import CuratorStore

    db = project.channels_dir / slug / "store.sqlite"
    if not db.exists():
        return set()
    store = CuratorStore(db)
    done = {s.id.rsplit("_", 1)[0] for s in store.iter_samples()}
    store.close()
    return done


# -- source stage ------------------------------------------------------------------------------


def cmd_list(project: CuratorProject, args: argparse.Namespace) -> int:
    """Size each selected channel (video counts, no download)."""
    from omni_curator.create.youtube import list_channel_videos

    for ch in project.selected_channels(args):
        ids = list_channel_videos(ch.url, limit=args.limit)
        print(f"{ch.slug:20s} {ch.tier:6s} {len(ids):>5} videos  {ch.url}")
    return 0


def cmd_download(project: CuratorProject, args: argparse.Namespace) -> int:
    """Download each selected channel's audio as 16 kHz FLAC; report hours landed."""
    from omni_curator.create.youtube import download_channel

    cookies = project.cookies_path if project.cookies_path.exists() else None
    channels = project.selected_channels(args)
    total_hours = 0.0
    for ch in channels:
        print(f"== {ch.slug} ({ch.tier}): {ch.url}")
        result = download_channel(
            ch.url, out_dir=project.create_dir / ch.slug, limit=args.limit, cookies=cookies
        )
        total_hours += result.hours
        print(f"   {result.flac_count} files, {result.hours:.2f} h"
              f" -> {project.create_dir / ch.slug}")
    print(f"TOTAL: {total_hours:.2f} h across {len(channels)} channel(s)"
          f" under {project.create_dir}")
    return 0


def cmd_cookies(project: CuratorProject, args: argparse.Namespace) -> int:
    """Refresh ``youtube_cookies.txt`` from a logged-in browser profile."""
    project.load_env()
    from omni_curator.create.youtube import refresh_cookies_from_browser

    profile = args.profile or os.environ.get("YT_COOKIES_PROFILE")
    if not profile:
        msg = "set YT_COOKIES_PROFILE in the env file (path to a logged-in Chrome profile)"
        raise SystemExit(msg)
    count = refresh_cookies_from_browser(Path(profile), project.cookies_path)
    print(f"refreshed {count} youtube.com cookies -> {project.cookies_path}")
    return 0


# -- split create pipeline ----------------------------------------------------------------------


def cmd_enqueue(project: CuratorProject, args: argparse.Namespace) -> int:
    """Seed the split-pipeline queue with not-yet-labeled channel videos (segment stage input)."""
    from omni_curator.create.queue import QueueStore, QVideo

    videos: list[QVideo] = []
    for ch in project.selected_channels(args):
        flacs = sorted((project.create_dir / ch.slug).glob("*.flac"))
        done = set() if args.all else _labeled_video_ids(project, ch.slug)
        for flac in flacs[: args.limit] if args.limit else flacs:
            video_id = f"{ch.slug}_{flac.stem}"
            if video_id in done:
                continue
            videos.append(QVideo(video_id, ch.slug, str(flac), ch.tier, ch.url))
    queue = QueueStore(project.queue_path)
    inserted = queue.enqueue_videos(videos)
    counts = queue.status_counts()
    queue.close()
    print(f"enqueued {inserted} new videos ({len(videos)} candidates) -> {project.queue_path}")
    print(f"  queue now: videos={counts['videos']} clips={counts['clips']}")
    return 0


def cmd_segment(project: CuratorProject, args: argparse.Namespace) -> int:
    """Segment stage: resident-model VAD producers cut queued videos into clips (CPU-bound)."""
    from omni_curator.create.queue import QueueStore
    from omni_curator.create.segment import run_segmenters

    run_segmenters(
        project.queue_path, procs=args.procs, clips_root=project.clips_dir,
        language=project.language, script=project.script,
        max_dur=args.max_duration, pending_hwm=args.hwm,
    )
    queue = QueueStore(project.queue_path)
    print(f"segment done. queue: {queue.status_counts()}")
    queue.close()
    return 0


def cmd_labelq(project: CuratorProject, args: argparse.Namespace) -> int:
    """Label stage: drain the clip queue with ~200-250 concurrent Scribe workers (I/O-bound)."""
    project.load_env()
    from omni_curator.create.labelq import run_labeler

    labeled = run_labeler(
        project.queue_path, workers=args.workers, batch=args.batch, runs=args.runs,
        idle_rounds=args.idle_rounds,
        on_progress=lambda n: print(f"  labeled {n}", flush=True) if n % 1000 == 0 else None,
        on_event=lambda msg: print(msg, flush=True),
    )
    print(f"labeled {labeled} clips")
    return 0


def cmd_harvest(project: CuratorProject, args: argparse.Namespace) -> int:
    """Fold labeled queue clips into the per-channel stores (idempotent insert-if-absent)."""
    from omni_curator.create.queue import QueueStore
    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore

    queue = QueueStore(project.queue_path)
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
                    id=c.clip_id, source=f"youtube-{c.channel}", language=c.language,
                    text=c.label, audio_path=c.clip_path,
                    duration=round(c.end - c.start, 3), sample_rate=16_000,
                    citation=c.citation,
                    meta={"variants": json.loads(c.variants)} if c.variants else {},
                )
            )
        for slug, samples in by_channel.items():
            if slug not in stores:
                stores[slug] = CuratorStore(project.channels_dir / slug / "store.sqlite")
            written += stores[slug].insert_if_absent(samples)
        queue.mark_harvested([c.clip_id for c in clips])
    for store in stores.values():
        store.close()
    queue.close()
    print(f"harvested {written} clips into {len(stores)} channel stores ({skipped} empty skipped)")
    return 0


def cmd_merge(project: CuratorProject, args: argparse.Namespace) -> int:  # noqa: ARG001
    """Merge the per-channel stores into the master store."""
    from omni_curator.store import CuratorStore

    master = CuratorStore(project.db)
    merged = 0
    for sub in sorted(project.channels_dir.glob("*/store.sqlite")):
        src = CuratorStore(sub)
        samples = list(src.iter_samples())
        merged += master.upsert(samples)
        src.close()
        print(f"  +{len(samples):>6} from {sub.parent.name}")
    print(
        f"merged {merged} clips -> {project.db}  "
        f"(store now {master.counts()}, {master.hours():.1f} h)"
    )
    master.close()
    return 0


# -- ingest stage -------------------------------------------------------------------------------


def fleurs_source(config: str) -> IngestFn:
    """Ready-made ingest: google/fleurs ``config`` (e.g. ``tg_tj``), splits preserved."""

    def load(project: CuratorProject) -> Iterable[Sample]:
        os.environ.setdefault("HF_HOME", str(project.raw_dir / "hf-cache"))
        from omni_curator.ingest.fleurs import load_fleurs

        return load_fleurs(
            config, language=project.language,
            audio_dir=project.canonical_dir / "fleurs", streaming=True,
        )

    return load


def commonvoice_source(datasets: Mapping[str, str]) -> IngestFn:
    """Ready-made ingest: Common Voice via the Mozilla Data Collective (name -> dataset id)."""

    def load(project: CuratorProject) -> Iterable[Sample]:
        api_key = os.environ.get("MDC_API_KEY")
        if not api_key:
            raise SystemExit("set MDC_API_KEY in the env file (Mozilla Data Collective API key)")
        from omni_curator.ingest.commonvoice import download_commonvoice, load_commonvoice
        from omni_curator.process import resample_samples

        for name, dataset_id in datasets.items():
            cv_dir = download_commonvoice(
                dataset_id, dest=project.raw_dir / "commonvoice" / name, api_key=api_key
            )
            loaded = load_commonvoice(
                cv_dir, language=project.language, source=f"commonvoice-{name}"
            )
            yield from resample_samples(loaded, project.canonical_dir / "commonvoice" / name)

    return load


def commonvoice_hf_mirror_source(
    repo: str, lang: str, *, source: str | None = None
) -> IngestFn:
    """Ready-made ingest: a Common Voice HF mirror (``fsicoli/common_voice_NN_0`` layout).

    Mozilla stopped publishing CV to HF at v17; the community mirrors keep the raw CV layout
    (``audio/<lang>/<split>/<lang>_<split>_<shard>.tar`` + ``transcript/<lang>/<split>.tsv``)
    as a *script* dataset, which modern ``datasets`` refuses to load. This downloads the raw
    files instead (hub download is cached/idempotent), flattens the tar shards into the
    ``clips/`` + per-split-tsv layout :func:`omni_curator.ingest.commonvoice.load_commonvoice`
    parses, and resamples to 16 kHz like the MDC path.
    """

    def load(project: CuratorProject) -> Iterable[Sample]:
        import json as _json
        import tarfile

        from huggingface_hub import hf_hub_download

        from omni_curator.ingest.commonvoice import load_commonvoice
        from omni_curator.process import resample_samples

        name = source or f"hf-cv-{lang}"
        os.environ.setdefault("HF_HOME", str(project.raw_dir / "hf-cache"))
        cv_dir = project.raw_dir / name
        clips = cv_dir / "clips"
        clips.mkdir(parents=True, exist_ok=True)

        def fetch(path: str) -> Path:
            return Path(hf_hub_download(repo, path, repo_type="dataset"))

        shards = _json.loads(fetch("n_shards.json").read_text(encoding="utf-8"))[lang]
        for split in ("train", "dev", "test"):
            tsv = cv_dir / f"{split}.tsv"
            if not tsv.exists():
                tsv.write_bytes(fetch(f"transcript/{lang}/{split}.tsv").read_bytes())
            for shard in range(shards[split]):
                tar_path = fetch(f"audio/{lang}/{split}/{lang}_{split}_{shard}.tar")
                with tarfile.open(tar_path) as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        target = clips / Path(member.name).name  # flatten the shard dir
                        if target.exists():
                            continue
                        extracted = tar.extractfile(member)
                        if extracted is not None:
                            target.write_bytes(extracted.read())
        loaded = load_commonvoice(cv_dir, language=project.language, source=name)
        yield from resample_samples(loaded, project.canonical_dir / name)

    return load


def huggingface_source(
    repo: str,
    *,
    config: str | None = None,
    splits: tuple[str, ...] = ("train", "dev", "test"),
    source: str | None = None,
    text_column: str | None = None,
    force_split: str | None = None,
) -> IngestFn:
    """Ready-made ingest: any HF audio dataset (column auto-detect; 16 kHz mono FLAC clips).

    ``source`` defaults to ``hf-<repo tail>`` (the store's corpus name). Gated datasets
    (Common Voice) need ``HF_TOKEN`` in the env file + accepted terms on the Hub.

    ``force_split`` overrides every ingested row's split — set ``"train"`` for third-party /
    machine-augmented datasets whose own "test" split must NOT become part of the project's
    benchmark partition (benchmark splits are exported ungated, so this is a trust decision).
    """

    def load(project: CuratorProject) -> Iterable[Sample]:
        import dataclasses

        os.environ.setdefault("HF_HOME", str(project.raw_dir / "hf-cache"))
        from omni_curator.ingest.huggingface import load_hf_audio

        name = source or f"hf-{repo.rsplit('/', 1)[-1].lower()}"
        samples = load_hf_audio(
            repo, language=project.language, source=name, config=config, splits=splits,
            text_column=text_column, audio_dir=project.canonical_dir / name,
        )
        if force_split is None:
            return samples
        return (dataclasses.replace(s, split=force_split) for s in samples)

    return load


def cmd_ingest(project: CuratorProject, args: argparse.Namespace) -> int:
    """Ingest one of the project's registered existing-labeled sources into the master store."""
    project.load_env()
    from omni_curator.store import CuratorStore

    source = project.ingests[args.dataset]
    store = CuratorStore(project.db)
    count = _store_batched(store, source(project))
    print(f"ingested {count} {args.dataset} samples -> {project.db}")
    print(f"store now: {store.counts()}  ({store.hours():.2f} h)")
    store.close()
    return 0


# -- verify stage -------------------------------------------------------------------------------


def cmd_verify(project: CuratorProject, args: argparse.Namespace) -> int:
    """Scribe-score every un-scored clip in the store (idempotent); print the spread."""
    project.load_env()
    from omni_curator.store import CuratorStore
    from omni_curator.verify import scribe_summary, verify_store

    store = CuratorStore(project.db)
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


def cmd_rescore(project: CuratorProject, args: argparse.Namespace) -> int:
    """Re-score verified rows whose hypothesis was rendered in the wrong script (no Scribe)."""
    project.load_env()
    from omni_curator.store import CuratorStore
    from omni_curator.verify import rescore_cross_script, scribe_summary

    store = CuratorStore(project.db)
    stats = rescore_cross_script(
        store, key=args.source, workers=args.workers,
        on_progress=lambda n: print(f"  rescored {n}", flush=True) if n % 5000 == 0 else None,
    )
    print(f"rescored {stats.scored}, failed {stats.failed}")
    for msg, n in stats.top_failures():
        print(f"  {n:>6}x {msg[:140]}")
    if stats.scored:
        print(f"  WER {stats.wer}  CER {stats.cer}")
    for source, summ in scribe_summary(store).items():
        print(f"  {source}: {summ}")
    store.close()
    return 0


# -- export stage -------------------------------------------------------------------------------


def _parse_weights(project: CuratorProject, values: list[str] | None) -> dict[str, float]:
    """``corpus=hours`` pairs -> dict; no flag -> the project's default recipe."""
    if not values:
        return dict(project.mixture_weights)
    weights: dict[str, float] = {}
    for value in values:
        corpus, _, hours = value.partition("=")
        if not hours:
            raise SystemExit(f"--mixture-weight expects corpus=hours, got {value!r}")
        weights[corpus] = float(hours)
    return weights


def cmd_export(project: CuratorProject, args: argparse.Namespace) -> int:
    """Materialize a dataset ablation: store -> omni-parquet under ``datasets/<name>``."""
    from omni_curator.export import Selection, export_dataset
    from omni_curator.store import CuratorStore

    heldout = frozenset() if args.no_heldout else project.heldout_videos()
    weights = {} if args.no_mixture_weights else _parse_weights(project, args.mixture_weight)
    selection = Selection(
        max_duration_seconds=args.max_duration, max_scribe_wer=args.max_wer,
        heldout_test_videos=heldout,
    )
    store = CuratorStore(project.db)
    stats = export_dataset(
        store, project.datasets_dir / args.name, version=0, selection=selection,
        coverage_check=project.coverage_check, strict=not args.no_strict,
        mixture_weights=weights or None,
    )
    store.close()
    print(f"exported {stats.rows} rows ({stats.hours:.2f} h)"
          f" -> {project.datasets_dir / args.name}")
    if heldout:
        print(f"  held-out conversational test: {len(heldout)} videos carved to split=test")
    if weights:
        print(f"  mixture weights -> language_distribution_weighted.tsv: {weights}")
    print(f"  by corpus: {stats.rows_by_corpus}")
    print(f"  by split: {stats.rows_by_split}")
    if stats.dropped_quality_total:
        print(f"  dropped by quality filter: {stats.dropped_by_quality}")
    print(f"  coverage gate <unk> rows: {stats.unk_rows}")
    return 0


# -- parser -------------------------------------------------------------------------------------


def _add_channel_args(parser: argparse.ArgumentParser, project: CuratorProject) -> None:
    parser.add_argument(
        "--channel", choices=sorted(project.channels_by_slug), help="one channel by slug"
    )
    parser.add_argument("--tier", choices=("clean", "noisy"), help="only this tier")
    parser.add_argument("--limit", type=int, help="cap to the first N videos per channel")


def _add_source_parsers(sub: argparse._SubParsersAction, project: CuratorProject) -> None:
    p_list = sub.add_parser("list", help="size channels (video counts, no download)")
    _add_channel_args(p_list, project)
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download channel audio -> data/create/<slug>")
    _add_channel_args(p_dl, project)
    p_dl.set_defaults(func=cmd_download)

    p_ck = sub.add_parser("cookies", help="refresh youtube_cookies.txt from the browser profile")
    p_ck.add_argument("--profile", help="Chrome profile dir (default: $YT_COOKIES_PROFILE)")
    p_ck.set_defaults(func=cmd_cookies)


def _add_create_parsers(sub: argparse._SubParsersAction, project: CuratorProject) -> None:
    p_eq = sub.add_parser("enqueue", help="seed the queue with not-yet-labeled videos")
    _add_channel_args(p_eq, project)
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


def _add_store_parsers(sub: argparse._SubParsersAction, project: CuratorProject) -> None:
    p_mg = sub.add_parser("merge", help="merge per-channel stores into the master store")
    p_mg.set_defaults(func=cmd_merge)

    p_in = sub.add_parser("ingest", help="ingest an existing-labeled dataset into the store")
    p_in.add_argument("dataset", choices=sorted(project.ingests))
    p_in.set_defaults(func=cmd_ingest)

    p_vf = sub.add_parser("verify", help="Scribe-score un-scored clips in the store")
    p_vf.add_argument("--source", help="restrict to one source (e.g. fleurs)")
    p_vf.add_argument("--scribe-language", help="Scribe language (default auto)")
    p_vf.add_argument("--workers", type=int, default=100)
    p_vf.add_argument("--force", action="store_true", help="re-score already-scored clips")
    p_vf.set_defaults(func=cmd_verify)

    p_rs = sub.add_parser("rescore", help="re-score wrong-script hypotheses (no Scribe calls)")
    p_rs.add_argument("--source", help="restrict to one source")
    p_rs.add_argument("--workers", type=int, default=50)
    p_rs.set_defaults(func=cmd_rescore)

    p_ex = sub.add_parser("export", help="store -> omni-parquet ablation (coverage-gated)")
    p_ex.add_argument("name", help="ablation dir under datasets/ (e.g. v4)")
    p_ex.add_argument("--max-wer", type=float, default=None, help="drop clips above this WER")
    p_ex.add_argument("--max-duration", type=float, default=OMNI_MAX_DURATION_S)
    p_ex.add_argument("--no-strict", action="store_true", help="warn instead of fail on <unk>")
    p_ex.add_argument("--no-heldout", action="store_true",
                      help="do NOT carve the held-out test videos to split=test")
    p_ex.add_argument("--mixture-weight", action="append", metavar="CORPUS=HOURS",
                      help="sampling-weight override for the weighted TSV (repeatable; "
                      "default: the project recipe)")
    p_ex.add_argument("--no-mixture-weights", action="store_true",
                      help="write no weighted TSV (true hours only)")
    p_ex.set_defaults(func=cmd_export)


def build_parser(project: CuratorProject) -> argparse.ArgumentParser:
    """The full curate CLI for one language project."""
    parser = argparse.ArgumentParser(
        description=f"Curate {project.name} ASR data via omni-curator."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_source_parsers(sub, project)
    _add_create_parsers(sub, project)
    _add_store_parsers(sub, project)
    return parser


def main(project: CuratorProject, argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected command."""
    args = build_parser(project).parse_args(argv)
    return int(args.func(project, args))
