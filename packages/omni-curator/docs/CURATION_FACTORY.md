# Curation Factory Operating Plan

Status: current plan as of 2026-06-25. This is the canonical curation/factory plan.
`CHANGELOG.md` is the historical record. `TODO.md` is the active backlog.

## Objective

Build a curation system that can run multi-language YouTube/audio curation without uncontrolled
GPU, Scribe API, or disk pressure.

The system must:

- keep SQLite and manifests as the durable source of truth for each stage;
- keep hot SSD use bounded and explainable;
- recover cleanly after process death, reboot, Scribe failures, and GPU OOM;
- publish, archive, or delete artifacts only after verification;
- keep current behavior here, completed work in `CHANGELOG.md`, and open work in `TODO.md`.

## Current Decisions

| Area | Decision |
|---|---|
| Scheduler | Use one polling factory. SQLite is truth; polling reconciles state; `flock` is liveness. |
| Factory-owned stages | `enqueue`, `segment`, `labelq`, `harvest`, and `archive`. |
| Post-factory workflows | `merge`, `verify`, `export`, and `publish` are explicit operator/release workflows, not factory stages. |
| Segment launch policy | No hard global segment cap. Segment launches are gated per project by locks, pending-clip HWM, clip-disk free space, worker lifecycle, and claim-token output ownership. |
| Scribe API budget | One global API budget is split across all live Scribe jobs: `labelq` and any manual `verify`. |
| Project working roots | `/mnt/tiny-2t/peacock-asr/<project>` backs the active language-project `data` symlinks except Russian. |
| Workers SSD | `/mnt/workerssd-2t` holds Russian working audio and the active `peacock-clips` scratch root. |
| Archive root | `/mnt/massive-22t/peacock-asr-archive` is the only cold archive root. |
| Documentation | This file is current. Stale plan files are deleted after their useful content is represented here. |

## Stage Names

Use these names in code, logs, docs, and runbooks.

| Stage | Contract | Normal owner |
|---|---|---|
| `prescan` | Resolve configured channels, check reachability, and record lane/status/count/error in `data/prescan.sqlite`. | Manual/source-preflight. |
| `download` | Fetch channel/source audio into an active source cache. | Manual today; factory-supervised later. |
| `enqueue` | Scan active source FLACs into `queue.sqlite.videos`. | Factory one-shot. |
| `segment` | Run VAD, cut source audio into clip FLACs, then write `queue.sqlite.clips`. | Factory daemon after P0 readiness. |
| `labelq` | Send queued clips through Scribe pass(es), fuse variants, and write labels to queue rows. Default is one `auto` Scribe pass unless configured otherwise. | Factory daemon. |
| `harvest` | Insert labeled queue rows into per-channel stores. | Factory one-shot. |
| `archive` | Move segmented source FLACs from hot/work storage to cold archive. | Factory one-shot. |
| `merge` | Insert per-channel store rows into the language master store. | Manual/post-factory. |
| `verify` | Scribe-score master-store rows for quality gates. | Manual/post-factory; shares the Scribe API budget when running. |
| `export` | Materialize a selected training dataset. | Manual/release. |
| `publish` | Upload/release artifacts and verify remote state. | Manual/release. |

## Source Metadata

YouTube source metadata is carried through the pipeline instead of being stranded in the source
registry:

1. `Channel` entries carry `tier` and `category`.
2. `prescan` records channel reachability and intended lane in `data/prescan.sqlite`.
3. `download` writes yt-dlp `*.info.json` sidecars.
4. `enqueue` stores bounded video metadata (`title`, `description`, `upload_date`, channel URL,
   duration, tags/categories) on `queue.sqlite.videos`.
5. `segment` copies `tier`, `category`, and metadata into clip rows.
6. `harvest` writes those fields into `Sample.meta`.
7. `export` writes row-level `metadata` JSON next to the training columns and license fields.
8. `export --youtube-stratified-splits` applies the 17-genre taxonomy and deterministic,
   category-stratified, whole-video dev/test assignment. Existing exports are unchanged unless the
   flag is passed.

## Verified Storage State

The workers SSD pressure is Russian canonical audio, not SQLite.

| Path | Approx size | Verified origin / writer |
|---|---:|---|
| `/mnt/tiny-2t/peacock-asr/{dari,farsi,georgian,tajik}-asr/data/create` | `~709G` total | Active source-audio landing zones for project `data/create` symlinks. |
| `/mnt/workerssd-2t/peacock-asr/russian-asr/canonical_audio` | `~1.2T` | Russian ingest working root. |
| `/mnt/workerssd-2t/peacock-asr/russian-asr/curator.sqlite` | `~8.4G` | Russian master SQLite store. Not the main space consumer. |
| `/mnt/workerssd-2t/peacock-clips` | bounded scratch | Active clip cache used by `segment --clips-root` and factory launches. |

Archive state:

- Canonical root: `/mnt/massive-22t/peacock-asr-archive`.
- Current top-level buckets: `dari`, `farsi`, `georgian`, `russian`, `tajik`.
- Removed stale language bucket: `/mnt/massive-22t/peacock-asr-archive/persian`.
- Former `persian/iran_international` material is preserved at
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy` pending an off-hours
  checksum dedup into `farsi/iran_international`.
- Previous pre-renaming archive root is gone.
- On 2026-06-25, 678 completed files (`17.6G`) were migrated from the removed root into the
  canonical root.
- Four stale `.partial` files in the removed root were deleted only after their original source
  FLACs were verified under the then-current project working roots.
- Empty old manifests and empty directories were removed.

Storage rules:

1. `/mnt/tiny-2t` owns active project `data` symlinks for Dari, Farsi, Georgian, Tajik, and CV26.
   Keep at least `250G` free; do not put new clip caches here.
2. Source audio lands in the owning project's `data/create` tree by default. Do not create top-level
   source-cache roots; use `--create-root` only for an explicit, temporary scratch relocation.
3. `/mnt/workerssd-2t/peacock-clips` is the active clip cache. Soft cap: `450G`.
4. `/mnt/workerssd-2t/peacock-asr/russian-asr/canonical_audio` is frozen working debt, not general
   scratch. It must be published or moved before the workers SSD is treated as a clean factory disk.
5. `/mnt/massive-22t/peacock-asr-archive` is the cold source-audio archive and release staging area.
   Keep at least `2T` free for large upload/download retries.
6. `/mnt/media-5t` is not an ASR write target.
7. Local deletion requires verified copy or verified HF publication.
8. Do not create new top-level storage namespaces without adding them here and to the config surface.

## P0 - Atomic Correctness

Factory production readiness requires all of this work to be true at the same time:

1. VAD never emits a clip above the configured max duration.
2. Segment parent death terminates child workers and releases GPU memory; verify with `nvidia-smi`.
3. Stale `segmenting` videos are reclaimed before fresh pending work.
4. Segment output publication is ownership-safe: workers cut into claim-token staging dirs and
   publish final clip files only inside token-guarded `complete_video()`.
5. `resegment` refuses active stage locks and refuses existing channel stores unless explicitly
   overridden.
6. Archive moves use the canonical root only and do not trust same-size destination files as proof of
   identical content.
7. Segment launch gating is based on real resource predicates: pending clip HWM, free GB, active
   locks, worker lifecycle, and output ownership. A permanent global cap is not used.
8. Scribe API calls are controlled by one global budget shared by `labelq` and any live `verify`.
9. Operational logs stay outside the repo root.

Exit criteria:

- focused tests cover VAD duration, orphan cleanup, stale claim priority, resegment guards, archive
  safety, factory backpressure, and Scribe budget splitting;
- a killed segment parent leaves no orphaned GPU workers;
- a dry run explains every launch/hold decision;
- `uv run --project packages/omni-curator pytest ...`, Ruff, and `git diff --check` pass on touched
  files.

## P1 - Factory Configuration

Implemented:

1. `omni_curator.factory --config <path>` reads one flat TOML config for roots, projects, stages,
   tick interval, segment worker counts, pending HWM, minimum free GB, archive root, and Scribe
   budget.
2. CLI flags override config values.
3. `--once --dry-run` remains the safe operator preview and does not write Scribe window files.
4. Dry-run logs `RUN`, `HOLD`, and `BALANCE` decisions with resource blockers.
5. The factory prints to stdout by default; it appends to a file only when `--log-file` or
   `log_file` config is set. No repo-root `factory.log` is created by default.

Still open:

1. Make `archive_needed()` cheaper than scanning every segmented source path every tick.

Exit criteria:

- factory defaults cannot fill the SSD;
- factory config matches documented roots;
- resource blockers are visible without reading the code.

## P2 - Storage Lifecycle

1. Treat `/mnt/workerssd-2t` as hot storage, not archive.
2. Keep active source and clip caches bounded.
3. Move cold source audio to `/mnt/massive-22t/peacock-asr-archive` after segmentation.
4. Publish or move Russian canonical audio before growing new hot corpora.
5. Avoid dumping millions of tiny clip files to exFAT cold storage unless sharded or intentionally
   accepted.

Exit criteria:

- per-tier usage is visible without manual detective work;
- local deletion requires verified copy or verified HF publication;
- no process silently creates a new top-level storage namespace.

## Quality Gates

WER here means the store-level Scribe verification score: the stored label scored against a fresh
Scribe transcription of the same clip. It is not a model eval WER.

Recording-type tiers:

| Recording type | Excellent | Good | Acceptable |
|---|---:|---:|---:|
| Read / broadcast: scripted Common Voice, FLEURS, audiobooks | `<= 5%` | `<= 15%` | `<= 25%` |
| Conversational / spontaneous: interviews, calls, drill audio, YouTube shows | `<= 15%` | `<= 35%` | `<= 60%` |

Fallback by language resource level when recording type is not enough: high-resource `<= 20%`,
medium-resource `<= 30%`, low-resource `<= 50%`.

Duration:

1. Never export above `OMNI_MAX_DURATION_S = 40s`; Omni ASR truncates input audio at 40 seconds.
2. Keep NeMo's `0.3s` minimum as a cheap artifact filter when the corpus can tolerate it.
3. The lower bound is not model-constant; CTC only requires enough encoder frames for the label.

Per-second text rates are a physical-plausibility backstop, not the main gate. Compute
`chars_per_second` and `words_per_second` on normalized labels to catch audio/text mismatch where
the transcript is too dense or too sparse for the clip duration. NeMo defines the metrics but does
not publish universal caps, so set language caps from a physiology anchor
(`~8 syllables/sec * script chars/syllable`) or just above the corpus `p99.9`.

Code constants live in `packages/omni-curator/src/omni_curator/audit/quality.py`. Primary source
docs are NVIDIA NeMo Curator WER filtering and audio quality metrics; speech-rate rationale comes
from the cross-language `~39 bits/sec` result and speech-tempo literature.

## P3 - VAD Throughput

Correctness comes before speed.

Order:

1. enforce max duration;
2. clean up child process lifecycle;
3. share decoded 16 kHz mono audio between VAD and cutting where memory permits;
4. batch/bucket VAD inputs;
5. evaluate sharded output only after Scribe/labelq can consume shard members or staged temp clips.

Exit criteria:

- VAD speed experiments include equivalence checks against current windows;
- no speed change reintroduces over-length clips or orphaned GPU workers.

## P4 - Hugging Face Release Lifecycle

1. Build release artifacts into an explicit staging root with a free-space preflight.
2. Keep one append-only release state log: `converted`, `uploaded`, `verified`, `complete`.
3. Use one chosen upload method per release: `hf upload-large-folder`, per-shard commits, or explicit
   Xet/LFS workflow.
4. Verify remote siblings and re-download/checksum representative artifacts.
5. Record shipped names, verification, and deletion decisions in `CHANGELOG.md`.
6. Delete or move local hot artifacts only after verification.

Exit criteria:

- no competing upload attempts for the same artifact set;
- remote state is verified before local cleanup;
- `CHANGELOG.md` records what shipped and what local data became safe to move/delete.

## Documentation Cleanup

Keep:

- `packages/omni-curator/docs/CURATION_FACTORY.md` as current plan;
- `CHANGELOG.md` as historical record;
- `TODO.md` as active backlog.

Rewrite or move:

- none currently identified.

Delete after replacement:

- none currently identified.

Exit criteria:

- no semi-current plan files remain;
- operators have one current plan and one active backlog.
