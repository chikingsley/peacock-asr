# Curation Factory Operating Plan

Status: current plan as of 2026-07-09. This is the canonical curation/factory plan. `CHANGELOG.md` is the historical record. `TODO.md` is the active backlog.

## Objective

Build a curation system that can run multi-language YouTube/audio curation without uncontrolled GPU, Scribe API, or disk pressure.

The system must:

- keep SQLite and manifests as the durable source of truth for each stage;
- keep hot SSD use bounded and explainable;
- recover cleanly after process death, reboot, Scribe failures, and GPU OOM;
- publish, archive, or delete artifacts only after verification;
- keep current behavior here, completed work in `CHANGELOG.md`, and open work in `TODO.md`.

## Current Decisions

| Area                    | Decision                                                                                                                                                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scheduler               | Use one polling factory. SQLite is truth; polling reconciles state; `flock` is liveness.                                                                                                                                                  |
| Factory-owned stages    | `enqueue`, `segment`, `labelq`, `harvest`, and `archive`.                                                                                                                                                                                 |
| Post-factory workflows  | `merge`, `verify`, `export`, and `publish` are explicit operator/release workflows, not factory stages.                                                                                                                                   |
| Segment launch policy   | No hard global segment cap. Segment launches are gated per project by locks, pending-clip HWM, clip-disk free space, worker lifecycle, and claim-token output ownership.                                                                  |
| VAD engine contract     | Cobra, Silero, and the pinned benchmark MarbleNet adapter from `a-vad-bench` commit `a838e7f` consume the same decoded 16 kHz mono audio and return raw speech intervals. Peacock queue/publication machinery remains engine-independent. |
| Interval policy         | One versioned Peacock postprocessor owns padding, minimum speech, gap merging, and hard splitting. Keep raw intervals and make emitted maximum duration profile/model-aware; 30 seconds is not a universal VAD truth.                     |
| VAD routing             | Route explicitly by project/language and recording profile. Support clean/read and noisy/conversational profiles; validate policy changes with bounded same-audio pilots before scale.                                                    |
| Segmentation provenance | Clip metadata and zero-output video rows store engine, model revision, native threshold/options, resolved runtime/backend, postprocessing profile, and policy revision. A separate durable production-run record remains open.            |
| Scribe API budget       | One global API budget is split across all live Scribe jobs: `labelq` and any manual `verify`.                                                                                                                                             |
| Project working roots   | `/mnt/tiny-2t/peacock-asr/<project>` backs the active language-project `data` symlinks except Russian.                                                                                                                                    |
| Workers SSD             | `/mnt/workerssd-2t` holds Russian working audio. It is only a clip scratch target when an operator explicitly passes `--clips-root`.                                                                                                      |
| Archive root            | `/mnt/massive-22t/peacock-asr-archive` is the only cold archive root.                                                                                                                                                                     |
| Documentation           | This file is current. Stale plan files are deleted after their useful content is represented here.                                                                                                                                        |

## Stage Names

Use these names in code, logs, docs, and runbooks.

| Stage      | Contract                                                                                                                                                 | Normal owner                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `prescan`  | Resolve configured channels, check reachability, and record lane/status/count/error in `data/prescan.sqlite`.                                            | Manual/source-preflight.                                        |
| `download` | Fetch channel/source audio into an active source cache.                                                                                                  | Manual today; factory-supervised later.                         |
| `enqueue`  | Scan active source FLACs into `queue.sqlite.videos`.                                                                                                     | Factory one-shot.                                               |
| `segment`  | Decode once, run the routed VAD adapter plus shared postprocessor, cut clip FLACs, and write `queue.sqlite.clips` with segmentation provenance.          | Factory daemon after P0 readiness.                              |
| `labelq`   | Send queued clips through Scribe pass(es), fuse variants, and write labels to queue rows. Default is one `auto` Scribe pass unless configured otherwise. | Factory daemon.                                                 |
| `harvest`  | Insert labeled queue rows into per-channel stores.                                                                                                       | Factory one-shot.                                               |
| `archive`  | Move segmented source FLACs from hot/work storage to cold archive.                                                                                       | Factory one-shot.                                               |
| `merge`    | Insert per-channel store rows into the language master store.                                                                                            | Manual/post-factory.                                            |
| `verify`   | Scribe-score master-store rows for quality gates.                                                                                                        | Manual/post-factory; shares the Scribe API budget when running. |
| `export`   | Materialize a selected training dataset.                                                                                                                 | Manual/release.                                                 |
| `publish`  | Upload/release artifacts and verify remote state.                                                                                                        | Manual/release.                                                 |

## Source Metadata

YouTube source metadata is carried through the pipeline instead of being stranded in the source registry:

1. `Channel` entries carry `tier` and `category`.
1. `prescan` records channel reachability and intended lane in `data/prescan.sqlite`.
1. `download` writes yt-dlp `*.info.json` sidecars.
1. `enqueue` stores bounded video metadata (`title`, `description`, `upload_date`, channel URL, duration, tags/categories) on `queue.sqlite.videos`.
1. Existing queue rows can be refreshed from registry changes with `<lang>-curate repair-metadata`.
1. `segment` copies `tier`, `category`, and metadata into clip rows.
1. `harvest` writes those fields into `Sample.meta`.
1. `export` writes row-level `metadata` JSON next to the training columns and license fields.
1. `export --youtube-stratified-splits` applies the 17-genre taxonomy and deterministic, category-stratified, whole-video dev/test assignment. Existing exports are unchanged unless the flag is passed.

## Verified Storage State

The workers SSD pressure is Russian canonical audio, not SQLite.

| Path                                                                   |                       Approx size | Verified origin / writer                                                                           |
| ---------------------------------------------------------------------- | --------------------------------: | -------------------------------------------------------------------------------------------------- |
| `/mnt/tiny-2t/peacock-asr/{dari,farsi,georgian,tajik}-asr/data/create` |                     `~716G` total | Active source-audio landing zones for project `data/create` symlinks.                              |
| `/mnt/workerssd-2t/peacock-asr/russian-asr/canonical_audio`            |                           `~1.2T` | Russian ingest working root.                                                                       |
| `/mnt/workerssd-2t/peacock-asr/russian-asr/curator.sqlite`             |                           `~8.4G` | Russian master SQLite store. Not the main space consumer.                                          |
| Project `data/clips`                                                   | bounded project-local clip output | Default clip root used by `segment`; an explicit `--clips-root` may redirect new clips to scratch. |

Archive state:

- Canonical root: `/mnt/massive-22t/peacock-asr-archive`.
- Current top-level buckets: `dari`, `farsi`, `georgian`, `russian`, `tajik`.
- Removed stale language bucket: `/mnt/massive-22t/peacock-asr-archive/persian`.
- Former `persian/iran_international` material was non-destructively merged into canonical `farsi/iran_international`; `16` same-name conflicts were resolved by promoting the manifest-backed legacy variants and preserving previous canonical variants under `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_conflicts_2026-06-30/current_variants_before_legacy_promotion`. The legacy folder is retained temporarily until an explicit deletion checkpoint.
- Previous pre-renaming archive root is gone.
- On 2026-06-25, 678 completed files (`17.6G`) were migrated from the removed root into the canonical root.
- Four stale `.partial` files in the removed root were deleted only after their original source FLACs were verified under the then-current project working roots.
- Empty old manifests and empty directories were removed.

Storage rules:

1. `/mnt/tiny-2t` owns active project `data` symlinks for Dari, Farsi, Georgian, Tajik, and CV26. Keep at least `250G` free. Bounded project-local `data/clips` is allowed, but segment launch must stop at the configured pending-clip and free-space limits.
1. Source audio lands in the owning project's `data/create` tree by default. Do not create top-level source-cache roots; use `--create-root` only for an explicit, temporary scratch relocation.
1. Project `data/clips` is the default clip cache. Use `/mnt/workerssd-2t` as a scratch clip root only by explicit operator choice, with a soft cap such as `450G`.
1. `/mnt/workerssd-2t/peacock-asr/russian-asr/canonical_audio` is frozen working debt, not general scratch. It must be published or moved before the workers SSD is treated as a clean factory disk.
1. `/mnt/massive-22t/peacock-asr-archive` is the cold source-audio archive and release staging area. Keep at least `2T` free for large upload/download retries.
1. `/mnt/media-5t` is not an ASR write target.
1. Local deletion requires verified copy or verified HF publication.
1. Do not create new top-level storage namespaces without adding them here and to the config surface.

## P0 - Atomic Correctness

Factory production readiness requires all of this work to be true at the same time:

1. VAD never emits a clip above the selected segmentation profile's maximum duration.
1. Segment parent death terminates child workers and releases GPU memory; verify with `nvidia-smi`.
1. Stale `segmenting` videos are reclaimed before fresh pending work.
1. Segment output publication is ownership-safe: workers cut into claim-token staging dirs and publish final clip files only inside token-guarded `complete_video()`.
1. `resegment` refuses active stage locks and refuses existing channel stores unless explicitly overridden.
1. Archive moves use the canonical root only and do not trust same-size destination files as proof of identical content.
1. Segment launch gating is based on real resource predicates: pending clip HWM, free GB, active locks, worker lifecycle, and output ownership. A permanent global cap is not used.
1. Scribe API calls are controlled by one global budget shared by `labelq` and any live `verify`.
1. Operational logs stay outside the repo root.

Exit criteria:

- focused tests cover VAD duration, orphan cleanup, stale claim priority, resegment guards, archive safety, factory backpressure, and Scribe budget splitting;
- a killed segment parent leaves no orphaned GPU workers;
- a dry run explains every launch/hold decision;
- `uv run --project packages/omni-curator pytest ...`, Ruff, and `git diff --check` pass on touched files.

## P1 - Factory Configuration

Implemented:

1. `omni_curator.factory --config <path>` reads one flat TOML config for roots, projects, stages, tick interval, segment worker counts, pending HWM, minimum free GB, archive root, and Scribe budget.
1. CLI flags override config values.
1. `--once --dry-run` remains the safe operator preview and does not write Scribe window files.
1. Dry-run logs `RUN`, `HOLD`, and `BALANCE` decisions with resource blockers.
1. The factory prints to stdout by default; it appends to a file only when `--log-file` or `log_file` config is set. No repo-root `factory.log` is created by default.

Still open:

1. Make `archive_needed()` cheaper than scanning every segmented source path every tick.

Exit criteria:

- factory defaults cannot fill the SSD;
- factory config matches documented roots;
- resource blockers are visible without reading the code.

## P2 - Storage Lifecycle

1. Treat `/mnt/workerssd-2t` as hot storage, not archive.
1. Keep active source and clip caches bounded.
1. Move cold source audio to `/mnt/massive-22t/peacock-asr-archive` after segmentation.
1. Publish or move Russian canonical audio before growing new hot corpora.
1. Avoid dumping millions of tiny clip files to exFAT cold storage unless sharded or intentionally accepted.

Exit criteria:

- per-tier usage is visible without manual detective work;
- local deletion requires verified copy or verified HF publication;
- no process silently creates a new top-level storage namespace.

## Quality Gates

WER here means the store-level Scribe verification score: the stored label scored against a fresh Scribe transcription of the same clip. It is not a model eval WER.

Recording-type tiers:

| Recording type                                                              | Excellent |     Good | Acceptable |
| --------------------------------------------------------------------------- | --------: | -------: | ---------: |
| Read / broadcast: scripted Common Voice, FLEURS, audiobooks                 |   `<= 5%` | `<= 15%` |   `<= 25%` |
| Conversational / spontaneous: interviews, calls, drill audio, YouTube shows |  `<= 15%` | `<= 35%` |   `<= 60%` |

Fallback by language resource level when recording type is not enough: high-resource `<= 20%`, medium-resource `<= 30%`, low-resource `<= 50%`.

Duration:

1. Never export above `OMNI_MAX_DURATION_S = 40s`; Omni ASR truncates input audio at 40 seconds.
1. Emitted clip caps are model/profile-specific: current Parakeet CTC recipes use 20 seconds, TDT uses 30 seconds, and Omni evaluation accepts 40 seconds. Preserve raw intervals so a curation run is not falsely described as having a universal 30-second VAD limit.
1. Keep NeMo's `0.3s` minimum as a cheap artifact filter when the corpus can tolerate it.
1. The lower bound is not model-constant; CTC only requires enough encoder frames for the label.

Per-second text rates are a physical-plausibility backstop, not the main gate. Compute `chars_per_second` and `words_per_second` on normalized labels to catch audio/text mismatch where the transcript is too dense or too sparse for the clip duration. NeMo defines the metrics but does not publish universal caps, so set language caps from a physiology anchor (`~8 syllables/sec * script chars/syllable`) or just above the corpus `p99.9`.

Code constants live in `packages/omni-curator/src/omni_curator/audit/quality.py`. Primary source docs are NVIDIA NeMo Curator WER filtering and audio quality metrics; speech-rate rationale comes from the cross-language `~39 bits/sec` result and speech-tempo literature.

Planned independent transcript/audio audit:

1. Add a beginning/end ASR-diff signal equivalent to NeMo Speech Data Processor's `DropASRErrorBeginningEnd`, using a draft model independent from the label teacher.
1. Add CTC-segmentation confidence and token/character timings with a different model family.
1. Treat unsupported characters and normalization/tokenizer failures as review, not bad audio.
1. Store all scores and reasons in an audit sidecar first. Do not make the first implementation an automatic destructive filter.
1. Calibrate a high-precision reject lane on a human-reviewed sample, then prove value with a size-matched cleaned-versus-random training ablation. Never filter dev/test by model confidence.

Primary implementations: NVIDIA NeMo Speech Data Processor [`DropASRErrorBeginningEnd`](https://nvidia.github.io/NeMo-speech-data-processor/_modules/sdp/processors/modify_manifest/data_to_dropbool.html) and [`ctc-segmentation`](https://github.com/lumaku/ctc-segmentation). These are model-conditioned quality signals, not proof that a transcript is true.

## P3 - Multi-VAD Segmentation

Correctness comes before speed.

Current state (2026-07-09): steps 1-5 and the same-audio 32-source Farsi interval pilot in step 7 are complete. The 160-item blinded review selected Silero 102-58 overall, 53-27 on clean/read, and 49-31 on noisy/broadcast audio. Farsi now defaults to the exact reviewed Silero ONNX model, threshold 0.5, `conservative-v1`, and CPU workers. The 320-clip source-balanced local-ASR gate completed with 640 Omni/C1Tech predictions, zero runtime errors, 100% Persian-script letter rate, and only three model-specific Omni empties on 0.316-0.668-second Silero fragments. VAD selection is promoted; downstream bounded carry-through remains open. A durable run table and live orphan-process validation also remain open.

Order:

1. define the typed engine and postprocessing profile contracts;
1. port Cobra, Silero, and the exact benchmark MarbleNet revision behind that contract;
1. decode each source once and share the 16 kHz mono array between VAD and cutting;
1. persist deterministic engine/profile provenance and enforce the selected maximum duration;
1. add bounded channel/video/manifest selectors plus isolated pilot output;
1. make worker/device/secret preflight engine-aware and verify child-process cleanup live;
1. run the same Farsi news and audiobook audio through all engines, then carry the selected profile through `labelq`, `harvest`, and `verify` before production scale;
1. batch/bucket VAD inputs only after correctness and output equivalence are established.

Exit criteria:

- every adapter passes the same interval, hard-split, provenance, and failure-contract tests;
- the bounded Farsi pilot records coverage, duration distribution, empty/error rate, ASR yield, output bytes, real curator throughput, and sampled boundary judgments;
- no engine or speed change reintroduces over-length clips, queue overwrite, or orphaned GPU workers.

## P4 - Hugging Face Release Lifecycle

1. Build release artifacts into an explicit staging root with a free-space preflight.
1. Keep one append-only release state log: `converted`, `uploaded`, `verified`, `complete`.
1. Use one chosen upload method per release: `hf upload-large-folder`, per-shard commits, or explicit Xet/LFS workflow.
1. Verify remote siblings and re-download/checksum representative artifacts.
1. Record shipped names, verification, and deletion decisions in `CHANGELOG.md`.
1. Delete or move local hot artifacts only after verification.

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
