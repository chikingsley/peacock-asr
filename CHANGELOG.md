# CHANGELOG — peacock-asr

Historical record of completed work, terse. Active work is in `TODO.md`; generated pipeline state
comes from `packages/omni-curator/status.py`; root `STATUS.md` is ignored and not canonical.

## MOSS MLX conversion

- Created `projects/moss-mlx-conversion`: pinned MOSS Transcribe 2B, captured PyTorch BF16
  reference/parity artifacts, converted all 838 BF16 tensors to MLX layout, verified Apple Silicon
  smoke parity, and added streamed LibriSpeech clean-test eval (20 rows: 1.58% WER, 0.65 RTF).

## Tajik

- Reproduced and extended the 110M Parakeet TDT result on 2026-07-09 with the promoted final
  `.nemo` loaded intact: FLEURS test `19.03%` WER / `6.72%` CER (600), Common Voice dev `17.77%` /
  `5.89%` (357), and the restored video-disjoint conversational test `33.85%` / `14.89%` (1,625;
  two empty outputs). The <=30-second slice is `34.38%` / `15.10%` (1,559), so 30 seconds is a
  training/operational profile rather than an accuracy-improving universal cap.
- Live same-audio comparison confirmed Omni CTC 300M v3 at `37.65%` WER / `14.04%` CER. The 110M
  TDT wins conversational WER by `3.80` absolute (`10.1%` relative) but loses CER by `0.85`, so the
  next step is per-source/error analysis rather than immediate retraining. TDT warm-cache throughput
  reached `684x` on FLEURS and `876x` on conversational audio; the first cold FLEURS pass was `216x`,
  proving cold I/O and warm model throughput must be reported separately.
- Restored the exact conversational test from `Peacockery/tajik-asr-corpus-v3` revision
  `3b05a4bb89104c21643081250729595347d1188e`: 18 Parquet files (`702,784,427` bytes), 1,625 rows,
  9.926 hours, all FLACs valid, no rows above 40 seconds. Added a repeatable
  `tajik-parakeet-materialize-eval` CLI with deterministic audio hashes, manifest provenance, resume
  checks, summary output, and tests.
- Restored `Peacockery/omni-ctc-300m-tajik` revision
  `cafa6e9fb394f7cef29caf79385feb96bcfc05ae` after its asset card pointed at a retired training-run
  directory. `model.pt` is `1,304,101,361` bytes with SHA256
  `18b3ef847ec56b7ea3c3a9fcddc4cf38b94392880c79cf35710b4ca23b01d6bb`; the live asset card now uses
  the HF-restored project data path.
- v2 trained + eval: best step_19500, FLEURS test WER 17.17 (base 19.74, v0 17.34). Recorded in EXPERIMENTS.md.
- Conversational test set + v3 — the data lever proven. Held-out 157 whole videos (frozen manifest, leakage-safe carve). Conversational held-out (1,625 clips): v0 49.89 → v3 37.65 WER (−12.2 pts / −24.5% rel from 1,070h); v3 (37.65) ≈ v2-contaminated (37.40) so it's real generalization. Shipping model `omni_ctc_300m_v2_tajik_v3_step_20000`. KenLM fusion proven (−16% rel, α=0.5/β=0).
- Export v2: WER ≤ 0.35 + descriptor-junk filter + language gate; 0 unk.
- Gate-fix recovery (`6259a355`): audited the 38,133 drops (~95% genuinely non-Tajik); added function-word vocabulary tiebreak to `keep_for_language`. +1,563 Tajik recovered, 0 regressions, 0 Russian re-admitted. Applies to v4.
- Re-ingested older HF datasets (`887854fe`): `hf-muhtasham` (1,300 rows, forced train) + `hf-commonvoice22` (282 gold rows, splits preserved); all 1,582 Scribe-scored, 0 failures. (CV25_tg never existed — Mozilla stopped at v17; Tajik comes from `fsicoli/common_voice_22_0`.)
- Language-learning channels wired (`887854fe`): `learning_tajik_achilovs` + three "with Chris" channels.
- Stale/done backlog items closed: export→train/eval (done by v3 cards + template); tajik `train.py` converged to preset; zero-span videos non-issue (queue.sqlite PK de-dups); 269 unscored rows are non-speech markers verify skips.
- Tidy: deleted 43 MB `ws_1.c4af3cd1` stub + empty root `src/`. Gotcha: editing a training config changes fairseq2's `ws_<hash>` (re-launch starts fresh; hand-move checkpoints to resume).

## Verify / scoring

- Parakeet evaluation safety: promoted `.nemo` models no longer run `change_vocabulary()` during
  evaluation; base-plus-checkpoint reconstruction requires explicit `--replace-tokenizer` and an
  exact state-dict match. The evaluator now records raw/normalized WER and CER, empty outputs,
  timed RTFx, warm-up count, peak CUDA allocation, predictions, and a JSON summary.
- Parakeet training safety: the generic NeMo wrapper is CTC-only and refuses TDT/RNNT models;
  dedicated TDT runs enforce the loss repair, eval loss, and `val_loss` selection. Training now saves
  distinct last-step `_final.nemo` and best-validation `_best-valloss.nemo` artifacts.
- Omni evaluation accepts materialized JSONL manifests, avoiding the four-minute / multi-GB Python
  expansion of embedded Parquet audio lists observed on the Tajik conversational benchmark.
- Script-aware verify scoring (Sonnet transliteration won the bake-off; hypothesis-only prompt) + `rescore` CLI.
- Full verify + 150k-row rescore — every scoreable row has an honest score.
- Dev split: FLEURS dev/test are the benchmarks (gates train-only by design — `Selection.gated_splits`).
- Codex review of breaker/renewal work — all 7 findings fixed (`c5b4cedf`).

## HuggingFace

- Policy set 2026-06-10: HF (Peacockery org, public) is the archive, local is working state. Shipped versions get plain-prose cards; superseded versions recorded + deleted locally. Naming: models `<family>-<size>-<language>`, datasets `<language>-asr-corpus-vN` + `<language>-asr-<scope>`.
- Fleurs mirror dupes deleted from HF (`fleurs-parquet`, `google-fleurs`).
- `hf-upload` skill bans the broken batch pattern; per-shard committer used for large sets.

## Templates / architecture (one pipeline, one structure)

- Deleted the fused path + chunks/align (`a4d2f067`): ~850 lines removed (`create/run.py`, `pipeline.py`, `align.py`, `fuse/stitch.py`, `fuse/polish.py`, `segmenters/chunks.py`, package `cli.py` + `omni-curator` entry point, tajik `cmd_label`); `cut_audio` moved to `create/audio.py`.
- `create/` reorg (`6c70295b`): flat, one module per stage in pipeline order (`youtube → queue → vad → segment → transcribe → fuse → labelq`).
- Curate-side language template (`8823c481`, Codex xhigh reviewed): 12-command CLI in `omni_curator/project.py`, parameterized by frozen `CuratorProject`; tajik + georgian `curate.py` ~45 lines of config. Ingest sources are a registry (`IngestFn`); coverage gate injected; fail-fast validation. Recipe `docs/NEW_LANGUAGE.md`.
- Model-side language template (`18248691`): `omni_finetune_core/project.py` owns train + eval via `FinetuneProject`; pinned typed `TrainingPreset`s (tajik v3 field-equivalent to YAML, `configs/` deleted) + georgian `--regime` path; `fragment_cache_dir` in typed config; eval ported with injected normalizer; 7 core tests.
- Georgian model side on the template (`georgian-train --regime gpu_max`); 145.3 h v0 export existed.
- persian-asr migration phase 1 (`38b9779f`): `src/persian_asr/` template package — `persian-curate`, `persian-train-v2` (scribe-v4-rewarm pinned, 0 field diffs), `persian-eval-v2`; production checkpoint registered (step_7000, dev WER 11.15). Existing cards imported, zero removals.
- Killed `Any` types (ProcessFn/Mapping/SuperwhisperClient throughout).

## Curator package (this session)

- Audited and removed abandoned pre-reset clip output: 4,227 FLAC/temp files (~2.8 GB) plus all
  empty descendants from Dari, Farsi, Georgian, and Tajik `data/clips`. Current queues, active and
  premigration stores, manifests, and training artifacts had zero exact references; all 128 source
  recordings remained locally or archive-resolvable. The four clip root directories were preserved.
- Package reorg (`audit/` `data/` `scribe/`); dead-code/shim inventory (codex: zero stale refs).
- Live Scribe concurrency control + cross-job balancer (`scribe/concurrency.py`, `scribe/balance.py`).
- GPU/hybrid VAD + codex fixes (`vad.py` NVML `resolve_devices`; `segment.py` thread caps + worker-exit checks).
- Dev tooling: ruff, ty, vulture, ai-slop-detector, dslop.
- Verify hardened for the SuperWhisper async-API migration (rides silent-clip 200s).
- Canonical curation/factory plan added at `packages/omni-curator/docs/CURATION_FACTORY.md`.
- Curation/factory docs realigned: `CURATION_FACTORY.md` is the plan, `TODO.md` is active backlog,
  and `CHANGELOG.md` is historical record.
- Archive root consolidated on `/mnt/storage/peacock-asr-archive`: migrated 678 completed files
  (`17.6G`) from `/mnt/storage/peacock-archive`, verified four stale partials still had source
  FLACs under `/mnt/overflow/peacock-asr`, removed those partials, and removed the old root.
- Removed the accidental root `factory.log` relocation and deleted `/home/simon/logs/peacock-asr/`.
- VAD hard-cap edge fixed: `split_window()` no longer has a tail-fold path that can exceed the
  configured max duration.
- `resegment` hardened to refuse active stage locks and existing per-channel stores unless the
  operator explicitly overrides.
- Deleted superseded curation docs: `packages/omni-curator/docs/factory_plan.md`,
  `packages/omni-curator/docs/segment_throughput_plan.md`, `packages/omni-curator/CURATING.md`,
  and `packages/omni-curator/docs/archive/PIPELINE_SPLIT.md`.
- Removed deprecated curator/model shims: `segment --procs`, `ScribeError.generation`, and bare
  `<lang>-train` / `<lang>-eval` script aliases from Tajik, Farsi, Dari, Georgian, and Russian
  project pyprojects; current examples use explicit `*-omni-*` commands.
- Collapsed `packages/omni-curator/QUALITY.md` into the canonical curation plan and updated
  `docs/ASR_PROJECT_STANDARD.md` plus `docs/NEW_LANGUAGE.md` to the explicit `*-omni-*` command
  surface.
- Georgian source registry expanded from the 2026-06-13 research pass: wired the non-duplicate
  channel entries into `projects/georgian-asr/src/georgian_asr/sources.py`, kept `@interpressnews`
  out after `yt-dlp` returned 404, and noted the duplicate audiobook channel.
- HF state verified on 2026-06-25: `tajik-asr-youtube`, `tajik-asr-corpus-v3`,
  `georgian-asr-corpus-v0`, `farsi-asr-corpus-v4`, `omni-ctc-300m-tajik`,
  `omni-ctc-300m-farsi`, and `parakeet-ctc-109m-farsi` are public/ungated via the HF API.
- Mechanical carry-overs closed: HF/FLEURS ingest defaults to non-streaming with an explicit
  `streaming=` knob; `omni-bench-llm` and Farsi KenLM eval honor `--limit` without full-shard
  parquet reads; `run_recipe()` restores `sys.argv` after `runpy`.
- Factory P0/P1 implementation closed: removed the hard global segment cap, added per-project
  resource blockers (pending clip HWM, min free GB, stage locks), claim-token staged clip
  publication, flat TOML factory config with CLI overrides, and dry-run `RUN`/`HOLD`/`BALANCE`
  status output. The factory CLI now logs to stdout only unless `--log-file` or config sets a path,
  so it does not create repo-root `factory.log` by default.
- Provenance/export contracts tightened: curator export now preserves row-level `metadata` JSON
  after the training columns/license fields; source provenance and license policy have typed models;
  language-gate bypass can be granted by trusted source or trusted provenance authority. Fine-tune
  policy now has Pydantic contracts for recipe/head/metric/eval decisions, including CTC-vs-
  transducer checkpoint metric validation and the Omni 40s eval cap.
- YouTube source preflight/metadata added: `prescan` writes project-local `data/prescan.sqlite`,
  channel registries carry `category`, downloads write yt-dlp info sidecars, and enqueue/segment/
  harvest/export preserve `tier`, `category`, and bounded video metadata. Export now has an
  opt-in `--youtube-stratified-splits` policy with a 17-genre taxonomy and deterministic,
  category-stratified, whole-video dev/test assignment.
- Queue metadata repair on 2026-06-30: added `<lang>-curate repair-metadata`, inferred registry
  categories when source entries omit them, refreshed active Dari/Farsi/Georgian/Tajik queues, and
  backfilled `meta.webpage_url`, `meta.channel_url`, `meta.tier`, and `meta.category` on every
  active row. Russian was skipped because it has no active split-pipeline YouTube queue.
- Storage tiering policy made explicit on the live mounts: `/mnt/tiny-2t` owns active
  non-Russian project `data` symlinks and source caches, `/mnt/workerssd-2t` currently holds
  Russian working audio plus any explicitly chosen scratch roots, `/mnt/massive-22t` is the cold
  archive/release staging root, and non-ASR media mounts are excluded from ASR writes.
- Root cleanup on 2026-06-30: removed stale `.prettierrc.json`, root `.python-version`,
  root `STATUS.md`, retired `docs/hf-cards/uploader.py`, and historical research imports after
  preserving the history in Git; aligned the live handoff docs around `TODO.md`,
  `CHANGELOG.md`, `docs/data-state-audit-2026-06-28.md`, and `CURATION_FACTORY.md`.
- Farsi archive merge on 2026-06-30: copied `267` legacy-only FLACs plus
  `archive_manifest.jsonl` from `farsi/iran_international_legacy` into canonical
  `farsi/iran_international`; promoted the `16` manifest-backed legacy variants for same-name
  conflicts, preserved the previous canonical variants under
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_conflicts_2026-06-30/current_variants_before_legacy_promotion`,
  and verified all `430` common canonical/legacy FLAC names now match by decoded-audio MD5.
- Farsi source inventory on 2026-06-30: enqueued the `444` completed local
  `iran_international` FLACs that were present in `downloaded.txt` but absent from `queue.sqlite`;
  renamed `@AvasBookClub` to registry slug `avas_book_club`, enqueued its `220` clean audiobook
  FLACs, and repaired metadata for those rows. The currently downloaded Farsi local source set now
  has `0` local unqueued FLACs.
- Farsi Tiny2T source backfill on 2026-06-30: copied the `428` queue-recorded
  `iran_international` project-create gaps (`8.92G`) from the canonical Massive archive back to
  `/mnt/tiny-2t/peacock-asr/farsi-asr/data/create/iran_international`; post-copy check returned
  `missing_after 0`.
- Tajik missing-source recovery on 2026-06-30: materialized the `31` queue rows missing from both
  local create paths and Massive archive fallback, re-downloaded them from their
  `meta.webpage_url` values into the recorded `data/create` paths, and verified all `31` expected
  FLACs now exist. Remaining Tajik local gaps are archive-backed.
- Georgian research cleanup: removed the unresolved `@interpressnews` paste-block/risk entry after
  live `yt-dlp` returned 404; it remains unwired.
- Live checks on 2026-06-25: factory dry-run against Dari segment held correctly on real blockers
  (`no pending/stale videos`; `426327` pending clips >= HWM `50000`), `nvidia-smi` showed no GPU
  worker processes, no repo-root `factory.log`/`/home/simon/logs/peacock-asr`/old archive root
  existed, and the canonical archive root had no `.partial` files.
- Verification for this cleanup: `uv run --project packages/omni-curator --locked pytest
  packages/omni-curator/tests` (`203 passed`), finetune-core tests (`14 passed`), curator Ruff,
  finetune-core Ruff, Farsi smoke
  tests/Ruff, touched language-project Ruff checks, live factory dry-run, `nvidia-smi`, storage
  snapshot, archive-partial scan, and `git diff --check`.

## Factory prereqs (completed before current plan)

- P1 — video claim-tokens + lease guard (`queue.py`).
- P2 — `merge` preserves `scribe_wer`/`meta` (`store.insert_if_absent`).
- P3 — verify unscoreable sentinel (`scribe_status`).
- P4 — abortable download (`--disk-guard`, mid-channel).
- Source-audio archiver: `create/archive.py` + `cmd_archive` (built, tested, running).

## Status snapshots (history)

- 2026-06-13: Tajik v3 ships (FLEURS 16.9 / conversational 37.6), v4 scale run wired (80 channels) + downloading. Farsi (ex-persian) atomic rename done; production `omni_ctc_300m_v2_farsi_v4_step_41000`, parked. Dari project scaffolded (`50817454`, `fas_Arab`, 27 channels, Farsi warm-start card). Georgian v0 trained (pooled 20.7 WER) + KenLM (24.7→18.9 FLEURS). GPU down — hardware bus-drop under load (Xid 79) ×2; needed host reboot + power cap (~175W) + temp logging before any training/eval.
