# TODO — peacock-asr

Active work is unchecked. Checked items below were closed or verified in the 2026-06-25/2026-06-30
cleanup passes and are kept briefly for review; `CHANGELOG.md` is the historical record.

## Curator pipeline

- [ ] Pre-resegmentation source-path policy: Farsi downloaded local source and the Tajik
  missing-from-both rows are reconciled. Before production resegmentation, decide per
  language/channel whether archive-backed non-Farsi gaps are restored to Tiny2T in bounded batches
  or served from Massive fallback for pilots; keep `docs/data-state-audit-2026-06-28.md` current as
  the handoff before replacing the segmenter.
- [ ] Farsi resegment pilot prep: current downloaded local source audio is reconciled in the queue
  (`iran_international=12,005`, `avas_book_club=220`); run a small Farsi pilot with the project
  `data/clips` default unless an operator explicitly chooses a scratch clip root.
- [ ] Farsi broad-registry downloads: the registry has `124` YouTube channels; only
  `iran_international` and `avas_book_club` currently have local FLACs, so download/enqueue the
  remaining `122` registry channels before calling the broad Farsi YouTube corpus complete.
  Current audit manifests and runnable commands are under `projects/farsi-asr/data/audit/`.
- [x] Farsi current source reconciliation: merged/promoted the legacy `iran_international` archive
  variants, restored the `428` Tiny2T source gaps from Massive, enqueued the `444` local
  `iran_international` files plus `220` `avas_book_club` files, and verified local Farsi FLACs
  outside the queue are `0`.
- [x] Tajik missing-from-both recovery: re-downloaded the former `31` rows absent from both
  project-create and Massive; remaining Tajik local source gaps are archive-backed.
- [x] Queue metadata/category refresh: repaired active Dari/Farsi/Georgian/Tajik YouTube queues
  from the channel registries; every active row now has non-uncategorized `category`,
  `meta.webpage_url`, `meta.tier`, and `meta.category`. Russian has no active split-pipeline
  YouTube queue to repair.
- [ ] Tajik v4 scale run: finish download → enqueue → segment → labelq → harvest → merge → verify → export v4 (`--max-wer 0.35`) → train (3–5 epochs / best-WER ckpt) → eval all models on v4 test + KenLM α=0.5/β=0. Dedup `tv_tajikistan`/`tvt_tojikiston`.
- [ ] Dari v0 remaining work: refresh cookies, download remaining wired channels, stage Pimsleur Dari audio into the create root, run enqueue → segment → labelq → harvest → merge → verify → export v0, then train cold-base vs Farsi-warm (`--regime warm_restart --lr 2e-6`). Optional: Pashto language gate for bilingual channels.
- [x] Georgian v1 source registry: wired the non-duplicate 2026-06-13 researched channels from `projects/georgian-asr/docs/georgian_youtube_channels.md` into `sources.py`; left `@interpressnews` out because `yt-dlp` still returns 404, and skipped the duplicate audiobook channel already present as `audiobooks_geo_ka`.
- [ ] Georgian v1 run work: download/enqueue the expanded channel registry, segment → labelq → harvest → merge → verify → export v1, then run KenLM-fused eval.
- [x] Georgian source cleanup: deleted the unresolved `@interpressnews` paste-block/risk note from
  the research doc after live `yt-dlp` returned 404.
- [x] YouTube channel prescan DB: added `prescan`, resolved channel handles through the registry,
  records lane/status/count/error in project-local `data/prescan.sqlite`, and exits non-zero on
  failed channel checks.
- [x] YouTube metadata model: added channel `category`, persisted `tier`/`category` plus bounded
  yt-dlp title/description/upload metadata through queue videos, clips, harvested store rows, and
  export row `metadata` JSON.
- [x] YouTube split policy: implemented the 17-genre taxonomy and opt-in category-stratified,
  video-disjoint dev/test assignment for exports.

## Factory

- [ ] P0 factory production readiness remaining work: rerun live fault tests for child cleanup/orphan
  GPU workers under a real segment worker.
- [x] Replace the hardcoded global segment cap with real scheduling predicates: pending clip HWM,
  min free GB, active locks, worker lifecycle, and clip-file ownership.
- [x] Segment output ownership: segment workers cut into claim-token staging dirs and publish files
  only through token-guarded `complete_video()`.
- [x] Scribe API status visibility: the balancer splits one global budget across live `labelq` and
  `verify` jobs, and dry-run reports the assignment without writing window files.
- [x] Factory config surface: roots, workers, HWM, min free GB, archive root, and Scribe budget can
  come from one flat TOML config with CLI overrides.
- [x] Documentation cleanup: `CURATION_FACTORY.md` is the plan, `CHANGELOG.md` is history, `TODO.md` is backlog; deleted superseded factory/throughput/curating/split docs, collapsed `QUALITY.md` into the canonical plan, and updated `NEW_LANGUAGE.md`.

## Data lifecycle / HuggingFace

- [ ] Export final YouTube datasets → push/append to HF → delete local clips (the only durable overflow relief; segmenting nets ~0).
- [ ] Replace the retired ad hoc HF uploader with a shared release workflow in the maintained
  packages: explicit staging root, append-only release state, one chosen upload method, remote
  sibling/checksum verification, CHANGELOG record, and local deletion only after proof.
- [x] `tajik-asr-youtube` community dataset upload verified on HF API: public, ungated, and has siblings.
- [x] Former in-flight uploads verified on HF API: `tajik-asr-corpus-v3`, `georgian-asr-corpus-v0`, `farsi-asr-corpus-v4`, `omni-ctc-300m-tajik`, `omni-ctc-300m-farsi`, and `parakeet-ctc-109m-farsi` are public/ungated.
- [ ] Remote/destructive cleanup: verify exact current repo names, then delete superseded Persian/Farsi model repos (`omni-ctc-300m-v2-fleurs-fa-ir`, `...-thomcles-continue` if still present); decide local deletion of Farsi scribe-v4 60G + Tajik v3 71G parquets only after HF copy is canonical.

## Storage

- [x] Storage tiering policy: current snapshot is `/mnt/tiny-2t` 52%, `/mnt/workerssd-2t` 64%,
  `/mnt/massive-22t` 54%, and `/` 41% in the 2026-06-28 audit; workers SSD has Russian
  canonical audio 1.2T and Russian SQLite 8.4G. New Farsi/Dari/Georgian/Tajik source audio stays
  on Tiny2T by default; any WorkersSSD clip scratch root requires an explicit operator choice.
  `CURATION_FACTORY.md` now defines tier ownership, minimum free-space floors, and source audio
  defaults to each project's `data/create` unless an operator explicitly overrides it.
- [x] Transparent re-segment-from-archive: segment resolves missing create-root sources under `/mnt/massive-22t/peacock-asr-archive`; covered by `packages/omni-curator/tests/test_segment.py`.
- [ ] Russian SSD pressure: move or publish `/mnt/workerssd-2t/peacock-asr/russian-asr/canonical_audio` before treating the workers SSD as general factory scratch.

## Cleanup / typing

- [x] Delete deprecated shims: removed bare `<lang>-train`/`-eval` aliases from project pyprojects, removed `segment --procs`, removed `ScribeError.generation`, and updated tests/docs to the explicit `*-omni-*` surface.
- [x] HF ingest shutdown guard: default HF/FLEURS ingest to non-streaming and keep an explicit `streaming=` knob for sources that need it.
- [x] Eval limit loading: main omni eval, `omni-bench-llm`, and Farsi KenLM eval use batch/early-stop loading instead of reading full parquet shards for `--limit`.
- [x] Recipe wrapper hygiene: `omni_finetune_core.train.run_recipe()` restores `sys.argv` after `runpy.run_module`.
- [x] Parquet schema duplication: added a Farsi smoke compatibility test proving curator export's
  training columns match the fine-tune reader schema.
- [x] Source-provenance typed models: added origin/authority/tool, transform history, and typed
  license gate objects with round-trip tests.
- [x] Fine-tune-policy typed models: recipe/head/metric/eval contracts as Pydantic + validation tests.
- [x] Language-gate trusted-source mode: trusted sources or provenance authorities can bypass the
  heuristic language gate where justified.

## Training

- [ ] LLM ceiling bench (`omni-bench-llm`) on Tajik/Georgian/Farsi — decides whether LLM-in-the-loop (GER/NAR) work is worth it. Weights pre-cached.
- [ ] Farsi (parked): re-upload corrected card; dedup Thomcles vs `asr_farsi_youtube` on the next export.

## Someday (blocked on missing inputs)

- [ ] Native-speaker ground-truth check — the only true conversational-accuracy measure vs the Scribe-label ceiling. Not actionable until a Tajik speaker is available.

## Reference

- Current curation/factory plan: `packages/omni-curator/docs/CURATION_FACTORY.md`.
- Live per-language pipeline state is printed on demand by `packages/omni-curator/status.py`;
  root `STATUS.md` is intentionally ignored and not a source of truth.
