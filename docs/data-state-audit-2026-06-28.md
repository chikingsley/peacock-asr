# Peacock ASR Data State Audit - 2026-06-28

Status: live-state audit after the root `peacock-create` cleanup and the archive `persian` bucket
cleanup. This document is a handoff for the pause-and-stabilize phase before resegmentation.

Updated 2026-06-30 after queue metadata repair, Farsi source reconciliation, and Tajik
missing-from-both recovery.

## Current Storage Layout

Canonical cold archive:

- `/mnt/massive-22t/peacock-asr-archive`
- Top-level buckets now present: `dari`, `farsi`, `georgian`, `russian`, `tajik`
- The stale top-level `persian` bucket was removed.
- Former `persian/iran_international` material is temporarily retained under:
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy`
- Its manifest is retained at:
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy/archive_manifest.jsonl`

Working mirror/project roots:

- `/mnt/tiny-2t/peacock-asr/dari-asr` - `188G`
- `/mnt/tiny-2t/peacock-asr/farsi-asr` - `277G`
- `/mnt/tiny-2t/peacock-asr/georgian-asr` - `105G`
- `/mnt/tiny-2t/peacock-asr/tajik-asr` - `169G`
- `/mnt/workerssd-2t/peacock-asr/russian-asr` - Russian working/canonical audio only

Current filesystem free space:

| Mount | Used | Free | Use |
|---|---:|---:|---:|
| `/` | `696G` | `1.1T` | `41%` |
| `/mnt/tiny-2t` | `889G` | `851G` | `52%` |
| `/mnt/workerssd-2t` | `1.2T` | `663G` | `64%` |
| `/mnt/massive-22t` | `11T` | `9.3T` | `54%` |

## Project Source Caches

Project-local `data/create` is the active source-audio landing area.

| Project | `data/create` size | FLAC files | Non-FLAC files |
|---|---:|---:|---:|
| `dari-asr` | `186G` | `3,880` | `1,521` |
| `farsi-asr` | `263G` | `12,213` | `1,534` |
| `georgian-asr` | `104G` | `11,166` | `2,022` |
| `tajik-asr` | `164G` | `7,570` | `1,480` |

Non-FLAC files are expected to include yt-dlp sidecars, download archives, and cache files. They
still need a later classifier pass before deletion.

## Queue State

All YouTube queue videos are currently pending and no clips are queued. This is a good pause point
for changing the segmenter before producing more clip data.

| Project | Videos | Video states | Clips |
|---|---:|---|---:|
| `dari-asr` | `8,399` | `pending=8,399` | `0` |
| `farsi-asr` | `12,225` | `pending=12,225` | `0` |
| `georgian-asr` | `12,696` | `pending=12,696` | `0` |
| `tajik-asr` | `27,347` | `pending=27,347` | `0` |

Path classes in `queue.sqlite.videos`:

| Project | Project-local `data/create` refs | Massive archive refs |
|---|---:|---:|
| `dari-asr` | `8,383` | `16` |
| `farsi-asr` | `12,116` | `109` |
| `georgian-asr` | `12,584` | `112` |
| `tajik-asr` | `27,344` | `3` |

Archive `persian` refs after cleanup:

- `dari-asr`: `0`
- `farsi-asr`: `0`
- `georgian-asr`: `0`
- `tajik-asr`: `0`

## 2026-06-30 Queue Metadata Refresh

Existing YouTube queues were refreshed from the current channel registries with
`<lang>-curate repair-metadata`.

| Project | Rows refreshed | Uncategorized rows | Rows with `meta.webpage_url` |
|---|---:|---:|---:|
| `dari-asr` | `8,399` | `0` | `8,399` |
| `farsi-asr` | `12,225` | `0` | `12,225` |
| `georgian-asr` | `12,696` | `0` | `12,696` |
| `tajik-asr` | `27,347` | `0` | `27,347` |

Russian was skipped because `/mnt/workerssd-2t/peacock-asr/russian-asr/queue.sqlite` has no active
split-pipeline queue tables, and `projects/russian-asr` has no YouTube channel registry.

Current category distribution:

- `dari-asr`: `news=8,399`
- `farsi-asr`: `audiobook=220`, `news=12,005`
- `georgian-asr`: `news=12,696`
- `tajik-asr`: `audiobook=4,429`, `children=316`, `documentary=111`, `education=544`,
  `entertainment=704`, `food=706`, `interview=2,676`, `language_learning=253`, `news=13,140`,
  `podcast=337`, `religion=1,025`, `talk=2,863`, `vlog=243`

Each refreshed queue row also has `meta.tier` and `meta.category` matching the row columns. The
direct per-video YouTube URL is now in `videos.meta.webpage_url`; citations remain channel-level.

## Channel Coverage Snapshot

Queue channels:

- `dari-asr`: `tolonews=7701`, `tolonews_talkshows=698`
- `farsi-asr`: `iran_international=12005`, `avas_book_club=220`
- `georgian-asr`: `gpb_first_channel=12696`
- `tajik-asr`: broad multi-channel queue; largest are `radio_ozodi=11486`,
  `asiaplus=4285`, `samo_tajikistan=1049`, `najm_tv=704`, `tajik_show=661`

Some create roots contain channel folders outside the queue's current channel set. Farsi's local
FLAC set has since been reconciled; other projects still need the same queue-vs-folder
classification before production segmentation, and new Farsi registry downloads will add more
folders and queue rows.

Examples:

- `dari-asr/data/create`: includes `@1TVKabul`, `@AfghanComedyOfficial`, `@TOLOnews`,
  `ariana_news`, `etilaatroz`, `hasht_e_subh`, `tamadon_tv`, `tolonews`, `tolonews_talkshows`,
  `voa_dari`
- `farsi-asr/data/create`: active FLACs are `avas_book_club` and `iran_international`; `manoto`
  exists only as an empty/local placeholder in this snapshot and remains part of the broad registry
  download work
- `georgian-asr/data/create`: includes `adjara_tv`, `audiobooks_geo_ka`, `formula_tv`,
  `gpb_first_channel`, `imedi_tv`, `mtavari_arkhi`, `radio_tavisupleba`, and others
- `tajik-asr/data/create`: includes many more channels than the largest queue subset

## 2026-06-30 Farsi Refresh

Massive is responsive again for metadata-scale checks. The Farsi legacy-only files have been copied
into the canonical archive, and the same-name conflicts have been resolved without redownload.

Archive state:

- `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international`: `2,456` files, `2,452`
  FLACs.
- `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy`: `8.4G`, `431`
  files, `430` FLACs, and `archive_manifest.jsonl`.
- The non-destructive merge copied `268` regular files from legacy into canonical
  `iran_international`: `267` legacy-only FLACs plus `archive_manifest.jsonl`.
- The `16` common-name size mismatches were valid duplicate audio variants, not corrupt files:
  all `32` files passed FLAC validation, and their decoded-audio MD5s differed before promotion.
- The legacy `archive_manifest.jsonl` matched the legacy sizes for all `16` conflicts, so the
  manifest-backed legacy variants were promoted into canonical `iran_international`.
- The previous canonical variants are preserved at
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_conflicts_2026-06-30/current_variants_before_legacy_promotion`.
- Filename/audio comparison now finds `0` legacy-only FLAC names, `430` common FLAC names, and `0`
  decoded-audio MD5 mismatches between canonical and legacy for common names.
- No rsync/temp leftovers were found under the Farsi archive.

Farsi source and queue state:

- `projects/farsi-asr/data` points at `/mnt/tiny-2t/peacock-asr/farsi-asr/data`.
- `data/create` contains `12,213` FLACs: `avas_book_club=220`, `iran_international=11,993`,
  `manoto=0`.
- Queue is now `12,225` videos, all `pending`, with `0` clip rows.
- Queue paths are `12,116` project-create refs and `109` canonical Massive archive refs.
- All recorded Farsi queue paths exist: `0` project-create refs are missing and `0` archive refs are
  missing.
- The former `428` missing project-create paths were copied back from canonical
  `farsi/iran_international` to `/mnt/tiny-2t/peacock-asr/farsi-asr/data/create/iran_international`
  on 2026-06-30. The copy transferred `428` files / `8.92G` and the post-copy check returned
  `missing_after 0`.
- The `444` completed local `iran_international` FLACs that were not present in the queue were
  enqueued on 2026-06-30.
- The `220` Avas Book Club FLACs were renamed from `@AvasBookClub` to registry slug
  `avas_book_club`, enqueued, and metadata-repaired on 2026-06-30.
- Local FLACs still not present in the queue: `0`.
- Farsi queue metadata is refreshed: `iran_international` rows are `category=news`,
  `avas_book_club` rows are `category=audiobook`, and all `12,225` rows have `meta.webpage_url`,
  `meta.tier`, and `meta.category`.
- By stem, `12` queued Farsi videos remain archive-only because their recorded paths point at
  Massive; this is intentional and different from a missing recorded path.
- The broad Farsi registry has `124` YouTube channels. Current local FLACs exist for `2` registry
  slugs (`iran_international`, `avas_book_club`); the other `122` registry channels still need
  download/enqueue work for a full broad Farsi corpus.
- A channel-level download-state manifest was written at
  `projects/farsi-asr/data/audit/farsi-channel-download-state-2026-06-30.tsv`; the missing-channel
  subset is `projects/farsi-asr/data/audit/farsi-missing-registry-channels-2026-06-30.tsv`.
  Representative missing channels (`manoto`, `bbc_persian`, `holakouee`) are reachable through the
  current `farsi-curate list --limit 1` path.

Operational conclusion:

- Do not bulk-copy all Farsi source audio to `/` or to `/mnt/workerssd-2t` before the pilot.
  Keep active source audio in `/mnt/tiny-2t/peacock-asr/farsi-asr/data/create`, use Massive only
  as archive fallback, and write clips to the project `data/clips` default unless an operator
  explicitly chooses a separate scratch clip root.
- Reading from Massive is acceptable for a small fallback set or pilot. It is not a good default
  for high-concurrency full-corpus segmentation: the current segmenter reads/decode each source
  twice, and Massive is an exFAT HDD.
- Do not redownload or rename the `16` former conflict files by default. Canonical now holds the
  manifest-backed legacy variants, and the previous canonical variants are preserved separately for
  provenance.
- Segmenting the current Farsi queue will include both news/broadcast (`iran_international`) and
  clean audiobook (`avas_book_club`) rows. Use a small pilot before scaling the full queue.

## Current Blockers

1. Farsi scope choice is now the lead decision.
   - The current two-channel Farsi queue is pilot-ready from a source/metadata standpoint:
     `iran_international=12,005`, `avas_book_club=220`, all `pending`, `clips=0`.
   - The broad Farsi corpus still requires download/enqueue work for the other `122` registry
     channels before it can be called complete.
   - The prepared handoff commands live at
     `projects/farsi-asr/data/audit/farsi-download-commands-2026-06-30.txt`.
2. Source-cache throughput policy is still open before full production.
   - `farsi-asr`: all recorded queue paths exist, with `12` intentional archive-only queued stems.
   - `dari-asr`: `6,041` recorded project-create paths missing locally; all `6,041` have Massive
     archive fallback under the same channel/name.
   - `georgian-asr`: `4,112` recorded project-create paths missing locally; all `4,112` have
     Massive archive fallback under the same channel/name.
   - `tajik-asr`: `20,261` recorded project-create paths missing locally; all remaining local gaps
     have Massive archive fallback.
   - The former `31` Tajik missing-from-both rows were re-downloaded to their recorded
     `data/create` paths on 2026-06-30. The manifest/logs are under
     `projects/tajik-asr/data/audit/`.
   - This is a throughput choice, not a data-loss blocker: pilots can use the resolver fallback;
     full runs should copy bounded channel/language batches back to Tiny2T or explicitly accept
     archive-read throughput.
3. The replacement segmenter/VAD path still needs to land behind the queue contract.
   - Current queues are paused cleanly with `clips=0`, which is the right state for the swap.
   - The pilot should verify source resolution, clipping ownership, duration gates, and downstream
     labelq/harvest behavior before scale-out.
4. The repo has a broad dirty worktree across curator, docs, project packages, and finetune core.
   - Stabilization should follow this audit, before production resegmentation.

## Recommended Order

1. Pick the Farsi operating scope.
   - Earliest signal: run a small pilot on the current two-channel queue.
   - Broad coverage: run the clean/noisy download, enqueue, and repair-metadata commands from
     `projects/farsi-asr/data/audit/farsi-download-commands-2026-06-30.txt` first.
   - Either path should keep clips under the project `data/clips` default unless an operator
     explicitly chooses a separate scratch root.

2. Lock the source-path policy for the next run.
   - For pilots, use Tiny2T source audio plus Massive fallback and verify the resolver path.
   - For production, copy bounded archive-backed channel/language batches back to Tiny2T when HDD
     read throughput becomes the bottleneck.
   - Keep a cheap path audit in the runbook: missing project-create paths, Massive fallback hits,
     archive-only queue rows, and local folders outside queue coverage.

3. Freeze old segmentation output.
   - Confirm `clips=0` stays true or explicitly archive/delete old clips after verification.
   - Hold new `segment` runs until the replacement segmenter is selected and tested.

4. Update the queue/source model for resegmentation.
   - Decide whether queue `videos.path` should point to project-local source caches, archive paths,
     or a resolver key plus channel/video id.
   - Add a cheap path audit command that reports missing source files, archive fallbacks, and stale
     channel folders.

5. Swap in the new segmenter behind a guarded interface.
   - Keep the existing queue contract.
   - Add fixture/equivalence tests on a few known source files per language.
   - Gate by max-duration, no orphan workers, and deterministic output ownership.

6. Run one small Farsi resegmentation pilot.
   - Start with a bounded `iran_international` + `avas_book_club` slice so the pilot includes
     broadcast/noisy and clean audiobook material.
   - Produce clips, label a small batch, harvest, and verify.
   - Use pilot output to choose duration thresholds, archive fallback behavior, and whether to
     download the full Farsi registry before larger segmentation.

7. Stabilize and commit the repo.
   - Keep `CURATION_FACTORY.md` as the current plan.
   - Keep active work in `TODO.md`.
   - Move history into `CHANGELOG.md`.
   - Run focused tests before committing.

8. Run full production in project order.
   - If prioritizing the original Farsi goal, scale Farsi after the pilot decision above.
   - If broad Farsi downloads are intentionally deferred, Georgian and Dari are the smaller
     non-Farsi queues to exercise next.
   - Tajik should come after the smaller runs because it has the largest queue and the largest
     archive-backed source gap.

## Deferred Cleanup

Massive is responsive enough for shallow scans. The non-destructive Farsi merge has run:

```bash
rsync -a --ignore-existing \
  /mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy/ \
  /mnt/massive-22t/peacock-asr-archive/farsi/iran_international/
```

The same-name conflict audit is complete. Keep `iran_international_legacy` only until an explicit
cleanup checkpoint confirms the docs/changelog are committed and the preserved current variants are
still present under
`/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_conflicts_2026-06-30/current_variants_before_legacy_promotion`.
