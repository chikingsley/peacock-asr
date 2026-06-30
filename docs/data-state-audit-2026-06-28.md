# Peacock ASR Data State Audit - 2026-06-28

Status: live-state audit after the root `peacock-create` cleanup and the archive `persian` bucket
cleanup. This document is a handoff for the pause-and-stabilize phase before resegmentation.

## Current Storage Layout

Canonical cold archive:

- `/mnt/massive-22t/peacock-asr-archive`
- Top-level buckets now present: `dari`, `farsi`, `georgian`, `russian`, `tajik`
- The stale top-level `persian` bucket was removed.
- Former `persian/iran_international` material was moved under:
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy`
- Its manifest was moved to:
  `/mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy/archive_manifest.jsonl`

Working mirror/project roots:

- `/mnt/tiny-2t/peacock-asr/dari-asr` - `188G`
- `/mnt/tiny-2t/peacock-asr/farsi-asr` - `269G`
- `/mnt/tiny-2t/peacock-asr/georgian-asr` - `105G`
- `/mnt/tiny-2t/peacock-asr/tajik-asr` - `169G`
- `/mnt/workerssd-2t/peacock-asr/russian-asr` - Russian working/canonical audio only

Current filesystem free space:

| Mount | Used | Free | Use |
|---|---:|---:|---:|
| `/` | `597G` | `1.2T` | `35%` |
| `/mnt/tiny-2t` | `881G` | `859G` | `51%` |
| `/mnt/workerssd-2t` | `1.2T` | `663G` | `64%` |
| `/mnt/massive-22t` | `11T` | `9.3T` | `54%` |

## Project Source Caches

Project-local `data/create` is the active source-audio landing area.

| Project | `data/create` size | FLAC files | Non-FLAC files |
|---|---:|---:|---:|
| `dari-asr` | `186G` | `3,880` | `1,521` |
| `farsi-asr` | `255G` | `11,785` | `1,534` |
| `georgian-asr` | `104G` | `11,166` | `2,022` |
| `tajik-asr` | `164G` | `7,539` | `1,449` |

Non-FLAC files are expected to include yt-dlp sidecars, download archives, and cache files. They
still need a later classifier pass before deletion.

## Queue State

All YouTube queue videos are currently pending and no clips are queued. This is a good pause point
for changing the segmenter before producing more clip data.

| Project | Videos | Video states | Clips |
|---|---:|---|---:|
| `dari-asr` | `8,399` | `pending=8,399` | `0` |
| `farsi-asr` | `11,561` | `pending=11,561` | `0` |
| `georgian-asr` | `12,696` | `pending=12,696` | `0` |
| `tajik-asr` | `27,347` | `pending=27,347` | `0` |

Path classes in `queue.sqlite.videos`:

| Project | Project-local `data/create` refs | Massive archive refs |
|---|---:|---:|
| `dari-asr` | `8,383` | `16` |
| `farsi-asr` | `11,452` | `109` |
| `georgian-asr` | `12,584` | `112` |
| `tajik-asr` | `27,344` | `3` |

Archive `persian` refs after cleanup:

- `dari-asr`: `0`
- `farsi-asr`: `0`
- `georgian-asr`: `0`
- `tajik-asr`: `0`

## Channel Coverage Snapshot

Queue channels:

- `dari-asr`: `tolonews=7701`, `tolonews_talkshows=698`
- `farsi-asr`: `iran_international=11561`
- `georgian-asr`: `gpb_first_channel=12696`
- `tajik-asr`: broad multi-channel queue; largest are `radio_ozodi=11486`,
  `asiaplus=4285`, `samo_tajikistan=1049`, `najm_tv=704`, `tajik_show=661`

Create roots contain more channels than the queue currently names. That means the next audit pass
must reconcile downloaded source folders against queue rows before running the replacement segmenter.

Examples:

- `dari-asr/data/create`: includes `@1TVKabul`, `@AfghanComedyOfficial`, `@TOLOnews`,
  `ariana_news`, `etilaatroz`, `hasht_e_subh`, `tamadon_tv`, `tolonews`, `tolonews_talkshows`,
  `voa_dari`
- `farsi-asr/data/create`: includes `@AvasBookClub`, `iran_international`, `manoto`
- `georgian-asr/data/create`: includes `adjara_tv`, `audiobooks_geo_ka`, `formula_tv`,
  `gpb_first_channel`, `imedi_tv`, `mtavari_arkhi`, `radio_tavisupleba`, and others
- `tajik-asr/data/create`: includes many more channels than the largest queue subset

## Current Blockers

1. The archive disk/path was too slow for an interactive file-level dedup of
   `farsi/iran_international_legacy` into `farsi/iran_international`.
   - A checksum dry-run did not complete interactively.
   - A copy with `rsync --ignore-existing` ran at roughly `100-300kB/s` and was stopped.
   - One interrupted rsync temp file was removed from `farsi/iran_international`.
   - The legacy files are preserved under the `farsi` bucket for a later off-hours dedup pass.
2. The queue paths are mixed between project-local source caches and archive refs.
   - This is workable only if the segmenter source resolver is verified before a full run.
3. Category metadata is still `uncategorized` in the current queue rows.
   - The newer registry category model exists, but existing queue rows predate full category
     coverage or need a metadata refresh.
4. The repo has a broad dirty worktree across curator, docs, project packages, and finetune core.
   - Stabilization should follow this audit, before production resegmentation.

## Recommended Order

1. Finish the source inventory.
   - For each project, compare `data/create/<channel>` folders to `queue.sqlite.videos.channel`.
   - Classify folders as queued, downloaded-not-queued, archive-only, stale cache, or intentionally
     parked.
   - Preserve yt-dlp `downloaded.txt` and `*.info.json` sidecars.

2. Freeze old segmentation output.
   - Confirm `clips=0` stays true or explicitly archive/delete old clips after verification.
   - Refuse new `segment` runs until the replacement segmenter is selected and tested.

3. Update the queue/source model for resegmentation.
   - Decide whether queue `videos.path` should point to project-local source caches, archive paths,
     or a resolver key plus channel/video id.
   - Add a cheap path audit command that reports missing source files, archive fallbacks, and stale
     channel folders.

4. Swap in the new segmenter behind a guarded interface.
   - Keep the existing queue contract.
   - Add fixture/equivalence tests on a few known source files per language.
   - Gate by max-duration, no orphan workers, and deterministic output ownership.

5. Run one small resegmentation pilot.
   - Pick one channel per language.
   - Produce clips, label a small batch, harvest, and verify.
   - Only then scale to full channel queues.

6. Stabilize and commit the repo.
   - Keep `CURATION_FACTORY.md` as the current plan.
   - Keep active work in `TODO.md`.
   - Move history into `CHANGELOG.md`.
   - Run focused tests before committing.

7. Run full production in project order.
   - Suggested order: Georgian, Dari, Farsi, Tajik.
   - Reasoning: Georgian/Dari/Farsi have narrower current queues; Tajik is largest and should
     benefit from lessons from the smaller runs.

## Deferred Cleanup

When Massive is responsive, deduplicate:

```bash
rsync -a --ignore-existing \
  /mnt/massive-22t/peacock-asr-archive/farsi/iran_international_legacy/ \
  /mnt/massive-22t/peacock-asr-archive/farsi/iran_international/
```

Then run a checksum duplicate audit before deleting `iran_international_legacy`.
