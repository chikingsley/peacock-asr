# Design: decouple segmentation from labeling (producer / consumer queue)

Status: **v3 — converged after two Codex 5.5 xhigh review rounds. Ready to implement.** Not yet
implemented. The current fused pipeline keeps running untouched until this lands.

## Problem

Today `label_to_store` → `vad_path`/`chunks_path` does, **per video, in one process**
(`create/pipeline.py:190`):

1. `segment_vad(audio)` — loads a NeMo frame-VAD model on CPU and runs it `batch_size=1`
   (`segmenters/vad.py:100‑103`). **CPU-bound.**
2. *then* `_label_spans(...)` — a `ThreadPoolExecutor(max_workers=workers)` doing, per span,
   `cut_audio` (ffmpeg) → `transcribe_clip` (Scribe ensemble, HTTP) → `compile_down`
   (SuperWhisper, HTTP) (`create/pipeline.py:155‑187`). **I/O-bound.**

The two stages are **sequential inside one process**, and the unit of parallelism is the *channel*
(`xargs -P N` over `curate label --channel X`). Consequences:

- **N redundant model loads** — each channel process loads its own NeMo model (the load spike).
- **Lumpy utilization** — during a channel's VAD phase its Scribe threads idle; during its Scribe
  phase the CPU idles. Neither side is ever saturated.
- **No global queue** — work is partitioned by channel; a channel with one huge video stalls others,
  and Scribe concurrency is implicitly `N_channels × workers`, not a controllable target.

## Goal

Two **decoupled** stages joined by a durable queue:

- **Segment stage (CPU producer):** load the model **once per worker**, run a few processes holding
  the 16 cores near ~80%, cut clips to disk, enqueue them.
- **Label stage (I/O consumer):** a pool sized to a **target Scribe concurrency (~200–250**, hard cap
  500) that continuously drains the queue. Backlog → workers pinned. Empty → workers wait.

No external infra: a local producer/consumer over SQLite.

## Scope of v1 (per Codex "simpler shape")

**VAD path only.** `chunks`/`align` is deferred — it cannot split into independent per-clip labels:
`stitch()` folds ordered labels only after *all* chunks exist (`pipeline.py:235`) and
`align_to_clips()` needs the full reference transcript before final clip creation (`align.py:240`).
Tajik already always passes `path="vad"` (`curate.py:171`), so VAD-only loses nothing today. chunks
becomes a later per-video state machine.

Four commands: **`enqueue` → `segment` → `labelq` → `harvest`.**

## Architecture

```
downloaded FLACs            work queue (data/queue.sqlite, WAL)            canonical
data/create/<slug>/*.flac                                                 data/channels/<slug>/store.sqlite
        │                                                                          ▲
        │  enqueue: seed `videos`                                                  │ harvest (idempotent,
        │                                                                          │ separate step)
        ▼                                                                          │
   ┌──────────┐  claim video    ┌──────────┐  pending clips   ┌──────────┐        │
   │ SEGMENT  │  (BEGIN IMMED.)  │  videos  │                  │  LABELQ  │        │
   │ P_seg    │─────────────────▶│  clips   │─────────────────▶│ 1 disp.  │────────┘
   │ procs,   │  cut→tmp→rename   └──────────┘  batch-claim     │ +W thr.  │  labels+variants
   │ resident │  insert clips                   (lease)         │ HTTP     │  written back to clips
   │ VAD      │  after cut durable                              └──────────┘
   └──────────┘
```

### Queue store: `data/queue.sqlite` (separate from CuratorStore)

Transient work-tracking only; WAL mode. `CuratorStore` stays canonical — its `text` is required and
`upsert` is `INSERT OR REPLACE` (`store.py:106‑114`), so writing work-state through it would clobber
`scribe_wer`/`cer`/`meta`. Hence a separate DB + a harvest step.

```sql
videos(
  video_id   TEXT PRIMARY KEY,        -- "<slug>_<stem>"
  channel    TEXT NOT NULL,
  path       TEXT NOT NULL,           -- source FLAC
  tier       TEXT NOT NULL,           -- clean|noisy (v1: both -> vad)
  status     TEXT NOT NULL,           -- pending|segmenting|segmented|failed
  n_clips    INTEGER,
  attempts   INTEGER NOT NULL DEFAULT 0,
  locked_by  TEXT, locked_at REAL,    -- lease (pid/uuid + monotonic-ish wall time)
  last_error TEXT,
  updated_at REAL
);
CREATE INDEX videos_status ON videos(status);

clips(
  clip_id     TEXT PRIMARY KEY,       -- "<video_id>_<clip_index:04d>"
  video_id    TEXT NOT NULL,
  channel     TEXT NOT NULL,
  clip_index  INTEGER NOT NULL,
  clip_path   TEXT NOT NULL,          -- durable cut FLAC
  start REAL, end REAL,
  language    TEXT NOT NULL, script TEXT NOT NULL, citation TEXT,
  status      TEXT NOT NULL,          -- pending|labeling|done|failed
  attempts    INTEGER NOT NULL DEFAULT 0,
  claim_token TEXT,                   -- guards late results from a retried clip
  locked_at   REAL,
  label       TEXT, variants TEXT,    -- variants = JSON, kept for provenance
  last_error  TEXT,
  done_at     REAL, harvested_at REAL,
  updated_at  REAL
);
CREATE INDEX clips_status ON clips(status);
CREATE INDEX clips_harvest ON clips(status, harvested_at);
```

### `enqueue`

Seed `videos` from `sources.YOUTUBE_CHANNELS` × `data/create/<slug>/*.flac`. Idempotent
(`video_id` PK → re-run is a no-op). Skips video_ids already present (incl. those the fused job
finished, if we choose to exclude them — see Coexistence).

### `segment` (CPU producer)

`P_seg` **processes**, each loading the NeMo VAD model **once** (a new resident-model entry; today's
`segment_vad()` loads inside every call, `vad.py:100`). Each process loops:

1. Claim next `videos.status='pending'` in `BEGIN IMMEDIATE` → `segmenting` + lease. Video claims are
   rare (seconds–minutes apart) → negligible contention.
2. VAD → spans. For each span: `cut_audio` to a **temp** path → `fsync`+`rename` to
   `data/clips/<slug>/<video_id>/seg_XXXX.flac` (durable) → buffer the clip row.
3. Insert all clip rows (`status='pending'`) **after** every cut for that video succeeded, in one txn,
   then mark the video `segmented` (`n_clips`). → labeler never sees a half-cut video.
4. On exception: `attempts++`, `failed` if over cap, else back to `pending`; `last_error` recorded.

`P_seg ≈ 4–6` to start, **cap `OMP_NUM_THREADS`/torch intra-op threads** so processes don't fight for
cores; measure and tune toward ~80% CPU. (Batched cross-video VAD is a *later* optimization, not the
first correctness risk.)

**Backpressure:** before claiming, if `count(clips WHERE status='pending') ≥ HWM` (e.g. 50k clips),
sleep. Prevents a dead labeler from letting the segmenter cut the whole corpus to FLAC.

### `labelq` (I/O consumer)

One process: a **single dispatcher** owns all DB access; a `ThreadPoolExecutor(W_label)` does only
HTTP. (No per-clip `BEGIN IMMEDIATE` from 250 threads — SQLite has one writer; per-worker claimers
just make a lock convoy.)

1. Claim a batch: `UPDATE clips SET status='labeling', claim_token=?, attempts=attempts+1,
   locked_at=? WHERE clip_id IN (SELECT clip_id FROM clips WHERE status='pending'
   ORDER BY clip_id LIMIT :batch) RETURNING …`. `batch ≈ 1–2 × W_label`.
2. Dispatch to the pool. Each thread uses a **thread-local `SuperwhisperClient` + Scribe fns** (not
   the shared client of `pipeline.py:172`): `transcribe_clip` → `compile_down`. No DB in threads.
3. Write results back in batched txns, **guarded by both status and `claim_token`**
   (`… WHERE clip_id=? AND status='labeling' AND claim_token=?`) so a late result from a
   *reclaimed* lease can't overwrite a retry: `status='done'`, `label`, `variants`, `done_at`.
   Failures → `attempts`-capped `failed`/`pending`.
4. Reclaim stale leases (`status='labeling' AND locked_at < now-lease`) → `pending` **and
   `claim_token=NULL`** (clearing the token is what makes step 3's guard reject the old worker's
   late write — status alone is insufficient since the retry re-enters `labeling`).
5. Drain, then idle-poll so it keeps running as `segment` feeds it.

### `harvest` (queue → canonical, idempotent)

Fold `clips.status='done' AND harvested_at IS NULL` into per-channel
`data/channels/<slug>/store.sqlite` as `Sample`s (`source="youtube-<slug>"`,
`id="<video_id>_<idx>"`, empty labels skipped as today). **Preserve provenance**: put `variants` into
`Sample.meta` (today's `to_samples` drops them, `pipeline.py:125`). Mark `harvested_at`.

**Idempotence is store-side insert-if-absent, not upsert.** Cross-DB atomicity (queue mark + store
write) isn't possible, so a crash between them must be safe to re-run. `CuratorStore.upsert` is
`INSERT OR REPLACE` (`store.py:106‑114`) — re-running it would clobber `scribe_wer`/`cer`/`meta` on a
clip that verify has since scored. Add a `CuratorStore.insert_if_absent` (`INSERT OR IGNORE` by
`id`); harvest uses it, so a partial-crash re-run that re-writes an already-stored clip is a no-op.
The existing `merge → curator.sqlite` flow is unchanged (`curate.py:183`).

## New files (nothing existing is edited until sign-off)

```
packages/omni-curator/src/omni_curator/create/
  queue.py     # QueueStore: schema, enqueue, claim/lease, batch-claim, write-back, harvest helpers
  segment.py   # resident-model VAD producer loop (one process); segment_run(queue, ...)
  labelq.py    # dispatcher + HTTP threadpool consumer loop; label_run(queue, ...)
```

`pipeline.py`'s span-cut and Scribe halves are reused (segment uses the VAD+cut half; labelq uses the
`transcribe_clip`/`compile_down` half). The fused `label_to_store` stays for back-compat and the
running job; deprecated once the split is proven. Project wiring: `<lang>-curate
enqueue|segment|labelq|harvest`. Georgian/Persian inherit it (logic is in the package).

## Coexistence with the running fused job (no double-labeling)

- **Fresh `queue.sqlite`, fresh cuts root** (`data/clips/`, distinct from the fused job's
  `data/labeled/<slug>/<stem>`).
- **Partition by channel, not by video.** A per-`video_id`-prefix exclusion only catches videos the
  fused job has *landed*; the fused loop stores a channel's rows at the *end* (`curate.py:161,177`),
  so an in-flight channel has no rows yet and a prefix check would wrongly enqueue its videos →
  double-label. So `enqueue` takes only channels the fused job is **not** touching — concretely the
  channels with **no `data/channels/<slug>/store.sqlite` yet** (the ~14 not-yet-started), recorded as
  an explicit allow-list at enqueue time. Shared source FLACs are fine; output (cuts, IDs, stores)
  stays disjoint by construction.
- The fused loop finishes its in-flight channels; we cut over fully once VAD-split is proven on a
  couple of the not-yet-started channels and the hours/quality match.

## Resolved decisions (was "open", now locked with Codex)

1. **Substrate:** SQLite `queue.sqlite`, separate from CuratorStore. Single dispatcher for label
   claims/writes; per-video `BEGIN IMMEDIATE` for the rare segment claims. (FS queue would hand-roll
   retries/leases/backpressure/harvest — rejected.)
2. **Segment parallelism:** resident-model **processes** (~4–6, capped torch threads). Batched VAD
   deferred.
3. **Output:** labelq writes labels/variants into `queue.sqlite`; a separate idempotent **harvest**
   writes per-channel CuratorStores (avoids the `INSERT OR REPLACE` clobber + cross-DB crash window).
4. **`W_label`:** 200–250, batch-claim 1–2×; pending-clips high-watermark for backpressure.
5. **Coexistence:** fresh queue + fresh cuts root + exclude already-labeled video_ids.
6. **chunks/align:** deferred to a later per-video state machine.

## Race/idempotence guarantees (locked round 2)

- **Reclaimed-lease late write:** reclaim sets `claim_token=NULL`; write-back requires
  `status='labeling' AND claim_token=?`. An old worker whose lease was reclaimed fails the guard.
- **Coexistence:** channel-level partition (only channels with no per-channel store yet) → cuts, IDs,
  and stores are disjoint from the fused job by construction.
- **Harvest re-run:** store-side `insert_if_absent` (`INSERT OR IGNORE`) → a crash between store-write
  and `harvested_at` is safe to re-run; verify's later scores are never clobbered.

## Implementation notes (tune at build time)

- Lease duration ≫ longest real Scribe call (e.g. 10 min); reclaim only well past it.
- `SuperwhisperClient` / Scribe fns: instantiate per-thread (thread-local) — they're HTTP wrappers,
  safe default regardless of proven thread-safety.
- Tune `P_seg` and `W_label` empirically against ~80% CPU / ~200–250 Scribe concurrency.
```
