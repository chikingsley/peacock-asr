# omni-curator Factory (#9) — Plan (for review; NOT yet implemented)

## Goal

The operator does exactly **two** manual things per language:

1. **`download`** — pull a language's channel audio (the one input).
2. **`export`** — decide + run the final ablation (the one output).

Everything between — enqueue → segment → labelq → harvest → (merge) → verify, plus
**archive-on-processed** to keep `/mnt/overflow` drained — runs automatically, driven by *claimable*
work appearing in the DBs. No manual chaining, no watchdog cron, and — once clip tiering (§4) is
chosen — no disk-full emergencies.

> This plan was codex-audited; the round-1 audit found real correctness gaps (videos have no claim
> token; `next_step` is a worklist not a scheduler; `merge`/`verify` aren't safely repeatable; pidfiles
> are unsound; clips are durable disk pressure). The design below incorporates the fixes, and calls out
> the **prerequisite code changes** the factory depends on (do these first — they are real bugs even
> without a factory).

## 0. Prerequisite fixes (land BEFORE the factory; each is an independent correctness bug)

- **P1 — video claim tokens + lease refresh.** `queue.videos` has `locked_by`/`locked_at` but **no
  claim token**; `complete_video()`/`fail_video()` update by `video_id` only. A segmenter whose lease
  expired (long video) can have its row reclaimed by a retry, and a late `fail_video()` can flip a
  successfully-`segmented` row back to `pending`/`failed`. Add a `claim_token` (or epoch) to `videos`,
  guard `complete_video`/`fail_video` by it, and have a long segment job **refresh its lease**
  periodically. Note `run_segmenters` spawns **`--procs` worker processes** that claim videos
  concurrently and **race each other on video rows today** (a long video can be reclaimed by another
  internal worker after `stale_after_s`). Until P1 lands the factory MUST run segment at **`--procs 1`**
  (one worker, no internal race); **P1 is the hard prerequisite for multi-proc segmenting.**
- **P2 — merge must preserve verification.** `cmd_merge()` uses `upsert()`/`INSERT OR REPLACE`, which
  overwrites master rows and **wipes `scribe_wer`/`meta`** if a channel store is re-merged after verify.
  Switch master-merge to **insert-if-absent** (`insert_if_absent` already exists) or a column-merge that
  preserves `scribe_wer`/`cer`/`meta`. Add a real "unmerged rows" predicate (a `merged_at` watermark) so
  the factory can tell when merge is needed and never re-clobbers.
- **P3 — verify needs an "unscoreable" sentinel + a verify lock.** Verify drops empty-normalized samples
  but leaves their `scribe_wer` NULL, so a naive `scribe_wer IS NULL` trigger **relaunches verify
  forever**. Persist an unscoreable marker (a `scribe_status` column, or a sentinel score) so the trigger
  counts only **scoreable** unscored rows. Verify has **no row-level claim**, so two verify processes
  double-spend Scribe — the factory must guarantee a single verify per project (see the lock below).
- **P4 — download must be factory-mediated and abortable.** `cmd_download` loops channels and only
  checks `--disk-guard` *before* each channel; `download_channel()` then hands a whole channel to one
  yt-dlp subprocess, so it cannot stop mid-channel. For the backpressure + hard-halt guarantees (§4) to
  hold, **download becomes a supervised stage** with its own `flock` and a channel work-queue that pulls
  **one channel at a time with a free-space check before each**, and `download_channel` is rewritten to
  fetch **one video at a time** (or use yt-dlp's between-entry hooks) with a **per-video disk guard** so it
  aborts mid-channel when overflow crosses the floor / on a hard-halt signal. The operator still *triggers*
  a language's download (the one manual input); the factory *mediates how it runs.*

## 1. The single-writer lock (the load-bearing primitive — replaces pidfiles)

Each long-running stage is owned by a **per-(project, stage) `flock`** on `data/.lock.<stage>`:

- A stage process acquires the exclusive `flock` at startup and holds it for its lifetime (the lock
  releases automatically on exit/crash/`kill -9` — no stale-PID problem, no PID reuse).
- The supervisor decides "is stage X already running?" by **non-blocking `flock` try-acquire**: if it
  can't get the lock, a live owner exists (manual OR factory-launched) and it leaves it alone; if it
  can, no one owns the stage and it may launch one.
- **Launch handshake:** after spawning a stage the supervisor does NOT count it as running until it
  confirms the child has **acquired** the lock (poll until held, short timeout); a child that dies before
  acquiring is retried. This closes the window where a child exits early but the supervisor thinks it is up.
- This gives **exactly-one-writer-per-(project, stage)**, which is what makes the rest safe: no duplicate
  labelq/verify (so the Scribe budget can't be doubled), no duplicate segmenter (covering the P1 gap even
  before video tokens land), and clean adoption of a manually-started stage.

Children spawn in their **own process group** and are tracked by `Popen` handle so the supervisor can
signal/reap them; the `flock` is the source of truth for liveness, the handle is for control.

## 2. Architecture — a single global **polling supervisor** (chosen over event-driven)

One long-lived process; every `TICK` (default 30 s), for each registered project, it **computes
independent per-stage predicates from the DBs** (NOT `next_step` — see below), then for each stage whose
predicate holds and whose `flock` is free, launches that stage. It also runs the **balancer** across all
live Scribe jobs.

**Why not `next_step()`:** `tools/status.py:next_step()` is a *sequential human worklist* (returns ONE
"do this next" string, and its `MERGE` branch is currently unreachable). Driving a scheduler off it would
serialize stages and skip steps. Instead: **extract the shared scan/count helpers** from `status.py` into
the package, and have the factory evaluate each stage's predicate **independently and in parallel** —
segment, labelq, and archive for one language are all eligible at once. The board keeps using the same
helpers for its human view.

**Why polling, not event-driven:** the DBs are the single source of truth; re-deriving state each tick is
inherently **restart-safe and idempotent** (a reboot just relaunches the supervisor — no event log to
lose or replay). Stages are self-draining, so a 30 s tick is ample.

### Per-stage predicate ("claimable now") / kind / safety

Triggers must mean **work claimable right now**, not just "rows in a transient state" — else a daemon
starts, finds nothing claimable, exits, and gets relaunched in a tight loop until a lease expires.

| Stage | Predicate (claimable now) | Kind | Safety |
|---|---|---|---|
| download | operator-triggered for a language; runs while channels remain AND not backpressured | daemon | own `flock`; channel queue; per-video disk guard (P4) |
| enqueue | `create/` has downloaded videos absent from `videos` | one-shot | insert-if-absent; `flock` |
| segment | `videos.status='pending'` OR (`segmenting` AND lease stale) > 0 | daemon | single-writer `flock`; **`--procs 1` until P1** |
| labelq | `clips.status='pending'` OR (`labeling` AND lease stale) > 0 | daemon | single-writer `flock`; clip `claim_token` |
| **archive** | `videos.status='segmented'` AND source FLAC still on disk | periodic one-shot | idempotent; `flock` |
| harvest | `clips.status='done' AND harvested_at IS NULL` > 0 | one-shot | `insert_if_absent` |
| merge | channel-store rows with no master row (P2 predicate) | one-shot | insert-if-absent (P2) |
| verify | **scoreable** `samples.scribe_wer IS NULL` > 0 (P3) | daemon | single verify `flock` (P3) |
| export | — | **MANUAL** | never auto-run |

A drained stage makes its predicate false, so it isn't relaunched — until new upstream work flips it
true again. Segment/labelq/archive for one language run **concurrently** (the queue decouples them).

## 3. Archive-on-processed (the behavior you asked for)

Archive fires whenever a `status='segmented'` video still has its source FLAC on `/mnt/overflow` — i.e.
**as soon as a video is processed, its now-redundant source drains to `/mnt/storage`.** It is NOT gated
on disk-free, so **segmented sources never accumulate** → that category of overflow pressure stays ~0.
`--only-if-free-gb` is therefore **redundant as the archive trigger** (demoted to a manual override).

## 4. Overflow resource model (corrected — clips are durable too)

Overflow holds **three** byte pools, and the factory tracks all three:

1. **Un-segmented sources** — transient, awaiting segment. The producer (download) can outrun the drain.
2. **Segmented-but-unarchived sources** — held to ~0 by archive-on-processed.
3. **Cut clips** — **durable**: `harvest` stores `audio_path = clip_path`, and verify/export read those
   clips. They are NOT freed after harvest; they are standing pressure that grows with the corpus.

→ Three controls (a byte cap per pool + a hard halt), not one:

- **Download backpressure** (pool 1): the supervisor **withholds new download starts** when overflow free
  < a floor (default 150 GB) or the un-segmented backlog exceeds a cap. CAVEAT: download is a batch over
  channels (`cmd_download` loops channels; `download_channel` hands a whole channel to yt-dlp), so the
  supervisor can gate *new* starts and stop *between channels* but **cannot pause an in-flight channel**.
  To make this a real bound, download runs as the **supervised `download` stage (P4)**: its own `flock`, a
  channel work-queue pulling **one channel at a time with a free-space check before each**, and a
  **hard per-video disk guard inside the rewritten `download_channel`** (abort mid-channel if free < floor
  or on a hard-halt signal) as the backstop for a single huge channel.
- **Clip tiering / retention** (pool 3): clips are durable and must eventually leave overflow. This is an
  operator decision (ties into #7): keep clips on a roomier tier (`/mnt/media` or fast-ssd) from the
  start, and/or **archive clips after export** (the master store + parquet export are the durable
  artifacts; raw clips are reproducible from archived sources). **This must be chosen before the
  "no disk-full" guarantee holds** — accumulated clips alone can fill overflow.
- **Hard halt (the safety net):** the supervisor enforces a byte cap per pool + a hard overflow floor. If
  the **archive target (`/mnt/storage`) errors or fills**, or the **clip pool exceeds its cap**, or
  overflow free crosses the hard floor, the factory **stops both download AND segment** (not just gates
  new starts) until the pressure clears — so segmenting can't keep writing clips while the drain is broken.

## 5. Sharing concurrency across languages

- **Scribe** (the bottleneck): each tick the supervisor runs the balancer over all live labelq+verify
  jobs (deduped by `lang:stage`), splitting one `--budget` so their sum stays within capacity. Because the
  single-writer `flock` makes duplicate Scribe jobs impossible, the balancer's per-job window is the true
  concurrency. **Write a conservative window file before launching a new Scribe job**, so a newly-spawned
  job never briefly runs at full window before the next rebalance.
- **CPU** (segment): a global cap on concurrent segment processes across languages (default cores−2),
  allocated round-robin.
- **Network** (download): one VPN lane per language (existing mapping).

## 6. Failure / restart safety

- **Reboot:** relaunch the supervisor (a uv-run systemd unit, or a boot hook). It re-derives state from
  the DBs and resumes; dead stages are relaunched because their predicate still holds. No double-run:
  `flock` (single writer) + clip `claim_token` + video `claim_token` (P1) + insert-if-absent.
- **A stage dies mid-run:** its `flock` releases; next tick the predicate is still true and the lock is
  free → relaunch. Queue lease-reclaim returns half-claimed rows (clips today; videos after P1).
- **Scribe outage:** the breaker rides through blips; a sustained outage aborts labelq/verify, the lock
  frees, and the supervisor relaunches on a **capped backoff** (track consecutive immediate-exits per
  stage and widen the relaunch interval — don't hammer a dead service).
- **Coexistence:** a human-started stage holds the `flock`; the supervisor sees it held and adopts (does
  not duplicate) it.

## 7. Start / stop / observability

- `uv run -m omni_curator.factory --budget 300 [--projects dari,farsi,...]` starts the supervisor
  (foreground, or via a uv-run systemd unit). `--once` does a single reconcile tick (testing / cron).
- Each tick rewrites `STATUS.md` (board stays live) + a `factory.log` of launches/exits/backpressure.
- Stop: SIGTERM the supervisor; running stages keep their locks (use `--drain` to also stop them).

## 8. What this makes REDUNDANT

1. **Manual chaining** + the ad-hoc `archive_chain.sh` from the disk emergency — gone.
2. **`--only-if-free-gb` as the archive trigger** — redundant (archive fires on processed); kept only as a
   manual override.
3. **The watchdog cron** (already deleted) — its "keep jobs alive / relaunch on death" role IS the
   supervisor; do not recreate it.
4. **Manual Scribe-window rebalancing** and **manual labelq/verify relaunch after a blip**.
5. **The "read the board, then run the next step by hand" ritual** — the board's count helpers are now the
   factory's predicates; the board becomes a read-only view.
6. **Routine per-stage `nohup … &` commands** — available for one-offs, no longer the normal path.

## 9. Out of scope / stays manual

`export` (the ablation decision), new-language registration (`sources.py`+`curate.py`), cookie refresh
for bot-blocked downloads, and the `clip_retention` tiering policy decision (#7).

## 10. Open questions for the operator

- TICK (30 s) and the overflow backpressure floor (150 GB) — tune.
- **Clip tiering**: keep clips on overflow until export then archive, or put clips on a roomier tier from
  the start? (Decides pool-3 pressure; ties into #7.)
- Should the factory ever re-trigger `download` to pull newly-published videos on a schedule, or is
  download always operator-initiated? (Plan assumes operator-initiated.)
- One global supervisor (chosen — simpler budget sharing) vs one-per-project.
