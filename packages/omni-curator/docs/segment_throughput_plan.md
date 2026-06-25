# Segment throughput plan — saturate the box

The `segment` stage does ~4 clips/s and leaves a 16-core / 96 GB / RTX 5070 box
idle. This is an architecture problem, not a tuning problem.

## Why the box is idle (code evidence)

1. **One ffmpeg process per clip, serially** — `_cut_clips` (`create/segment.py:45-49`)
   loops over every VAD span and calls `cut_audio` → `to_16k_flac`
   (`process/audio.py:34-41`), which `subprocess.run(["ffmpeg", ...])`. A video with
   200 spans spawns **200 ffmpeg processes** in a `for` loop. Each uses input-side
   `-ss` but with accurate-seek still **re-decodes the source from a keyframe up to the
   clip start**, so cutting N clips is ~O(N × file length) of redundant decode + N
   spawns. The worker waits on fork/exec + decode — it never computes. This is the
   dominant cost.
2. **Millions of tiny files** — each span is its own ~80-200 KB
   `<channel>/<video_id>/seg_NNNN.flac` (`segment.py:40-50`), temp + atomic rename =
   two metadata ops each. Small-file IOPS-bound write; catastrophic on the exfat
   archive drive (`create/archive.py`).
3. **GPU VAD at `batch_size=1`** — `vad.py:165` transcribes one file at a time. The
   5070 starves; NeMo Frame-VAD supports `batch_size>1` + `split_duration`, unused.
4. **RAM untouched** — source is read off disk by VAD, then re-decoded off disk again
   by every clip's ffmpeg. Nothing is decoded once and held. 96 GB idle.
5. **Per-job thread sizing for the whole box, no core pinning** — `run_segmenters`
   (`segment.py:156-159`) sizes off `os.cpu_count()`, so two language jobs each grab 16
   cores and oversubscribe; no `sched_setaffinity`.

## Compatibility constraint (shapes the phasing)

`labelq` sends each clip to the ASR service **as an absolute file path** read off a
shared mount (`scribe/swservice.py:143-155`); `harvest` stores `clip_path` into the
canonical store (`project.py`). So every clip must be an individually-readable file at
label time — tar/sharded output is gated on a service change.

## Best practice (cited)

- Decode-once, slice in memory: `soundfile.read(start, stop)` does a sample-exact
  partial read; collapses O(N×file) → O(file) with one process.
- `ffmpeg -f segment` muxer: one decode pass, N outputs — drop-in if staying on ffmpeg.
- Lhotse `MonoCut` (lazy, resolved at `load_audio()` by seek+partial read) +
  `save_audios(num_jobs)` / `to_shar(shard_size, num_jobs)` — the ASR-standard cutter.
- Sharded output (Lhotse Shar / WebDataset / NeMo tarred), ~100 MB-1 GB shards:
  sequential I/O ~8-15× faster than random small-file reads; kills the exfat inode
  problem.
- GPU-batched VAD with duration bucketing + `split_duration` (inverse batch size for
  long files) — WhisperX reports up to ~11× from uniform-length batching.
- RAM/tmpfs: stage in `/dev/shm`, `posix_fadvise(WILLNEED)` the source pre-VAD.
- Parallelism: N workers = N cores, `*_NUM_THREADS=1` + `torch.set_num_threads(1)`
  (already done), plus `os.sched_setaffinity` pinning + a shared cross-job core budget.

## Phased plan (Phases 1-4 keep queue/store/labelq/harvest unchanged)

| Phase | Change | Expected gain | Downstream |
|---|---|---|---|
| 1 | Decode source once into a numpy array, slice every clip by sample index in RAM, write. No per-clip ffmpeg. Same output files. | ~5-20× on the cut | unchanged |
| 2 | Parallel clip writes (`num_jobs`), `sched_setaffinity` core pinning, shared core budget across language jobs | ~2-4× | unchanged |
| 3 | Batched + length-bucketed GPU VAD (`batch_size>1`, `split_duration`) | ~3-10× on VAD | unchanged |
| 4 | `/dev/shm` staging + `posix_fadvise` page-cache priming | ~1.3-2× | unchanged |
| 5 | Sharded output (Lhotse Shar / WebDataset) — **gated** on the ASR service reading clips by shard+key/byte-range, or labelq staging clips to temp files first | 3-10× write/read/archive | **service change** |

Phase 1 alone takes the cut step from ~4 clips/s into the tens, and makes the box
CPU/decode-bound instead of spawn/IO-bound. Start there; measure before/after on dari.

Key files: `create/segment.py`, `process/audio.py`, `create/vad.py`,
`scribe/swservice.py` (the path constraint), `create/queue.py` + `project.py`.
