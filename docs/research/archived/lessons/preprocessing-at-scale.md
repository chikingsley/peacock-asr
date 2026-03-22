# Preprocessing 283K Audio Examples in a 64GB Container

**Date:** 2026-03-04
**Task:** Pre-extract SeamlessM4T mel features from LibriSpeech alignments and push to HuggingFace Hub
**Dataset:** `gilkeyio/librispeech-alignments` (281K examples, 4 splits) -> `Peacockery/librispeech-phoneme-features`
**Environment:** RunPod CPU pod, 32 vCPUs, 64GB cgroup memory limit, NFS workspace

## The Problem

Training a phoneme scoring head on w2v-BERT 2.0 used `set_transform()` for on-the-fly
feature extraction. This worked but was slow and made `group_by_length` incompatible.
Pre-extracting features into a Hub dataset would eliminate this bottleneck and simplify
the training pipeline.

The challenge: processing 281K audio files through SeamlessM4TFeatureExtractor on a
CPU-only pod with a hard 64GB memory limit, then uploading ~155GB of parquet files to
HuggingFace Hub.

## The Journey: Six Failed Approaches

### Attempt 1: HF Datasets with `num_proc=14` (v3)

```python
ds = ds.map(process_fn, num_proc=14, keep_in_memory=True)
```

**What happened:** Both processes (360 and 500 splits) appeared alive but showed zero
CPU time change over 5 seconds. `strace` revealed all threads stuck on `futex_wait_queue`.

**Root cause:** `keep_in_memory=True` loads the full Arrow table into the parent process.
When `num_proc > 1` triggers `fork()`, the pyarrow thread pool state is copied but
becomes inconsistent across processes, causing a deadlock.

**Lesson:** Never combine `keep_in_memory=True` with `num_proc > 1` in HF datasets.
The fork-safety of Arrow's internal thread pool is not guaranteed.

### Attempt 2: HF Datasets with `num_proc=1` (v4)

```python
ds = ds.map(process_fn, num_proc=1, keep_in_memory=True)
```

**What happened:** Process silently disappeared during the `.filter()` step. No error
in logs, no core dump.

**Root cause:** Likely the 64GB cgroup OOM killer. When the kernel OOM-kills a process
inside a cgroup, it doesn't always leave a trace in the container's logs.

**Lesson:** Silent process death in containers is often the cgroup OOM killer. Check
`/sys/fs/cgroup/memory.max` to know your actual limit, not `free` (which shows host RAM).

### Attempt 3: Simple for-loop, in-memory (v5)

```python
results = {"input_features": [], "labels": [], ...}
for example in dataset:
    results["input_features"].append(process(example))
```

**What happened:** Worked initially at 21-23 ex/s. At 11K examples, the results dict
was ~5GB and growing at 480MB per 1000 examples. The process would OOM at ~20K.

**Root cause:** Storing all processed results in a Python dict in memory. At 104K
examples, the results dict alone would be ~50GB.

**Lesson:** You can't accumulate 100K+ processed audio feature arrays in memory. Each
example's mel features are ~100KB, but 100K of them = 10GB minimum.

### Attempt 4: NFS-backed Arrow dataset (v6)

```python
ds = ds.map(process_fn, num_proc=1)  # Arrow cache on NFS, not memory
```

**What happened:** Processing started at 42.6 ex/s (faster than v5!). But cgroup
memory climbed from 53GB to 58GB in 2 minutes, despite no `keep_in_memory`.

**Root cause:** Linux page cache. When HF datasets writes Arrow files to NFS, the
kernel caches file pages in RAM. These pages count toward the cgroup memory limit.
The kernel was caching the growing output Arrow file, eating ~1GB of cgroup memory
per minute.

**Attempted fix:** `echo 1 > /proc/sys/vm/drop_caches` — but this is read-only
inside containers. The kernel's page cache eviction is the only option, and it
won't evict fast enough when the cgroup is under steady write pressure.

**Lesson:** In cgroup-limited containers, page cache for NFS-backed files counts
toward your memory limit. `drop_caches` doesn't work from inside containers.

### Attempt 5: NFS-backed Arrow + cache dropper (v6b)

Same as v6 but with a background process trying to drop caches. Failed because
`/proc/sys/vm/drop_caches` is read-only in the container.

## The Solution: Bypass Everything (v7)

The winning approach abandoned the HF `datasets` library entirely and used raw
`pyarrow` + `huggingface_hub`:

```python
# Pseudocode for the v7 approach
for parquet_file in source_parquet_files:          # 49 or 66 files
    table = pq.read_table(download(parquet_file))  # ~2GB in memory
    results = process_all_rows(table)               # ~1.2GB output
    pq.write_table(results, output_shard)           # Write to NFS
    evict_page_cache(output_shard)                  # posix_fadvise trick
    del table; gc.collect()                         # Free memory
```

**Key design decisions:**

1. **One source parquet at a time.** Download, process, write, delete. Peak memory
   is bounded by the size of one parquet file (~2GB) plus the output shard (~1.2GB).

2. **Direct pyarrow I/O.** No HF datasets library, no Arrow cache directory, no
   memory-mapped files, no `.map()` infrastructure. Just `pq.read_table()` and
   `pq.write_table()`.

3. **huggingface_hub for uploads.** `HfApi.upload_folder()` pushes the output
   directory of parquet shards directly. No need to construct a `Dataset` object.

4. **`posix_fadvise(POSIX_FADV_DONTNEED)` for page cache eviction.** This is the
   key trick that made it work. After writing each 1.2GB output shard, the kernel
   caches the file pages in memory (counting toward the 64GB cgroup limit). Unlike
   `drop_caches`, `posix_fadvise` works from userspace without root:

   ```python
   fd = os.open(shard_path, os.O_RDONLY)
   size = os.fstat(fd).st_size
   os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
   os.close(fd)
   ```

   A background loop ran this every 2 minutes on all output shards, keeping cgroup
   memory stable at ~20GB instead of climbing to 64GB.

### Results

| Split | Examples | Shards | Processing Time | Upload Time |
|-------|----------|--------|-----------------|-------------|
| train_clean_100 | 28,538 | 24 | ~20 min | ~5 min |
| train_clean_360 | 104,005 | 49 | 72.5 min | 18 min |
| train_other_500 | 148,642 | 66 | 94.4 min | 25 min |
| eval | 2,703 | 2 | ~2 min | <1 min |
| **Total** | **283,888** | **141** | **~190 min** | **~49 min** |

Processing rate: ~23-25 examples/second (CPU-only, feature extraction bound).
Total dataset size on Hub: ~155GB.

## Key Takeaways

### 1. `free` lies in containers

`free` and `/proc/meminfo` show HOST memory. Your actual limit is in
`/sys/fs/cgroup/memory.max` (cgroup v2) or `/sys/fs/cgroup/memory/memory.limit_in_bytes`
(cgroup v1). RunPod CPU pods have a 64GB cgroup limit regardless of host RAM.

### 2. Page cache counts toward cgroup limits

Writing large files to NFS doesn't just use disk — the kernel caches file pages in
RAM, and these count toward your cgroup memory limit. This is invisible to most
monitoring tools and catches people off guard.

### 3. `posix_fadvise(DONTNEED)` is the userspace page cache escape hatch

When `drop_caches` is read-only (any non-privileged container), `posix_fadvise` with
`POSIX_FADV_DONTNEED` tells the kernel to evict specific file pages. It works per-file
from userspace. Run it in a background loop on your output files.

### 4. Sometimes the abstraction IS the problem

HF `datasets` is designed for convenience. But its Arrow caching, memory mapping,
multiprocessing, and progress tracking all consume resources. When you're at the edge
of a memory limit, every abstraction layer costs you. Raw pyarrow + huggingface_hub
gave us full control over memory and I/O patterns.

### 5. Silent process deaths = check the OOM killer

When a process vanishes without errors in containers, the cgroup OOM killer is the
most likely cause. It's invisible from inside the container. Always verify your
memory ceiling before starting large jobs.

### 6. HF datasets + fork() = dragons

`keep_in_memory=True` + `num_proc > 1` causes deadlocks (pyarrow thread pool across
fork). Even `num_proc=1` can fail silently if the dataset + processing buffers exceed
the cgroup limit. If your data doesn't fit comfortably in ~50% of available memory,
don't use `keep_in_memory=True`.
