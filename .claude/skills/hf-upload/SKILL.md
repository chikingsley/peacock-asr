---
name: hf-upload
description: Upload a large dataset or model folder to the Hugging Face Hub so files actually land and stay. Use whenever pushing datasets/models to the Peacockery org, especially over a slow or flaky uplink. Covers the per-shard single-commit pattern and why upload_large_folder is the wrong tool here.
---

# Uploading large folders to the Hugging Face Hub

## The rule: one file per commit

Use **`scripts/hf_upload_dataset.py`** (per-shard, single-commit). Each file is uploaded as its
own commit, so it lands on the Hub the instant it finishes, already-uploaded files are skipped on
re-run, and a dropped connection costs at most the one file in flight.

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run --with "huggingface_hub[cli,hf_transfer]" \
  python scripts/hf_upload_dataset.py Peacockery/<repo> <local_folder> \
  --repo-type dataset --glob '*.parquet'
```

Run it inside `tmux` for anything large; it is safe to kill and re-run — it resumes from what is
already committed on the Hub. Upload the README/card separately with `hf upload <repo> card.md
README.md --repo-type dataset` (also a single commit).

## Do NOT use upload_large_folder / `hf upload-large-folder` for this

They **pre-upload every file first and commit once at the very end**. Consequences on this box's
uplink (which has measured as low as ~14 MB/s):

- the repo shows **0 files for hours** — no way to tell real progress from a hang;
- an interruption (reboot, crash, kill) loses the entire pre-upload — nothing is committed;
- the final single commit can itself hang with no per-file granularity.

This pattern cost a full night on the 191 GB `tajik-asr-youtube` upload. The per-shard script
replaced it and committed visibly from file 1. Only reach for `upload_large_folder` on a fast,
reliable connection where the all-at-once commit is actually an advantage.

## Throughput reality

Total throughput is bounded by the machine's upstream, not the tool. `hf_transfer` (the Rust path,
enabled by `HF_HUB_ENABLE_HF_TRANSFER=1`) parallelizes a single file's chunks but cannot exceed the
physical uplink. If an upload is "too slow," measure the line first:

```bash
dd if=/dev/urandom of=/tmp/up.bin bs=1M count=300 2>/dev/null
curl -s -o /dev/null -w '%{speed_upload}\n' -T /tmp/up.bin https://speed.cloudflare.com/__up | \
  awk '{printf "%.0f Mbps up\n", $1*8/1e6}'; rm -f /tmp/up.bin
```

If that reads far below the plan's rated upstream, the bottleneck is the line (ISP/switch/router),
not the upload code — no tool fixes that.

## Don't keep huge staging copies on the system SSD

The HF audio-dataset exporter (`omni_curator.publish.export_hf_audio_dataset`) writes a parquet
re-encoding of the store — a full second copy of the audio. That staging is **regenerable** (rebuilt
from the curator store in minutes), so delete it after the upload commits rather than letting it sit
on `/`. Project data should live on the `/mnt/overflow` mount (symlinked back), not the system SSD.
