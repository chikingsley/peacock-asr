"""Upload a folder to the HF Hub one file per commit — durable, resumable, progress-visible.

WHY this exists: `upload_large_folder` / `hf upload-large-folder` pre-upload every file and
then commit in one shot at the very end. On a slow/flaky uplink that means the repo shows
0 files for hours and an interruption loses everything. This uploads ONE file per commit, so
each file lands on the Hub the instant it finishes, already-committed files are skipped on
re-run, and a dropped connection costs at most the file in flight.

  uv run --with "huggingface_hub[cli,hf_transfer]" python scripts/hf_upload_dataset.py \
      Peacockery/<repo> <local_folder> --repo-type dataset --glob '*.parquet'

Set HF_HUB_ENABLE_HF_TRANSFER=1 in the environment for the fast Rust transfer path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def committed_files(repo: str, repo_type: str) -> set[str]:
    from huggingface_hub import HfApi

    return {f.rsplit("/", 1)[-1] for f in HfApi().list_repo_files(repo, repo_type=repo_type)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--glob", default="*", help="which files under folder/ to upload")
    ap.add_argument("--path-prefix", default="data", help="dir in the repo to upload into")
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    files = sorted(p for p in args.folder.rglob(args.glob) if p.is_file())
    if not files:
        sys.exit(f"no files matching {args.glob!r} under {args.folder}")
    on_hub = committed_files(args.repo, args.repo_type)
    done = 0
    for i, f in enumerate(files, 1):
        dest = f"{args.path_prefix}/{f.name}"
        if f.name in on_hub:
            print(f"[{i}/{len(files)}] skip (on hub): {f.name}", flush=True)
            done += 1
            continue
        for attempt in range(1, args.retries + 1):
            r = subprocess.run(
                ["hf", "upload", args.repo, str(f), dest, "--repo-type", args.repo_type],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                done += 1
                print(f"[{i}/{len(files)}] COMMITTED {f.name}  (on hub: {done})", flush=True)
                break
            print(f"[{i}/{len(files)}] attempt {attempt} failed: {r.stderr.strip()[:120]}",
                  flush=True)
        else:
            print(f"[{i}/{len(files)}] GAVE UP on {f.name} after {args.retries} tries", flush=True)
    print(f"DONE: {done}/{len(files)} on hub", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
