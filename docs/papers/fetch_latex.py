#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "tqdm"]
# ///
"""
Fetch LaTeX source for all papers in docs/papers/pdf/ using arXiv IDs
extracted from git rename history.

Usage: uv run docs/papers/fetch_latex.py
"""

import re
import subprocess
import tarfile
import time
import json
from pathlib import Path

import httpx
from tqdm import tqdm

PAPERS_DIR = Path(__file__).parent
LATEX_DIR = PAPERS_DIR / "latex"
ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")

RATE_LIMIT_DELAY = 3  # seconds between requests (arXiv asks for courtesy)


def extract_arxiv_id(old_path: str) -> str | None:
    """Extract arXiv ID from the old git path."""
    filename = Path(old_path).stem
    m = ARXIV_ID_RE.search(filename)
    return m.group(1) if m else None


def build_id_map() -> dict[str, str]:
    """
    Parse git rename history to build:
        new_pdf_basename -> arxiv_id
    Deduplicates: same arXiv ID may appear under multiple new names.
    """
    result = subprocess.run(
        ["git", "show", "HEAD", "--name-status", "--format="],
        capture_output=True, text=True, cwd=PAPERS_DIR.parent.parent
    )
    id_map = {}  # new_basename -> arxiv_id

    for line in result.stdout.splitlines():
        if not line.startswith("R"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        _, old_path, new_path = parts
        if "docs/papers/pdf/" not in new_path:
            continue

        arxiv_id = extract_arxiv_id(old_path)
        new_basename = Path(new_path).name
        id_map[new_basename] = arxiv_id  # may be None for non-arXiv papers

    return id_map


def fetch_latex(arxiv_id: str, dest_dir: Path, client: httpx.Client) -> str:
    """
    Download LaTeX source tarball for arxiv_id and extract to dest_dir.
    Returns status: 'ok', 'no_source', or 'error_...'
    """
    url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or len(resp.content) < 1000:
                return "no_source"  # arXiv returned a PDF (no LaTeX source)
            dest_dir.mkdir(parents=True, exist_ok=True)
            tar_path = dest_dir / f"{arxiv_id}.tar.gz"
            tar_path.write_bytes(resp.content)
            try:
                with tarfile.open(tar_path) as tf:
                    tf.extractall(dest_dir)
                tar_path.unlink()
                return "ok"
            except tarfile.TarError:
                # Sometimes arXiv returns a single .tex file, not a tarball
                tar_path.rename(dest_dir / f"{arxiv_id}.tex")
                return "ok"
        elif resp.status_code == 404:
            return "no_source"
        else:
            return f"error_{resp.status_code}"
    except Exception as e:
        return f"error_{e}"


def main():
    LATEX_DIR.mkdir(exist_ok=True)

    print("Extracting arXiv ID mapping from git history...")
    id_map = build_id_map()

    # Unique arXiv IDs only (deduplicate)
    unique_ids: dict[str, list[str]] = {}  # arxiv_id -> [pdf_names]
    no_id = []
    for pdf_name, arxiv_id in id_map.items():
        if arxiv_id:
            unique_ids.setdefault(arxiv_id, []).append(pdf_name)
        else:
            no_id.append(pdf_name)

    print(f"\nFound {len(unique_ids)} unique arXiv IDs across {len(id_map)} PDFs")
    print(f"No arXiv ID (conference/non-arXiv): {len(no_id)}")
    if no_id:
        for f in no_id:
            print(f"  SKIP (no ID): {f}")

    # Check which we already have
    to_fetch = [
        arxiv_id for arxiv_id in unique_ids
        if not (LATEX_DIR / arxiv_id).exists()
    ]
    already_done = len(unique_ids) - len(to_fetch)
    if already_done:
        print(f"\nAlready downloaded: {already_done}")
    print(f"To fetch: {len(to_fetch)}\n")

    results: dict[str, list[str]] = {"ok": [], "no_source": [], "error": []}

    with httpx.Client(headers={"User-Agent": "peacock-asr research/1.0"}) as client:
        for arxiv_id in tqdm(to_fetch, desc="Fetching LaTeX"):
            dest = LATEX_DIR / arxiv_id
            status = fetch_latex(arxiv_id, dest, client)
            bucket = status if status in results else "error"
            results[bucket].append(arxiv_id)
            if status == "ok":
                papers = ", ".join(unique_ids[arxiv_id])
                tqdm.write(f"  ✓ {arxiv_id}  ({papers})")
            elif status == "no_source":
                tqdm.write(f"  ✗ {arxiv_id} — no LaTeX source (PDF-only submission)")
            else:
                tqdm.write(f"  ! {arxiv_id} — {status}")
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n=== Done ===")
    print(f"  Downloaded:  {len(results['ok'])}")
    print(f"  No source:   {len(results['no_source'])}")
    print(f"  Errors:      {len(results['error'])}")

    summary_path = LATEX_DIR / "fetch_summary.json"
    summary = {
        "downloaded": results["ok"],
        "no_source": results["no_source"],
        "errors": results["error"],
        "skipped_no_arxiv_id": no_id,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
