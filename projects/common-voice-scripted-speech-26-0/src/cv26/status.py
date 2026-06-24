"""Report progress per collection and list which datasets still need MDC term-acceptance.

MDC exposes no "terms accepted" flag, so the only signal is a download attempt: a ``403`` recorded
in the download report means terms aren't accepted yet. This reads the manifest + processing state
+ download report and classifies every queued dataset, and writes the needs-terms list to
``data/reports/needs-terms.tsv`` so it can be worked through on the MDC site over time.
"""

from __future__ import annotations

import json
from collections import Counter

from cv26 import config
from cv26.manifest import load_manifest
from cv26.pipeline import load_state

NEEDS_TERMS_PATH = config.REPORTS / "needs-terms.tsv"


def _download_failures() -> dict[str, str]:
    """Return ``dataset_id -> error`` for failed downloads; a later success clears the entry."""
    failures: dict[str, str] = {}
    if not config.DOWNLOAD_REPORT.exists():
        return failures
    for line in config.DOWNLOAD_REPORT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("status") == "failed":
            failures[event["dataset_id"]] = str(event.get("error", ""))
        elif "sha256" in event:
            failures.pop(event["dataset_id"], None)
    return failures


def _classify(dataset_id: str, state: dict[str, str], failures: dict[str, str]) -> str:
    status = state.get(dataset_id)
    if status == "complete":
        return "done"
    if status in {"converted", "uploaded"}:
        return "processing"
    if status == "no_shards":
        return "no_shards"
    error = failures.get(dataset_id)
    if error and "has not been granted" in error:
        return "needs_terms"
    if error:
        return "failed"
    return "not_tried"


def report() -> None:
    """Print the per-collection breakdown and write the needs-terms list."""
    rows = load_manifest(config.MANIFEST)
    state = load_state(config.PROCESSING_STATE)
    failures = _download_failures()

    counts: dict[str, Counter[str]] = {}
    needs_terms = []
    for row in rows:
        cat = _classify(row.dataset_id, state, failures)
        counts.setdefault(row.collection, Counter())[cat] += 1
        if cat == "needs_terms":
            needs_terms.append(row)

    order = ["done", "processing", "no_shards", "needs_terms", "failed", "not_tried"]
    for collection in sorted(counts):
        c = counts[collection]
        summary = "  ".join(f"{k}={c[k]}" for k in order if c[k])
        print(f"{collection}: {sum(c.values())} langs | {summary}")

    NEEDS_TERMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NEEDS_TERMS_PATH.open("w", encoding="utf-8") as handle:
        handle.write("language\tlocale\tcollection\tmdc_url\n")
        for row in sorted(needs_terms, key=lambda r: (r.collection, r.language)):
            url = f"https://mozilladatacollective.com/datasets/{row.dataset_id}"
            handle.write(f"{row.language}\t{row.locale}\t{row.collection}\t{url}\n")
    print(f"\nneeds-terms list ({len(needs_terms)}) written to {NEEDS_TERMS_PATH}")
