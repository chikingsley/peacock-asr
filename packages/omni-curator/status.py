# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""omni-curator pipeline status board — the single source of truth, derived from the DBs.

Scans every projects/<lang>-asr, reads queue.sqlite + curator.sqlite + datasets/, and shows
which pipeline steps are done vs pending per project. Never hand-maintained, never stale.

Pipeline: download -> enqueue -> segment -> labelq -> harvest -> merge -> ingest -> verify -> export

Run:   uv run --project packages/omni-curator packages/omni-curator/status.py
Add `--output scratch/status.md` to also write the board to a file.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / "projects"
THOUSAND = 1_000
MILLION = 1_000_000
STATUS_COUNT_QUERIES = {
    "videos": "SELECT status, count(*) n FROM videos GROUP BY status",
    "clips": "SELECT status, count(*) n FROM clips GROUP BY status",
}


def _q(db: Path, sql: str) -> list[sqlite3.Row] | None:
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None


def _counts(db: Path, table: str) -> dict[str, int]:
    rows = _q(db, STATUS_COUNT_QUERIES[table])
    return {r["status"]: r["n"] for r in rows} if rows else {}


def _k(n: float) -> str:
    n = int(n)
    if n >= MILLION:
        return f"{n / MILLION:.1f}M"
    if n >= THOUSAND:
        return f"{n / THOUSAND:.1f}k"
    return str(n)


def scan(proj: Path) -> dict:
    data = proj / "data"
    qdb, cdb = data / "queue.sqlite", data / "curator.sqlite"
    videos, clips = _counts(qdb, "videos"), _counts(qdb, "clips")
    h = _q(qdb, "SELECT count(*) n FROM clips WHERE harvested_at IS NOT NULL")
    harvested = h[0]["n"] if h else 0
    src = _q(
        cdb,
        "SELECT source, count(*) n, count(scribe_wer) scored, "
        "coalesce(sum(duration),0) dur FROM samples GROUP BY source",
    )
    sources = (
        {r["source"]: {"n": r["n"], "scored": r["scored"], "dur": r["dur"]} for r in src}
        if src
        else {}
    )
    create = data / "create"
    channels = sum(1 for d in create.iterdir() if d.is_dir()) if create.exists() else 0
    ds = data / "datasets"
    exports = (
        sorted(
            d.name
            for d in ds.iterdir()
            if d.is_dir() and ((d / "export_summary.json").exists() or (d / "version=0").exists())
        )
        if ds.exists()
        else []
    )
    return {
        "name": proj.name.replace("-asr", ""),
        "videos": videos,
        "clips": clips,
        "harvested": harvested,
        "sources": sources,
        "channels": channels,
        "exports": exports,
        "has_q": qdb.exists(),
        "has_c": cdb.exists(),
    }


def next_step(s: dict) -> str:
    v, c = s["videos"], s["clips"]
    enq = sum(v.values())
    seg_pending = v.get("pending", 0) + v.get("segmenting", 0)
    lab_pending = c.get("pending", 0) + c.get("labeling", 0)
    done_clips = c.get("done", 0)
    unharvested = max(0, done_clips - s["harvested"])
    yt = {k: x for k, x in s["sources"].items() if k.startswith("youtube-")}
    total = sum(x["n"] for x in s["sources"].values())
    scored = sum(x["scored"] for x in s["sources"].values())
    steps = (
        (s["channels"] and enq == 0, f"ENQUEUE ({s['channels']} channels downloaded, none queued)"),
        (seg_pending, f"SEGMENT ({_k(seg_pending)} videos pending)"),
        (lab_pending, f"LABELQ ({_k(lab_pending)} clips pending Scribe)"),
        (unharvested, f"HARVEST ({_k(unharvested)} labeled clips to fold)"),
        (
            yt and not any(k.startswith("youtube-") for k in s["sources"]),
            "MERGE (channel stores -> curator)",
        ),
        (total and scored < total, f"VERIFY ({_k(total - scored)} samples unscored)"),
        (total and not s["exports"], "EXPORT (no dataset built yet)"),
        (total, "— up to date"),
    )
    return next((message for pending, message in steps if pending), "— no data yet")


def render(rows: list[dict]) -> str:
    lines = [
        f"# omni-curator pipeline status   ({datetime.now(UTC).astimezone():%Y-%m-%d %H:%M})",
        "",
        (
            "pipeline: download → enqueue → segment → labelq(Scribe) → harvest → merge → ingest → "
            "verify → export"
        ),
        "",
    ]
    for s in rows:
        v, c = s["videos"], s["clips"]
        total = sum(x["n"] for x in s["sources"].values())
        scored = sum(x["scored"] for x in s["sources"].values())
        hours = sum(x["dur"] for x in s["sources"].values()) / 3600
        yt = {k: x for k, x in s["sources"].items() if k.startswith("youtube-")}
        ing = {k: x for k, x in s["sources"].items() if not k.startswith("youtube-")}
        lines.append(f"## {s['name'].upper()}    →  NEXT: {next_step(s)}")
        if s["has_q"]:
            vparts = " ".join(f"{k}={_k(n)}" for k, n in sorted(v.items())) or "(empty)"
            cparts = " ".join(f"{k}={_k(n)}" for k, n in sorted(c.items())) or "(empty)"
            lines.append(f"  queue.videos : {vparts}")
            lines.append(f"  queue.clips  : {cparts}  harvested={_k(s['harvested'])}")
        else:
            lines.append("  queue        : none (no YouTube pipeline staged)")
        if s["has_c"]:
            hrs = f"{hours:,.0f}h" if hours < MILLION else "?h (corrupt durations)"
            lines.append(
                f"  curator      : {_k(total)} samples ({hrs})  scored={_k(scored)}/{_k(total)}"
            )
            if yt:
                youtube_samples = _k(sum(x["n"] for x in yt.values()))
                lines.append(f"     youtube   : {len(yt)} channels, {youtube_samples} samples")
            if ing:
                top = sorted(ing.items(), key=lambda kv: -kv[1]["n"])[:5]
                lines.append(
                    f"     datasets  : {len(ing)} sources — "
                    + ", ".join(f"{k}={_k(x['n'])}" for k, x in top)
                )
        else:
            lines.append("  curator      : none")
        if s["exports"]:
            lines.append(f"  exports      : {', '.join(s['exports'])}")
        lines.append(f"  downloads    : {s['channels']} channels in create/")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the omni-curator pipeline status board.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown output path. By default, status is printed only.",
    )
    parser.add_argument(
        "--write-root-status",
        action="store_true",
        help="Compatibility escape hatch: write the ignored root STATUS.md snapshot.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print status to stdout.")
    parser.add_argument("--md", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    projs = sorted(p for p in PROJECTS.iterdir() if p.is_dir() and p.name.endswith("-asr"))
    rows = [scan(p) for p in projs]
    out = render(rows)
    output = ROOT / "STATUS.md" if args.write_root_status else args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(out + "\n", encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
