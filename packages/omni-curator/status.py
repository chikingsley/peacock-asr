# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""omni-curator pipeline status board — the single source of truth, derived from the DBs.

Scans every projects/<lang>-asr, reads queue.sqlite + curator.sqlite + datasets/, and shows
which pipeline steps are done vs pending per project. Never hand-maintained, never stale.

Pipeline: download -> enqueue -> segment -> labelq -> harvest -> merge -> ingest -> verify -> export

Run:   uv run tools/status.py          (prints + writes STATUS.md)
       uv run tools/status.py --md     (write STATUS.md only, no stdout)
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"


def _q(db: Path, sql: str):
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
    rows = _q(db, f"SELECT status, count(*) n FROM {table} GROUP BY status")
    return {r["status"]: r["n"] for r in rows} if rows else {}


def _k(n: float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def scan(proj: Path) -> dict:
    data = proj / "data"
    qdb, cdb = data / "queue.sqlite", data / "curator.sqlite"
    videos, clips = _counts(qdb, "videos"), _counts(qdb, "clips")
    h = _q(qdb, "SELECT count(*) n FROM clips WHERE harvested_at IS NOT NULL")
    harvested = h[0]["n"] if h else 0
    src = _q(cdb, "SELECT source, count(*) n, count(scribe_wer) scored, "
                  "coalesce(sum(duration),0) dur FROM samples GROUP BY source")
    sources = {r["source"]: dict(n=r["n"], scored=r["scored"], dur=r["dur"]) for r in src} if src else {}
    create = data / "create"
    channels = sum(1 for d in create.iterdir() if d.is_dir()) if create.exists() else 0
    ds = data / "datasets"
    exports = sorted(d.name for d in ds.iterdir()
                     if d.is_dir() and ((d / "export_summary.json").exists() or (d / "version=0").exists())) if ds.exists() else []
    return dict(name=proj.name.replace("-asr", ""), videos=videos, clips=clips, harvested=harvested,
                sources=sources, channels=channels, exports=exports,
                has_q=qdb.exists(), has_c=cdb.exists())


def next_step(s: dict) -> str:
    v, c = s["videos"], s["clips"]
    enq = sum(v.values())
    seg_pending = v.get("pending", 0) + v.get("segmenting", 0)
    lab_pending = c.get("pending", 0) + c.get("labeling", 0)
    done_clips = c.get("done", 0)
    unharvested = max(0, done_clips - s["harvested"])
    yt = {k: x for k, x in s["sources"].items() if k.startswith("youtube-")}
    ingest = {k: x for k, x in s["sources"].items() if not k.startswith("youtube-")}
    total = sum(x["n"] for x in s["sources"].values())
    scored = sum(x["scored"] for x in s["sources"].values())
    if s["channels"] and enq == 0:
        return f"ENQUEUE ({s['channels']} channels downloaded, none queued)"
    if seg_pending:
        return f"SEGMENT ({_k(seg_pending)} videos pending)"
    if lab_pending:
        return f"LABELQ ({_k(lab_pending)} clips pending Scribe)"
    if unharvested:
        return f"HARVEST ({_k(unharvested)} labeled clips to fold)"
    if yt and not any(k.startswith("youtube-") for k in s["sources"]):
        return "MERGE (channel stores -> curator)"
    if total and scored < total:
        return f"VERIFY ({_k(total - scored)} samples unscored)"
    if total and not s["exports"]:
        return "EXPORT (no dataset built yet)"
    if total:
        return "— up to date"
    return "— no data yet"


def render(rows: list[dict]) -> str:
    L = [f"# omni-curator pipeline status   ({datetime.now():%Y-%m-%d %H:%M})",
         "",
         "pipeline: download → enqueue → segment → labelq(Scribe) → harvest → merge → ingest → verify → export",
         ""]
    for s in rows:
        v, c = s["videos"], s["clips"]
        total = sum(x["n"] for x in s["sources"].values())
        scored = sum(x["scored"] for x in s["sources"].values())
        hours = sum(x["dur"] for x in s["sources"].values()) / 3600
        yt = {k: x for k, x in s["sources"].items() if k.startswith("youtube-")}
        ing = {k: x for k, x in s["sources"].items() if not k.startswith("youtube-")}
        L.append(f"## {s['name'].upper()}    →  NEXT: {next_step(s)}")
        if s["has_q"]:
            vparts = " ".join(f"{k}={_k(n)}" for k, n in sorted(v.items())) or "(empty)"
            cparts = " ".join(f"{k}={_k(n)}" for k, n in sorted(c.items())) or "(empty)"
            L.append(f"  queue.videos : {vparts}")
            L.append(f"  queue.clips  : {cparts}  harvested={_k(s['harvested'])}")
        else:
            L.append("  queue        : none (no YouTube pipeline staged)")
        if s["has_c"]:
            hrs = f"{hours:,.0f}h" if hours < 1_000_000 else "?h (corrupt durations)"
            L.append(f"  curator      : {_k(total)} samples ({hrs})  scored={_k(scored)}/{_k(total)}")
            if yt:
                L.append(f"     youtube   : {len(yt)} channels, {_k(sum(x['n'] for x in yt.values()))} samples")
            if ing:
                top = sorted(ing.items(), key=lambda kv: -kv[1]["n"])[:5]
                L.append(f"     datasets  : {len(ing)} sources — " + ", ".join(f"{k}={_k(x['n'])}" for k, x in top))
        else:
            L.append("  curator      : none")
        if s["exports"]:
            L.append(f"  exports      : {', '.join(s['exports'])}")
        L.append(f"  downloads    : {s['channels']} channels in create/")
        L.append("")
    return "\n".join(L)


def main() -> int:
    projs = sorted(p for p in PROJECTS.iterdir() if p.is_dir() and p.name.endswith("-asr"))
    rows = [scan(p) for p in projs]
    out = render(rows)
    (ROOT / "STATUS.md").write_text(out + "\n", encoding="utf-8")
    if "--md" not in sys.argv:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
