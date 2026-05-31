"""Ingest Georgian ASR sources into the curator store (the data/ layout).

  georgian-ingest fleurs        # google/fleurs ka_ge -> canonical_audio/ + curator.sqlite
  georgian-ingest commonvoice   # needs COMMONVOICE_KA_DIR (extracted Common Voice ka, from Mozilla)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from georgian_asr import LANGUAGE

if TYPE_CHECKING:
    from omni_curator.sample import Sample

_ROOT = Path(__file__).resolve().parents[2]
DATA = _ROOT / "data"
RAW = DATA / "raw"
CANONICAL = DATA / "canonical_audio"
DB = DATA / "curator.sqlite"

_BATCH = 200


def _load_root_env() -> None:
    """Load KEY=VALUE lines from the monorepo-root .env into the environment (HF_TOKEN, ...)."""
    env_path = Path(__file__).resolve().parents[4] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ingest_fleurs() -> int:
    # FLEURS is already 16 kHz mono -> write canonical FLAC straight into canonical_audio/.
    os.environ.setdefault("HF_HOME", str(RAW / "hf-cache"))  # transient HF cache lands in raw/
    from omni_curator.ingest.huggingface import load_fleurs
    from omni_curator.store import CuratorStore

    store = CuratorStore(DB)
    count = 0
    batch: list[Sample] = []
    for sample in load_fleurs(
        "ka_ge", language=LANGUAGE, audio_dir=CANONICAL / "fleurs", streaming=False
    ):
        batch.append(sample)
        count += 1
        if len(batch) >= _BATCH:
            store.upsert(batch)
            batch = []
    if batch:
        store.upsert(batch)
    store.close()
    return count


def _ingest_commonvoice() -> int:
    cv_dir = os.environ.get("COMMONVOICE_KA_DIR")
    if not cv_dir:
        msg = (
            "set COMMONVOICE_KA_DIR to an extracted Common Voice ka directory "
            "(download the tarball direct from commonvoice.mozilla.org)"
        )
        raise SystemExit(msg)
    from omni_curator.ingest.commonvoice import load_commonvoice
    from omni_curator.process import resample_sample
    from omni_curator.store import CuratorStore

    store = CuratorStore(DB)
    count = 0
    batch: list[Sample] = []
    for sample in load_commonvoice(Path(cv_dir), language=LANGUAGE):
        batch.append(resample_sample(sample, CANONICAL / "commonvoice"))  # mp3 48k -> 16k FLAC
        count += 1
        if len(batch) >= _BATCH:
            store.upsert(batch)
            batch = []
    if batch:
        store.upsert(batch)
    store.close()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a Georgian source into the curator store.")
    parser.add_argument("source", choices=("fleurs", "commonvoice"))
    args = parser.parse_args(argv)
    _load_root_env()

    count = _ingest_fleurs() if args.source == "fleurs" else _ingest_commonvoice()

    from omni_curator.store import CuratorStore

    store = CuratorStore(DB)
    print(f"ingested {count} {args.source} samples -> {DB}")
    print(f"store now: {store.counts()}  ({store.hours():.2f} h)")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
