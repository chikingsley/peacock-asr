"""Mixture summary (``language_distribution_0.tsv``) for fairseq2's MIXTURE_PARQUET sampler.

fairseq2 weights each corpus by the hours listed in this TSV (tempered by ``beta_corpus``).
Hours are the total audio per ``(corpus, language)`` across all splits, derived from the
``audio_size`` column (samples / 16000 / 3600).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

SAMPLE_RATE = 16_000


def corpus_hours(version_root: Path) -> dict[tuple[str, str], float]:
    """Map ``(corpus, language) -> hours`` over the whole ``version=0`` tree."""
    samples: dict[tuple[str, str], int] = {}
    for path in sorted(version_root.glob("corpus=*/split=*/language=*/*.parquet")):
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
        key = (parts["corpus"], parts["language"])
        sizes = pq.read_table(path, columns=["audio_size"]).column("audio_size").to_pylist()
        samples[key] = samples.get(key, 0) + sum(sizes)
    return {key: n / SAMPLE_RATE / 3600 for key, n in samples.items()}


def write_language_distribution(version_root: Path, out_path: Path) -> Path:
    """Write ``corpus<TAB>language<TAB>hours`` for every corpus under ``version_root``."""
    hours = corpus_hours(version_root)
    lines = ["corpus\tlanguage\thours"]
    lines.extend(
        f"{corpus}\t{language}\t{hours[corpus, language]:.8f}"
        for corpus, language in sorted(hours)
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
