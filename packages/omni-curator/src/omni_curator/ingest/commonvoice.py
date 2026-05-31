"""Mozilla Common Voice -> ``Sample``. Pulled DIRECT from Mozilla, not the HF Hub.

Common Voice is downloaded as a per-language, per-version tarball from a signed URL that Mozilla
gates behind acceptance (put the URL in the env, e.g. ``COMMONVOICE_KA_URL``). After download +
extract, the language folder holds split ``.tsv`` files (``train``/``dev``/``test``/``validated``,
columns include ``client_id``, ``path``, ``sentence``) and a ``clips/`` dir of mp3s (typically
48 kHz — ``process`` resamples to 16 kHz). ``clip_durations.tsv`` gives per-clip duration.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Common Voice ships these; we keep train/dev/test (the disjoint official splits).
_CV_SPLITS = ("train", "dev", "test")


def _durations(cv_dir: Path) -> dict[str, float]:
    """Map clip filename -> seconds from clip_durations.tsv (duration is in ms), if present."""
    path = cv_dir / "clip_durations.tsv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        field = "duration[ms]" if "duration[ms]" in (reader.fieldnames or []) else "duration"
        return {row["clip"]: float(row[field]) / 1000.0 for row in reader if row.get(field)}


def load_commonvoice(
    cv_dir: Path,
    *,
    language: str,
    source: str = "commonvoice",
    splits: tuple[str, ...] = _CV_SPLITS,
) -> Iterator[Sample]:
    """Read an extracted Common Voice language dir -> ``Sample``s (clips stay mp3 until process)."""
    clips = cv_dir / "clips"
    durations = _durations(cv_dir)
    for split in splits:
        tsv = cv_dir / f"{split}.tsv"
        if not tsv.exists():
            continue
        with tsv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                clip = row.get("path")
                text = str(row.get("sentence") or "").strip()
                if not clip or not text:
                    continue
                yield Sample(
                    id=f"{source}_{language}_{split}_{clip}",
                    source=source,
                    language=language,
                    text=text,
                    audio_path=str(clips / clip),
                    duration=durations.get(clip, 0.0),
                    sample_rate=0,  # unknown until process probes/resamples the mp3
                    split=split,
                    speaker_id=row.get("client_id"),
                    citation="commonvoice.mozilla.org",
                    meta={"age": row.get("age"), "gender": row.get("gender")},
                )
