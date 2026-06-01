"""create: generate labels for raw, untranscribed audio.

For sources whose audio has NO transcript (YouTube, podcasts, shows), this synthesizes the labels:
segment -> Scribe ensemble -> compile-down -> (stitch) -> polish. Two finalized paths,
``vad_path`` (dense speech) and ``chunks_path`` (sparse/drill audio). The counterpart is
:mod:`omni_curator.ingest`, which pulls datasets that ALREADY have transcripts.
"""

from __future__ import annotations

from omni_curator.create.align import align_to_clips
from omni_curator.create.pipeline import Clip, Transcript, chunks_path, vad_path
from omni_curator.create.run import label_to_store, label_youtube

__all__ = [
    "Clip",
    "Transcript",
    "align_to_clips",
    "chunks_path",
    "label_to_store",
    "label_youtube",
    "vad_path",
]
