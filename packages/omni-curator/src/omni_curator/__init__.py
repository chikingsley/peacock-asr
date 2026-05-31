"""omni-curator: turn raw audio into ASR fine-tuning transcripts.

Pipeline: segment -> Scribe ensemble -> compile-down -> (stitch) -> polish. Two finalized paths
in :mod:`omni_curator.pipeline` (``vad_path`` and ``chunks_path``); building blocks in
:mod:`omni_curator.segmenters`, :mod:`omni_curator.transcribe`, and :mod:`omni_curator.fuse`.
"""

from __future__ import annotations

from omni_curator.pipeline import Clip, Transcript, chunks_path, vad_path

__all__ = ["Clip", "Transcript", "chunks_path", "vad_path"]
