"""The standardized record that flows through the whole curator.

Every source (ingest/* and create/*) produces ``Sample``s; ``process`` normalizes/validates them;
``store`` persists them in SQLite; ``benchmark`` scores them. One shape for everything, so we
never re-invent a per-dataset record again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from omni_curator.data.provenance import SourceProvenance


@dataclass
class Sample:
    """One labelled utterance: an (audio, text) pair plus provenance."""

    id: str
    source: str  # 'fleurs', 'commonvoice', 'youtube', ...
    language: str  # the curator's language code, e.g. 'kat_Geor', 'tgk_Cyrl'
    text: str  # the transcript / label
    audio_path: str  # path to the (ideally 16 kHz mono) audio file
    duration: float  # seconds
    sample_rate: int
    split: str = "train"  # train | dev | test
    speaker_id: str | None = None
    citation: str | None = None  # where it came from (URL / dataset id) — credit the source
    # Store-level Scribe verification scores: the label scored against a fresh Scribe pass
    # (lower is cleaner). ``None`` until ``omni_curator.audit.verify.verify_store`` scores it;
    # the full jiwer detail + the Scribe hypothesis live in ``meta["scribe"]``.
    scribe_wer: float | None = None
    scribe_cer: float | None = None
    meta: dict[str, object] = field(default_factory=dict)  # source-specific extras

    @property
    def provenance(self) -> SourceProvenance | None:
        """Typed provenance decoded from ``meta['provenance']`` when present."""
        return SourceProvenance.from_meta(self.meta)

    def with_provenance(self, provenance: SourceProvenance) -> Sample:
        """Return a copy with typed provenance serialized into ``meta``."""
        meta = dict(self.meta)
        meta["provenance"] = provenance.to_meta()
        return replace(self, meta=meta)
