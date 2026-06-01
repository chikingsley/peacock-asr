"""The standardized record that flows through the whole curator.

Every source (ingest/* and create/*) produces ``Sample``s; ``process`` normalizes/validates them;
``store`` persists them in SQLite; ``benchmark`` scores them. One shape for everything, so we
never re-invent a per-dataset record again.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # (lower is cleaner). ``None`` until ``omni_curator.verify.verify_store`` has scored the clip;
    # the full jiwer detail + the Scribe hypothesis live in ``meta["scribe"]``.
    scribe_wer: float | None = None
    scribe_cer: float | None = None
    meta: dict[str, object] = field(default_factory=dict)  # source-specific extras
