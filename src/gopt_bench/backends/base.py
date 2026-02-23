from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class PhonemeBackend(Protocol):
    """Protocol that all phoneme model backends must implement."""

    @property
    def name(self) -> str: ...

    @property
    def vocab(self) -> list[str]:
        """Return the full phoneme vocabulary (index 0 = blank/pad)."""
        ...

    @property
    def blank_index(self) -> int:
        """Index of the CTC blank token."""
        ...

    def load(self) -> None:
        """Download and load the model into memory."""
        ...

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute frame-level posteriors from audio.

        Args:
            audio: 1-D float32 waveform
            sample_rate: sampling rate of the audio

        Returns:
            Posterior matrix of shape [T, V] where T is frames
            and V is vocab size. Each row should sum to ~1.0 (softmax output).
        """
        ...

    def map_phone(self, arpabet_phone: str) -> int | None:
        """Map an ARPABET phone to this backend's vocab index.

        Returns None if the phone has no mapping.
        """
        ...
