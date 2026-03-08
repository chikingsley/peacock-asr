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

    def unload(self) -> None:
        """Release any large runtime state after posterior collection."""
        ...

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute frame-level posteriors from audio."""
        ...

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        """Map an ARPABET phone to this backend's vocab indices."""
        ...
