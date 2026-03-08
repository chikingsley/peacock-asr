"""P001-local backend implementations."""

from p001_gop.backends.ctc_gop_original import OriginalBackend
from p001_gop.backends.xlsr_espeak import XLSREspeakBackend
from p001_gop.backends.zipa import ZIPABackend

__all__ = [
    "OriginalBackend",
    "XLSREspeakBackend",
    "ZIPABackend",
]
