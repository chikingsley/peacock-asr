"""Vendored ZIPA inference helpers (fbank extraction + CTC greedy decode).

Adapted from the ZIPA reference inference code (https://github.com/lingjzhu/zipa,
`inference/utils.py`), MIT License, Copyright (c) 2025 jzhu — see `ZIPA_LICENSE.txt`.
Only the CTC path is kept (the transducer decoder and the subprocess CLI are dropped);
the recognizer in `zipa.py` calls these in-process. The full ZIPA training repo is not vendored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lhotse.features.kaldi.extractors import Fbank, FbankConfig


def load_tokens(token_file: str | Path) -> dict[int, str]:
    """Parse a `tokens.txt` (`<token> <id>` per line) into {id: token}."""
    tokens: dict[int, str] = {}
    with Path(token_file).open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = int(parts[1]) if len(parts) > 1 else len(tokens)
            tokens[idx] = parts[0]
    return tokens


def get_fbank_extractor() -> Fbank:
    """80-bin Kaldi fbank, matching ZIPA's training front-end (no dither, no edge snipping)."""
    return Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))


def ctc_greedy_decode(log_probs: np.ndarray, vocab: dict[int, str], blank_id: int = 0) -> list[str]:
    """Greedy CTC collapse of a (Time, Vocab) log-prob matrix to a phone list."""
    preds = np.argmax(log_probs, axis=-1)
    decoded: list[str] = []
    prev = -1
    for idx in preds:
        idx_int = int(idx)
        if idx_int not in (blank_id, prev):
            decoded.append(vocab.get(idx_int, ""))
        prev = idx_int
    return decoded
