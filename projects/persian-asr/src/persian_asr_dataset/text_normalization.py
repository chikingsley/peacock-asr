"""Persian (Farsi) text normalization for ASR.

`maybe_normalize` is adapted from the NVIDIA model card normalization script
("The transcripts were additionally normalized according to the following
script (empty results were discarded)."):

  Source:   https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large
  File:     README.md
  Revision: 249cf5bf70dda7220a60ddeeecff2f6aad8e1784

Refresh the upstream reference with:

  hf download nvidia/stt_fa_fastconformer_hybrid_large README.md \
    --revision 249cf5bf70dda7220a60ddeeecff2f6aad8e1784 \
    --local-dir /tmp/nvidia-stt-fa-fastconformer

Local additions on top of the upstream script (see
docs/zwnj-normalization-decision-20260529.md): all Unicode format characters
(ZWNJ/ZWJ/bidi marks/soft-hyphen/BOM) are mapped to a space so the Omni char
tokenizer never sees an unrepresentable character; a few tokenizer-incompatible
symbols are discarded; and ARABIC LETTER ALEF WASLA folds to a plain ALEF.
"""

from __future__ import annotations

import string
import unicodedata

SOURCE_REPO = "nvidia/stt_fa_fastconformer_hybrid_large"
SOURCE_FILE = "README.md"
SOURCE_REVISION = "249cf5bf70dda7220a60ddeeecff2f6aad8e1784"
SOURCE_README_SHA256 = "f98ae540031ed90105b887ad3529f412a17ecfd452a5341d904fb4733913ce7e"

SKIP = set(
    list(string.ascii_letters)
    + [
        "=",  # occurs only 2x in utterance (transl.): "twenty = xx"
        "ā",  # occurs only 4x together with "š"
        "š",
        # Arabic letters
        "ة",
    ]
)

DISCARD = [
    # "(laughter)" in Farsi
    "(خنده)",
    # ASCII
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    # Unicode punctuation?
    "–",
    "“",
    "”",
    "…",
    "؟",
    "،",
    "؛",
    "ـ",
    # Unicode whitespace?
    "ً",
    "ٌ",
    "َ",
    "ُ",
    "ِ",
    "ّ",
    "ْ",
    "ٔ",
    # Other
    "«",
    "»",
    # Tokenizer-incompatible symbols (no Omni piece) -> removed.
    "٪",  # ARABIC PERCENT SIGN
    "×",  # MULTIPLICATION SIGN
    "÷",  # DIVISION SIGN
    "�",  # REPLACEMENT CHARACTER
]

REPLACEMENTS = {
    "أ": "ا",
    "ۀ": "ە",
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ﯽ": "ی",
    "ﻮ": "و",
    "ے": "ی",
    "ﺒ": "ب",
    "ﻢ": "ﻡ",
    "٬": " ",
    "ە": "ه",
    # Arabic letter variant the Omni tokenizer lacks -> canonical Persian form.
    "ٱ": "ا",  # ARABIC LETTER ALEF WASLA -> ALEF
}


def maybe_normalize(text: str) -> str | None:
    # Skip selected with banned characters
    if set(text) & SKIP:
        return None  # skip this

    # Remove hashtags - they are not being read in Farsi CV
    text = " ".join(w for w in text.split() if not w.startswith("#"))

    # Replace selected characters with others
    for lhs, rhs in REPLACEMENTS.items():
        text = text.replace(lhs, rhs)

    # Replace selected characters with empty strings
    for tok in DISCARD:
        text = text.replace(tok, "")

    # Drop Unicode format/control chars (Cf: ZWNJ, ZWJ, bidi marks/embeddings/
    # isolates, soft hyphen, BOM) and other symbols (So: emoji, musical notes ♪♫,
    # misc symbols). The Omni char tokenizer has no piece for these, so they
    # would encode to <unk> ("⁇"); none are speech, so map to a space and let the
    # final whitespace-collapse merge it. See docs/zwnj-normalization-decision-20260529.md.
    text = "".join(" " if unicodedata.category(ch) in {"Cf", "So"} else ch for ch in text)

    # Unify the symbols that have the same meaning but different Unicode representation.
    text = unicodedata.normalize("NFKC", text)

    # Remove hamza's that were not merged with any letter by NFKC.
    text = text.replace("ء", "")

    # Remove double whitespace etc.
    return " ".join(t for t in text.split() if t)
