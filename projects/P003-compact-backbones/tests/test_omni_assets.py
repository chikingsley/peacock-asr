from __future__ import annotations

from p003_compact.omni_assets import (
    PHONEME_MODEL_NAME,
    PHONEME_TOKENIZER_NAME,
    load_ordered_vocab,
)


def test_omni_asset_names_are_stable() -> None:
    assert PHONEME_MODEL_NAME == "omniASR_CTC_300M_v2_arpabet_41"
    assert PHONEME_TOKENIZER_NAME == "omniASR_tokenizer_arpabet_41_v1"


def test_ordered_vocab_keeps_pad_and_unk_last() -> None:
    tokens = load_ordered_vocab()
    assert tokens[-2:] == ["[UNK]", "[PAD]"]
    assert len(tokens) == 41
