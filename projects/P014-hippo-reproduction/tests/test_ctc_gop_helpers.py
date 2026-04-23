from p014.features.ctc_gop import _canonical_phone_ids, _effective_vocab


def test_effective_vocab_filters_tokenizer_entries_beyond_model_head() -> None:
    raw_vocab = {"<pad>": 0, "aa": 1, "ae": 2, "<s>": 3, "</s>": 4}

    filtered = _effective_vocab(raw_vocab, model_vocab_size=3)

    assert filtered == {"<pad>": 0, "aa": 1, "ae": 2}


def test_canonical_phone_ids_supports_lowercase_and_unk_fallback() -> None:
    vocab = {"<pad>": 0, "aa": 1, "ae": 2, "<unk>": 3}

    ids = _canonical_phone_ids(["AA1", "ZZ0"], vocab)

    assert ids == [1, 3]
