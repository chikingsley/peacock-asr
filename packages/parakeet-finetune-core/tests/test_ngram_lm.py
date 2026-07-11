from __future__ import annotations

from parakeet_finetune_core.ngram_lm import encode_token, write_token_corpus


class FakeTokenizer:
    def encode(self, text, *, out_type):
        assert out_type is int
        return {"hello world": [0, 2], "goodbye": [1]}[text]


def test_encode_token_uses_nemo_offset():
    assert encode_token(0) == "d"
    assert encode_token(2) == "f"


def test_write_token_corpus_normalizes_lines(tmp_path):
    corpus = tmp_path / "corpus.txt"
    output = tmp_path / "tokens.txt"
    corpus.write_text(" hello   world \n\ngoodbye\n", encoding="utf-8")

    counts = write_token_corpus(corpus, output, FakeTokenizer())

    assert counts == (2, 3)
    assert output.read_text(encoding="utf-8") == "d f\ne\n"
