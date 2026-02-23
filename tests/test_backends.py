"""Tests for backend phone mapping."""

from __future__ import annotations

from gopt_bench.backends.ctc_gop_original import ARPABET_TO_IDX, ARPABET_VOCAB
from gopt_bench.backends.xlsr_espeak import ARPABET_TO_IPA as XLSR_ARPABET_TO_IPA
from gopt_bench.backends.zipa import ARPABET_TO_IPA as ZIPA_ARPABET_TO_IPA


class TestOriginalBackendVocab:
    def test_vocab_size(self):
        assert len(ARPABET_VOCAB) == 40

    def test_blank_is_first(self):
        assert ARPABET_VOCAB[0] == "<pad>"

    def test_all_phones_mapped(self):
        """Every phone in vocab (except pad) has a mapping."""
        for phone in ARPABET_VOCAB[1:]:
            assert phone in ARPABET_TO_IDX

    def test_idx_roundtrip(self):
        """Index lookup should round-trip correctly."""
        for phone, idx in ARPABET_TO_IDX.items():
            assert ARPABET_VOCAB[idx] == phone


class TestXLSREspeakMapping:
    def test_all_arpabet_mapped(self):
        # Every ARPABET phone in the original vocab should have a mapping
        for phone in ARPABET_VOCAB[1:]:
            assert phone in XLSR_ARPABET_TO_IPA, f"{phone} missing"

    def test_ipa_values_nonempty(self):
        for phone, ipa in XLSR_ARPABET_TO_IPA.items():
            assert len(ipa) > 0, f"{phone} maps to empty IPA"


class TestZIPAMapping:
    def test_all_arpabet_mapped(self):
        for phone in ARPABET_VOCAB[1:]:
            assert phone in ZIPA_ARPABET_TO_IPA, f"{phone} missing"
