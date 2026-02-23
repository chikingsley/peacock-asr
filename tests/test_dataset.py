"""Tests for the dataset module."""

from __future__ import annotations

from gopt_bench.dataset import strip_stress


class TestStripStress:
    def test_removes_stress_digit(self):
        assert strip_stress("AA0") == "AA"
        assert strip_stress("AA1") == "AA"
        assert strip_stress("AA2") == "AA"

    def test_plain_phone_unchanged(self):
        assert strip_stress("AA") == "AA"
        assert strip_stress("B") == "B"
        assert strip_stress("SH") == "SH"

    def test_special_tokens_unchanged(self):
        assert strip_stress("<unk>") == "<unk>"
        assert strip_stress("SIL") == "SIL"

    def test_lowercase_passthrough(self):
        assert strip_stress("hello") == "hello"
