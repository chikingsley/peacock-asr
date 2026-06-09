"""Export Selection contract: gates are train-only, held-out videos regroup leakage-safe.

Locks in two measurement-integrity rules learned the hard way:
- benchmark splits are never censored by curation gates (commit 082a04c1 — the v2 export's
  WER gate silently ate 20 FLEURS rows);
- held-out conversational test videos are gated like the train rows they are stored as,
  with survivors regrouped to test and the rest dropped, so no held-out video reaches train
  (commit a6b067e2).
"""

from __future__ import annotations

import pytest

from omni_curator.export import Selection, write_weighted_distribution
from omni_curator.quality import is_descriptor_only


def test_wer_gate_applies_to_train_only(make_sample):
    sel = Selection(max_scribe_wer=0.35)
    assert not sel.keeps(make_sample(split="train", scribe_wer=0.9))
    # The same terrible score on benchmark splits is KEPT — never censor the exam.
    assert sel.keeps(make_sample(split="dev", scribe_wer=0.9))
    assert sel.keeps(make_sample(split="test", scribe_wer=0.9))


def test_unscored_clip_is_never_silently_dropped(make_sample):
    sel = Selection(max_scribe_wer=0.35)
    assert sel.keeps(make_sample(split="train", scribe_wer=None))


def test_descriptor_filter_is_train_only(make_sample):
    sel = Selection()
    assert not sel.keeps(make_sample(split="train", text="[outro jingle]"))
    assert sel.keeps(make_sample(split="test", text="[outro jingle]"))


def test_duration_bound_applies_everywhere(make_sample):
    sel = Selection(max_duration_seconds=40.0)
    # Structural (model-imposed) bound: applies to benchmarks too.
    assert not sel.keeps(make_sample(split="test", duration=55.0))


def test_heldout_video_is_gated_then_regrouped(make_sample):
    sel = Selection(max_scribe_wer=0.35, heldout_test_videos=frozenset({"chan_vid001"}))
    good = make_sample(id="chan_vid001_0003", split="train", scribe_wer=0.1)
    bad = make_sample(id="chan_vid001_0004", split="train", scribe_wer=0.9)
    other = make_sample(id="chan_vid999_0000", split="train", scribe_wer=0.1)

    assert sel.is_heldout(good)
    assert sel.is_heldout(bad)
    assert not sel.is_heldout(other)
    # Held-out clips are still curation-gated (machine labels: a failing clip is dropped
    # entirely — never train, not test either)...
    assert sel.keeps(good)
    assert not sel.keeps(bad)
    # ...and the survivors are destined for split=test (regrouped in _normalize_and_filter).
    assert sel.gates(good)


def test_descriptor_only_cases():
    junk = ["[outro jingle]", "[музыка]", "♪", "...", "(background noise)", "[singing] ♪", ""]
    real = ["Салом [музыка]", "дар як намоиш буд", "The Barefoot Investor by Scott Pape."]
    for text in junk:
        assert is_descriptor_only(text), f"junk not flagged: {text!r}"
    for text in real:
        assert not is_descriptor_only(text), f"real label flagged as junk: {text!r}"


def test_write_weighted_distribution(tmp_path):
    true_tsv = tmp_path / "language_distribution_0.tsv"
    true_tsv.write_text(
        "corpus\tlanguage\thours\n"
        "fleurs\ttgk_Cyrl\t11.83351667\n"
        "youtube-chan\ttgk_Cyrl\t94.02941667\n",
        encoding="utf-8",
    )
    out = write_weighted_distribution(true_tsv, tmp_path / "weighted.tsv", {"fleurs": 490.0})
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "fleurs\ttgk_Cyrl\t490.00000000"  # overridden
    assert lines[2] == "youtube-chan\ttgk_Cyrl\t94.02941667"  # untouched

    with pytest.raises(ValueError, match="not in the export"):
        write_weighted_distribution(true_tsv, tmp_path / "bad.tsv", {"flerus": 490.0})  # typo
