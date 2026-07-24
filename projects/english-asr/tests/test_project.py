from __future__ import annotations

from pathlib import Path

from english_asr import DATA, LANGUAGE, ROOT
from english_asr.curate import PROJECT as CURATOR
from english_asr.parakeet import PROJECT as PARAKEET
from english_asr.sources import ACTIVE_HF_CORPORA, EXCLUDED_BENCHMARKS, LICENSES


def test_project_paths_and_language() -> None:
    assert ROOT.name == "english-asr"
    assert DATA == ROOT / "data"
    assert LANGUAGE == "eng_Latn"
    assert CURATOR.language == LANGUAGE
    assert PARAKEET.language == LANGUAGE


def test_active_sources_are_pinned_bounded_train_only() -> None:
    assert ACTIVE_HF_CORPORA
    for source in ACTIVE_HF_CORPORA:
        assert source.revision
        assert source.splits
        assert all(split == "train" or split.startswith("train.") for split in source.splits)
        assert source.max_hours is not None
        assert 0 < source.validation_fraction < 1
        assert source.split_group_column


def test_parakeet_preserves_embedded_tokenizer() -> None:
    assert PARAKEET.default_tokenizer_dir is None
    assert Path(PARAKEET.default_tdt_model).name == "parakeet-tdt_ctc-110m.nemo"
    assert PARAKEET.default_validation_manifest.name == "dev.jsonl"
    assert "earnings21" in EXCLUDED_BENCHMARKS


def test_common_voice_archive_source_is_registered() -> None:
    assert "common-voice-26-english" in CURATOR.ingests
    assert "common-voice-spontaneous-4-english" in CURATOR.ingests
    assert LICENSES["common-voice-26-english"] == ("CC0-1.0", True)
    assert LICENSES["common-voice-spontaneous-4-english"] == ("CC0-1.0", True)


def test_librispeech_replay_is_train_only_and_pinned() -> None:
    source = next(
        source
        for source in ACTIVE_HF_CORPORA
        if source.name == "librispeech-train-clean-100-replay"
    )
    assert source.splits == ("train.100",)
    assert source.split_group_column == "speaker_id"
    assert source.max_hours == 100.0
    assert LICENSES[source.name] == ("CC-BY-4.0", True)
