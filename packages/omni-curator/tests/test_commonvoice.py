from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from omni_curator.ingest.commonvoice import load_commonvoice_archive


def _wav_bytes(frequency: float) -> bytes:
    samples = np.sin(2 * np.pi * frequency * np.arange(8_000) / 8_000).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 8_000, format="WAV")
    return buffer.getvalue()


def _member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _archive(path, clips):
    header = "client_id\tpath\tsentence_id\tsentence\tup_votes\tdown_votes\n"
    rows = "".join(
        f"speaker-{index}\t{name}\tsentence-{index}\t{text}\t2\t0\n"
        for index, (name, text, _payload) in enumerate(clips)
    )
    with tarfile.open(path, "w:gz") as tar:
        _member(tar, "cv-corpus/en/train.tsv", (header + rows).encode())
        for name, _text, payload in clips:
            _member(tar, f"cv-corpus/en/clips/{name}", payload)


def _spontaneous_archive(path, rows):
    header = (
        "client_id\taudio_id\taudio_file\tduration_ms\tprompt_id\tprompt\ttranscription\t"
        "votes\tage\tgender\taccents\tvariant\tlanguage\tprompt_upvotes\tprompt_reports\t"
        "is_edited\tsplit\tchar_per_sec\tquality_tags\n"
    )
    body = "".join(
        f"speaker-{index}\t{index}\t{name}\t500\t{index}\tPrompt?\t{text}\t1\t\t\t\t\t"
        f"English\t0\t0\t0\t{split}\t10\t{quality_tags}\n"
        for index, (name, text, split, quality_tags, _payload) in enumerate(rows)
    )
    with tarfile.open(path, "w:gz") as tar:
        _member(tar, "sps/en/ss-corpus-en.tsv", (header + body).encode())
        for name, _text, _split, _quality_tags, payload in rows:
            _member(tar, f"sps/en/audios/{name}", payload)


def test_archive_ingest_streams_train_and_excludes_benchmark_identities(tmp_path):
    keep = _wav_bytes(220)
    by_clip = _wav_bytes(330)
    by_audio = _wav_bytes(440)
    archive = tmp_path / "cv.tar.gz"
    _archive(
        archive,
        [
            ("keep.mp3", "Keep this sentence.", keep),
            ("clip-excluded.mp3", "Exclude by clip.", by_clip),
            ("hash-excluded.mp3", "Exclude by hash.", by_audio),
        ],
    )

    samples = list(
        load_commonvoice_archive(
            archive,
            language="eng_Latn",
            source="common-voice-26-english",
            audio_dir=tmp_path / "audio",
            validation_fraction=0,
            excluded_clip_ids=frozenset({"clip-excluded.mp3"}),
            excluded_audio_sha256=frozenset({hashlib.sha256(by_audio).hexdigest()}),
        )
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.text == "Keep this sentence."
    assert sample.split == "train"
    assert sample.meta["clip_id"] == "keep.mp3"
    assert sample.meta["encoded_audio_sha256"] == hashlib.sha256(keep).hexdigest()
    info = sf.info(sample.audio_path)
    assert info.samplerate == 16_000
    assert info.channels == 1


def test_archive_ingest_reuses_valid_flac_on_resume(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    _archive(archive, [("keep.mp3", "Keep this sentence.", _wav_bytes(220))])
    kwargs = {
        "language": "eng_Latn",
        "source": "common-voice-26-english",
        "audio_dir": tmp_path / "audio",
        "validation_fraction": 0,
    }

    first = next(iter(load_commonvoice_archive(archive, **kwargs)))
    output = Path(first.audio_path)
    first_mtime = output.stat().st_mtime_ns
    second = next(iter(load_commonvoice_archive(archive, **kwargs)))

    assert second.id == first.id
    assert output.stat().st_mtime_ns == first_mtime


def test_archive_ingest_treats_prompt_quotes_as_literal_text(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    _archive(
        archive,
        [
            ("open-quote.mp3", 'She said "hello.', _wav_bytes(220)),
            ("balanced-quote.mp3", 'He replied "goodbye".', _wav_bytes(330)),
            ("plain.mp3", "This must stay a separate row.", _wav_bytes(440)),
        ],
    )

    samples = list(
        load_commonvoice_archive(
            archive,
            language="eng_Latn",
            source="common-voice-26-english",
            audio_dir=tmp_path / "audio",
            validation_fraction=0,
        )
    )

    assert [sample.text for sample in samples] == [
        'She said "hello.',
        'He replied "goodbye".',
        "This must stay a separate row.",
    ]


def test_archive_ingest_supports_spontaneous_train_metadata(tmp_path):
    archive = tmp_path / "spontaneous.tar.gz"
    _spontaneous_archive(
        archive,
        [
            ("train.mp3", "I [disfluency] agree.", "train", "", _wav_bytes(220)),
            ("dev.mp3", "Keep this held out.", "dev", "short-audio", _wav_bytes(330)),
        ],
    )

    samples = list(
        load_commonvoice_archive(
            archive,
            language="eng_Latn",
            source="common-voice-spontaneous-4-english",
            audio_dir=tmp_path / "audio",
            validation_fraction=0,
        )
    )

    assert len(samples) == 1
    assert samples[0].text == "I [disfluency] agree."
    assert samples[0].meta["audio_id"] == "0"
    assert samples[0].meta["quality_tags"] == ""
    assert samples[0].meta["upstream_split"] == "train"


def test_archive_ingest_requires_target_split_metadata(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _member(tar, "cv-corpus/en/README.txt", b"metadata unavailable")

    with pytest.raises(ValueError, match=r"no train\.tsv or spontaneous metadata TSV"):
        list(
            load_commonvoice_archive(
                archive,
                language="eng_Latn",
                source="common-voice-26-english",
                audio_dir=tmp_path / "audio",
            )
        )


def test_archive_ingest_fails_before_writing_when_clips_precede_tsv(tmp_path):
    archive = tmp_path / "cv.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _member(tar, "cv-corpus/en/clips/keep.mp3", _wav_bytes(220))

    with pytest.raises(ValueError, match=r"audio before train\.tsv or spontaneous metadata"):
        list(
            load_commonvoice_archive(
                archive,
                language="eng_Latn",
                source="common-voice-26-english",
                audio_dir=tmp_path / "audio",
            )
        )
    assert not (tmp_path / "audio").exists()
