"""On-disk pre-labelled corpora -> ``Sample``. Manifest parsers for the three local Russian sets.

The counterpart to the network ingesters (fleurs / huggingface / commonvoice): these read a corpus
that already lives on disk in its own native layout. Each ``load_*`` reads the corpus's manifest,
yields ``Sample``s pointing at the ORIGINAL audio (``sample_rate=0``), and leaves the 16 kHz mono
FLAC conversion to ``process.resample_samples`` — the same division of labour as
:func:`omni_curator.ingest.commonvoice.load_commonvoice`. :func:`local_corpus_source` wraps a
``load_*`` into an ``IngestFn`` that runs the resample pass into the project's canonical dir.

On-disk formats (verified 2026-06-15 against the real data under russian-asr/data/):

ru_open_stt (snakers4/open_stt, the ``ru_open_stt_opus`` distribution)
    Corpus root holds one ``<subset>.csv`` per extracted subset. Each CSV is headerless with three
    columns: ``audio_path,txt_path,duration_seconds`` (e.g.
    ``asr_calls_2_val/9/33/14bbb530d04d.wav,asr_calls_2_val/9/33/14bbb530d04d.txt,0.32``). Paths are
    relative to the corpus root; the transcript is a sibling UTF-8 ``.txt`` (one Cyrillic line).
    Audio is mono 16 kHz PCM ``.wav`` (some distributions ship ``.opus`` — ffmpeg reads either).
    LICENSE: CC-BY-NC (non-commercial) — wire behind an opt-in, not the default export.

sova_dataset (sova-tts/sova-dataset; RuDevices / RuAudiobooks)
    A tree of ``.wav`` each paired with a same-stem sibling ``.txt`` holding the transcript. Some
    releases also ship a ``manifest.{csv,tsv,jsonl}`` (path/audio + text/transcript columns); when
    present it is used, else we walk the tree and pair wav <-> .txt by stem. (Empty on disk
    as of 2026-06-15 — parser ready for when the archive is extracted.) LICENSE: CC-BY-4.0.

timit_ru (the local ``TIMIT/`` dir)
    Classic NIST/DARPA TIMIT layout: ``{TRAIN,TEST}/DR{1..8}/<SPKR>/<UTT>.{WAV,TXT,WRD,PHN}``. The
    ``.WAV`` is NIST SPHERE (ffmpeg decodes it); the ``.TXT`` is ``start end <sentence>`` — the two
    leading sample indices are stripped to recover the transcript. NOTE: this dir is the original
    ENGLISH DARPA TIMIT, not Russian, and is LDC-restricted; the russian-asr project quarantines it
    and does NOT ingest it. The parser is provided for completeness / non-Russian reuse.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from omni_curator.project import CuratorProject

_MIN_OPEN_STT_COLS = 2  # audio,txt[,duration]
_TIMIT_TXT_PARTS = 3  # "start end <sentence...>"
_DR_DEPTH = 1  # TIMIT path under the corpus root: <split>/DR*/<spkr>/<utt>


def _slug(value: str) -> str:
    """Filesystem path / id fragment -> a safe underscore-joined id token."""
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _read_text_file(path: Path) -> str:
    """Read a sibling transcript ``.txt`` (UTF-8, one line); empty if missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


# -- ru_open_stt --------------------------------------------------------------------------------


def load_ru_open_stt(path: Path, *, language: str, source: str = "ru_open_stt") -> Iterator[Sample]:
    """snakers4/open_stt -> ``Sample``s. Reads every ``<subset>.csv`` at the corpus root.

    Each CSV row is ``audio_path,txt_path,duration`` (relative to ``path``); the transcript is the
    sibling ``.txt``. Rows whose audio or transcript is missing are skipped (a partly-extracted
    archive leaves CSV rows with no files yet).
    """
    for manifest in sorted(path.glob("*.csv")):
        subset = manifest.stem
        with manifest.open(encoding="utf-8") as handle:
            for index, row in enumerate(csv.reader(handle)):
                if len(row) < _MIN_OPEN_STT_COLS:
                    continue
                audio = path / row[0]
                text = _read_text_file(path / row[1])
                if not text or not audio.exists():
                    continue
                duration = float(row[2]) if len(row) > _MIN_OPEN_STT_COLS and row[2] else 0.0
                yield Sample(
                    id=f"{source}_{_slug(subset)}_{index}",
                    source=source,
                    language=language,
                    text=text,
                    audio_path=str(audio),
                    duration=duration,
                    sample_rate=0,  # set when process resamples
                    split="train",
                    citation="github.com/snakers4/open_stt",
                    meta={"subset": subset},
                )


# -- sova_dataset -------------------------------------------------------------------------------


def _sova_rows(manifest: Path) -> Iterable[dict[str, object]]:
    if manifest.suffix == ".jsonl":
        lines = manifest.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    delimiter = "\t" if manifest.suffix == ".tsv" else ","
    with manifest.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _sova_from_manifest(
    manifest: Path, *, root: Path, language: str, source: str
) -> Iterator[Sample]:
    """Parse a SOVA manifest (jsonl, or csv/tsv with path+text columns)."""
    for index, row in enumerate(_sova_rows(manifest)):
        rel = row.get("path") or row.get("audio") or row.get("audio_path") or row.get("wav")
        text = str(row.get("text") or row.get("transcript") or row.get("sentence") or "").strip()
        if not rel or not text:
            continue
        audio = root / str(rel)
        if not audio.exists():
            continue
        yield Sample(
            id=f"{source}_{_slug(str(rel))}_{index}",
            source=source,
            language=language,
            text=text,
            audio_path=str(audio),
            duration=0.0,
            sample_rate=0,
            split="train",
            citation="github.com/sovaai/sova-dataset",
        )


def _sova_from_tree(root: Path, *, language: str, source: str) -> Iterator[Sample]:
    """No manifest: walk the tree, pairing each ``.wav`` with its same-stem sibling ``.txt``."""
    index = 0
    for audio in sorted(root.rglob("*.wav")):
        text = _read_text_file(audio.with_suffix(".txt"))
        if not text:
            continue
        yield Sample(
            id=f"{source}_{_slug(str(audio.relative_to(root)))}_{index}",
            source=source,
            language=language,
            text=text,
            audio_path=str(audio),
            duration=0.0,
            sample_rate=0,
            split="train",
            citation="github.com/sovaai/sova-dataset",
        )
        index += 1


def load_sova(path: Path, *, language: str, source: str = "sova_dataset") -> Iterator[Sample]:
    """SOVA dataset -> ``Sample``s. Prefers ``manifest.{jsonl,csv,tsv}``; else pairs wav<->.txt."""
    for name in ("manifest.jsonl", "manifest.csv", "manifest.tsv"):
        manifest = path / name
        if manifest.exists():
            yield from _sova_from_manifest(manifest, root=path, language=language, source=source)
            return
    yield from _sova_from_tree(path, language=language, source=source)


# -- timit_ru -----------------------------------------------------------------------------------


def _timit_transcript(txt: Path) -> str:
    """Recover the sentence from a TIMIT ``.TXT`` line: strip the two leading sample indices."""
    line = _read_text_file(txt)
    parts = line.split(maxsplit=_TIMIT_TXT_PARTS - 1)
    return parts[-1].strip() if len(parts) == _TIMIT_TXT_PARTS else line


def load_timit_ru(path: Path, *, language: str, source: str = "timit_ru") -> Iterator[Sample]:
    """DARPA TIMIT layout -> ``Sample``s. ``<split>/DR*/<spkr>/<utt>.{WAV,TXT}``.

    Forced to ``split="train"`` regardless of the on-disk TRAIN/TEST split (a third-party set's own
    test split must not enter the project benchmark). Speaker id is the parent dir name.
    """
    for wav in sorted(path.rglob("*.WAV")):
        text = _timit_transcript(wav.with_suffix(".TXT"))
        if not text:
            continue
        rel = wav.relative_to(path)
        yield Sample(
            id=f"{source}_{_slug(str(rel.with_suffix('')))}",
            source=source,
            language=language,
            text=text,
            audio_path=str(wav),
            duration=0.0,
            sample_rate=0,
            split="train",
            speaker_id=wav.parent.name,
            citation="LDC TIMIT (LDC93S1)",
            meta={"dr": rel.parts[_DR_DEPTH] if len(rel.parts) > _DR_DEPTH else None},
        )


# -- factory ------------------------------------------------------------------------------------

_LOADERS: dict[str, Callable[..., Iterator[Sample]]] = {
    "ru_open_stt": load_ru_open_stt,
    "sova_dataset": load_sova,
    "timit_ru": load_timit_ru,
}


def local_corpus_source(name: str, path: Path, *, force_split: str = "train"):  # noqa: ANN201
    """Wrap a local-corpus loader into an ``IngestFn`` (parse manifest -> 16 kHz mono FLAC).

    ``name`` selects the parser (``ru_open_stt`` | ``sova_dataset`` | ``timit_ru``); ``path`` is the
    corpus root on disk. Audio is resampled into ``project.canonical_dir / name`` and every row's
    split is forced to ``force_split`` (default ``train`` — these are third-party sets that must not
    seed the project benchmark). Returns the loader the project wires into its ``ingests`` map.
    """
    import dataclasses

    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(f"unknown local corpus {name!r}; expected one of {sorted(_LOADERS)}")

    def load(project: CuratorProject) -> Iterable[Sample]:
        from omni_curator.process import resample_samples

        samples = loader(path, language=project.language, source=name)
        resampled = resample_samples(samples, project.canonical_dir / name)
        return (dataclasses.replace(s, split=force_split) for s in resampled)

    return load
