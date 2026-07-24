"""English curation configuration over the shared omni-curator pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_curator.audit.coverage import nemo_sentencepiece_coverage
from omni_curator.ingest.commonvoice import load_commonvoice_archive
from omni_curator.project import CuratorProject, huggingface_source
from omni_curator.project import main as project_main

from english_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources
from english_asr.cv26 import load_identity_ledger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omni_curator.data.sample import Sample

_BASE_MODEL = ROOT.parents[1] / "base_models" / "parakeet" / "parakeet-tdt_ctc-110m.nemo"
_CV26_ARCHIVE = (
    ROOT.parents[1]
    / "projects/common-voice-scripted-speech-26-0/data/raw/archives"
    / "common-voice-scripted-speech-26-0-englis-c84784ae.tar.gz"
)
_CV_SPONTANEOUS_ARCHIVE = (
    ROOT.parents[1]
    / "projects/common-voice-scripted-speech-26-0/data/raw/archives"
    / "common-voice-spontaneous-speech-4-0-engl-c643378f.tar.gz"
)
_CV9_LEDGER = DATA / "ledgers/cv9-open-asr-test-b6bdcd0b.jsonl"


def _load_cv26(project: CuratorProject) -> Iterable[Sample]:
    """Stream benchmark-clean CV26 upstream train directly from the MDC archive."""
    excluded = load_identity_ledger(_CV9_LEDGER)
    return load_commonvoice_archive(
        _CV26_ARCHIVE,
        language=project.language,
        source="common-voice-26-english",
        audio_dir=project.canonical_dir / "common-voice-26-english",
        validation_fraction=0.05,
        split_seed=17,
        excluded_clip_ids=excluded.clips,
        excluded_audio_sha256=excluded.audio_sha256,
    )


def _load_cv_spontaneous(project: CuratorProject) -> Iterable[Sample]:
    """Stream the official Common Voice Spontaneous upstream train split."""
    return load_commonvoice_archive(
        _CV_SPONTANEOUS_ARCHIVE,
        language=project.language,
        source="common-voice-spontaneous-4-english",
        audio_dir=project.canonical_dir / "common-voice-spontaneous-4-english",
        upstream_split="train",
        validation_fraction=0.05,
        split_seed=17,
    )


PROJECT = CuratorProject(
    name="english",
    language=LANGUAGE,
    script=SCRIPT,
    data=DATA,
    db=DB,
    channels=(),
    ingests={
        **{
            source.name: huggingface_source(
                source.repo,
                config=source.config,
                revision=source.revision,
                splits=source.splits,
                source=source.name,
                id_column=source.id_column,
                speaker_column=source.speaker_column,
                streaming=source.streaming,
                max_hours_per_split=source.max_hours,
                shuffle_seed=source.shuffle_seed,
                validation_fraction=source.validation_fraction,
                split_group_column=source.split_group_column,
            )
            for source in sources.ACTIVE_HF_CORPORA
        },
        "common-voice-26-english": _load_cv26,
        "common-voice-spontaneous-4-english": _load_cv_spontaneous,
    },
    licenses=sources.LICENSES,
    env_file=ROOT.parents[1] / ".env",
    coverage_check=nemo_sentencepiece_coverage(_BASE_MODEL),
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
