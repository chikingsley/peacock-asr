"""Typed English corpus registry and bounded first-wave ingest configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class CorpusSource:
    """One serious English corpus candidate, including its legal and operational state."""

    name: str
    wave: str
    status: str
    role: str
    hours: str
    domain: str
    license: str
    commercial_use: bool
    label_authority: str
    homepage: str
    repo: str | None = None
    revision: str | None = None
    config: str | None = None
    splits: tuple[str, ...] = ("train",)
    id_column: str | None = None
    speaker_column: str | None = None
    split_group_column: str | None = None
    max_hours: float | None = None
    streaming: bool = True
    shuffle_seed: int | None = 17
    validation_fraction: float = 0.05
    enabled: bool = False
    note: str = ""


# The active wave is deliberately bounded. Every source reads only its upstream train split and
# carves a deterministic group-disjoint dev partition from that train material. Upstream
# validation/test audio never enters training or internal validation.
CORPORA: tuple[CorpusSource, ...] = (
    CorpusSource(
        name="peoples-speech-microset",
        wave="smoke",
        status="ready",
        role="pipeline proof",
        hours="roughly 1.3",
        domain="mixed public speech",
        license="CC-BY / CC-BY-SA (row lineage must be retained)",
        commercial_use=True,
        label_authority="upstream mixed human and machine transcripts",
        homepage="https://mlcommons.org/datasets/peoples-speech/",
        repo="MLCommons/peoples_speech",
        revision="f10597c5d3d3a63f8b6827701297c3afdf178272",
        config="microset",
        id_column="id",
        split_group_column="id",
        max_hours=2.0,
        enabled=True,
        note="336-row, roughly 92 MB smoke corpus; proves ingest through training manifests.",
    ),
    CorpusSource(
        name="gigaspeech-xs",
        wave="1",
        status="ready",
        role="novel training",
        hours="10",
        domain="audiobook, podcast, and YouTube",
        license="Apache-2.0",
        commercial_use=True,
        label_authority="upstream GigaSpeech train transcripts",
        homepage="https://github.com/SpeechColab/GigaSpeech",
        repo="speechcolab/gigaspeech",
        revision="63c0836b643dc6136a608de041e56b67c12649b3",
        config="xs",
        id_column="segment_id",
        speaker_column="speaker",
        split_group_column="audio_id",
        max_hours=10.0,
        enabled=True,
        note="Train only. Do not ingest the public validation/test splits.",
    ),
    CorpusSource(
        name="gigaspeech-s-100h",
        wave="2",
        status="preparing",
        role="larger replacement ablation for GigaSpeech XS",
        hours="100 bounded from the S training subset",
        domain="audiobook, podcast, and YouTube",
        license="Apache-2.0",
        commercial_use=True,
        label_authority="upstream GigaSpeech train transcripts",
        homepage="https://github.com/SpeechColab/GigaSpeech",
        repo="speechcolab/gigaspeech",
        revision="63c0836b643dc6136a608de041e56b67c12649b3",
        config="s",
        id_column="segment_id",
        speaker_column="speaker",
        split_group_column="audio_id",
        max_hours=100.0,
        enabled=True,
        note="Replacement arm only: the S subset contains the smaller XS material.",
    ),
    CorpusSource(
        name="ami-ihm-25h",
        wave="1",
        status="ready",
        role="novel training",
        hours="25 bounded from roughly 78 train",
        domain="multi-speaker meetings, headset microphones",
        license="CC-BY-4.0",
        commercial_use=True,
        label_authority="official human orthographic transcripts",
        homepage="https://groups.inf.ed.ac.uk/ami/corpus/",
        repo="edinburghcstr/ami",
        revision="46f28f2503e2ec48f8867a84eef356c70476beab",
        config="ihm",
        id_column="audio_id",
        speaker_column="speaker_id",
        split_group_column="meeting_id",
        max_hours=25.0,
        enabled=True,
        note="Meeting-disjoint internal dev carve; upstream validation/test remain untouched.",
    ),
    CorpusSource(
        name="icsi",
        wave="1",
        status="official-adapter-needed",
        role="novel training",
        hours="roughly 70",
        domain="natural meetings",
        license="CC-BY-4.0",
        commercial_use=True,
        label_authority="official human orthographic transcripts",
        homepage="https://groups.inf.ed.ac.uk/ami/icsi/",
        note="Use the official release; no public HF adapter was confirmed in the live audit.",
    ),
    CorpusSource(
        name="common-voice-26-english",
        wave="1",
        status="archive-download-resumable-ingest-ready",
        role="benchmark-clean training with measured replay when available",
        hours="2784 validated in CV26",
        domain="crowdsourced read prompts and broad accents",
        license="CC0-1.0",
        commercial_use=True,
        label_authority="Common Voice validated community references",
        homepage="https://commonvoice.mozilla.org/en/datasets",
        enabled=True,
        note=(
            "The english-cv26 prep command reads upstream train only and requires frozen benchmark "
            "identities. An exact CV7 ledger may additionally split base replay from post-CV7 "
            "candidates; without one, benchmark-clean rows remain usable and replay is recorded as "
            "unknown rather than guessed."
        ),
    ),
    CorpusSource(
        name="common-voice-spontaneous-4-english",
        wave="1",
        status="ready",
        role="novel conversational training",
        hours="small",
        domain="crowdsourced spontaneous speech",
        license="CC0-1.0",
        commercial_use=True,
        label_authority="Common Voice community references",
        homepage="https://commonvoice.mozilla.org/en/datasets",
        enabled=True,
        note="Official upstream train only; group-disjoint internal dev carve by client id.",
    ),
    CorpusSource(
        name="peoples-speech-clean",
        wave="2",
        status="ready-after-pilot",
        role="large-scale training",
        hours="30000+",
        domain="diverse public speech",
        license="CC-BY / CC-BY-SA (row lineage must be retained)",
        commercial_use=True,
        label_authority="mixed upstream transcripts",
        homepage="https://mlcommons.org/datasets/peoples-speech/",
        repo="MLCommons/peoples_speech",
        revision="f10597c5d3d3a63f8b6827701297c3afdf178272",
        config="clean",
        id_column="id",
        split_group_column="id",
        max_hours=100.0,
        enabled=True,
        note=(
            "The first immutable export remains the 10-hour pilot; prepare a 100-hour replacement "
            "arm."
        ),
    ),
    CorpusSource(
        name="librispeech-train-clean-100-replay",
        wave="replay",
        status="ready",
        role="bounded base-distribution retention replay",
        hours="100 from official train-clean-100 only",
        domain="read English audiobooks",
        license="CC-BY-4.0",
        commercial_use=True,
        label_authority="official LibriSpeech train-clean-100 references",
        homepage="https://www.openslr.org/12/",
        repo="openslr/librispeech_asr",
        revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        config="clean",
        splits=("train.100",),
        id_column="id",
        speaker_column="speaker_id",
        split_group_column="speaker_id",
        max_hours=100.0,
        enabled=True,
        note=(
            "Replay-only source at 10% mixture exposure. Upstream validation and test remain "
            "untouched reporting exams; the internal dev carve is speaker-disjoint."
        ),
    ),
    CorpusSource(
        name="nsc-parts-2-6",
        wave="2",
        status="license-and-adapter-audit",
        role="accent and conversational training",
        hours="3000+",
        domain="Singapore English read and conversational speech",
        license="source-specific",
        commercial_use=False,
        label_authority="official corpus references",
        homepage="https://www.imda.gov.sg/how-we-can-help/national-speech-corpus",
        note=(
            "Part 1 was in the 110M base training set; target only Parts 2-6 after license review."
        ),
    ),
    CorpusSource(
        name="nvidia-granary-english",
        wave="2",
        status="source-license-and-contamination-audit",
        role="large pseudo-label pool",
        hours="roughly 250000 en-US",
        domain="mixed",
        license="mixed by source",
        commercial_use=False,
        label_authority="machine pseudo-labels plus source metadata",
        homepage="https://huggingface.co/datasets/nvidia/Granary",
        note="Do not flatten source licenses or benchmark overlap into one permissive corpus.",
    ),
    CorpusSource(
        name="yodas-english",
        wave="later",
        status="weak-label-license-and-dedup-audit",
        role="large weak-label pool",
        hours="roughly 169853",
        domain="YouTube captions",
        license="source-specific/unclear",
        commercial_use=False,
        label_authority="weak captions",
        homepage="https://huggingface.co/datasets/espnet/yodas",
    ),
    CorpusSource(
        name="libri-light",
        wave="later",
        status="pseudo-label-and-dedup-needed",
        role="unlabelled read-speech pool",
        hours="60000",
        domain="audiobooks",
        license="CC-BY-4.0",
        commercial_use=True,
        label_authority="none; teacher required",
        homepage="https://github.com/facebookresearch/libri-light",
        note="Low domain novelty and high LibriSpeech overlap make it lower priority.",
    ),
    CorpusSource(
        name="mls-english",
        wave="replay",
        status="dedup-before-delta",
        role="base replay and possible novel delta",
        hours="44659 train",
        domain="audiobooks",
        license="CC-BY-4.0",
        commercial_use=True,
        label_authority="official MLS references",
        homepage="https://www.openslr.org/94/",
        note="The 110M base already saw roughly 2000 hours of MLS English.",
    ),
    CorpusSource(
        name="ted-lium-3",
        wave="research-only",
        status="noncommercial-no-derivatives",
        role="do not use for a distributable model",
        hours="452",
        domain="TED talks",
        license="CC-BY-NC-ND-3.0",
        commercial_use=False,
        label_authority="official corpus references",
        homepage="https://www.openslr.org/51/",
    ),
    CorpusSource(
        name="spgispeech",
        wave="research-only",
        status="restrictive-agreement",
        role="do not ingest without legal approval",
        hours="roughly 4900 train",
        domain="financial presentations",
        license="Kensho agreement",
        commercial_use=False,
        label_authority="professional references",
        homepage="https://datasets.kensho.com/datasets/scribe",
    ),
)

ACTIVE_HF_CORPORA = tuple(source for source in CORPORA if source.enabled and source.repo)

LICENSES = {
    source.name: (source.license, source.commercial_use) for source in CORPORA if source.enabled
}

# Exact source families reported in the NVIDIA 110M model card. These can be useful as a replay
# anchor, but they are not counted as novel data: LibriSpeech, Fisher, NSC Part 1, VCTK,
# VoxPopuli English, Europarl-ASR English, about 2k hours of MLS English, and Common Voice v7.
BASE_TRAINING_REPLAY = (
    "librispeech-960",
    "fisher",
    "nsc-part-1",
    "vctk",
    "voxpopuli-en",
    "europarl-asr-en",
    "mls-en-2kh",
    "common-voice-v7-en",
)

# Never train on leaderboard exams. Public availability is not permission to include them in the
# training pool.
EXCLUDED_BENCHMARKS = (
    "earnings21",
    "earnings22",
    "open-asr-leaderboard packaged test sets",
    "long-form leaderboard test sets",
    "every upstream validation/test split used for external reporting",
)
