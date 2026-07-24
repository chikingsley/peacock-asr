"""Export Selection contract: gates are train-only, held-out videos regroup leakage-safe.

Locks in two measurement-integrity rules learned the hard way:
- benchmark splits are never censored by curation gates (commit 082a04c1 — the v2 export's
  WER gate silently ate 20 FLEURS rows);
- held-out conversational test videos are gated like the train rows they are stored as,
  with survivors regrouped to test and the rest dropped, so no held-out video reaches train
  (commit a6b067e2).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from omni_curator.audit.quality import is_descriptor_only
from omni_curator.data.export import (
    Selection,
    YoutubeSplitPolicy,
    _normalize_and_filter,
    export_dataset,
    export_nemo_manifests,
    normalize_youtube_category,
    write_weighted_distribution,
)
from omni_curator.data.provenance import (
    LicenseInfo,
    SourceProvenance,
    TransformStep,
    normalize_license_registry,
)
from omni_curator.data.store import CuratorStore


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


def test_store_collection_filters_select_only_requested_rows(make_sample, tmp_path):
    store = CuratorStore(tmp_path / "store.sqlite")
    store.upsert(
        [
            make_sample(id="a-train", source="a", split="train"),
            make_sample(id="a-dev", source="a", split="dev"),
            make_sample(id="b-train", source="b", split="train"),
        ]
    )

    ids = [sample.id for sample in store.iter_samples(sources=["a"], splits=["train"])]
    store.close()

    assert ids == ["a-train"]


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


def test_trusted_source_bypasses_language_gate(make_sample):
    sel = Selection(trusted_language_sources=frozenset({"fleurs"}))
    trusted = make_sample(source="fleurs", split="train")
    ordinary = make_sample(source="youtube", split="train")

    assert not sel.applies_language_gate(trusted)
    assert sel.applies_language_gate(ordinary)


def test_trusted_provenance_authority_bypasses_language_gate(make_sample):
    provenance = SourceProvenance(origin="hf", authority="gold-corpus", tool="ingest")
    sample = make_sample(source="hf-example", split="train").with_provenance(provenance)
    sel = Selection(trusted_language_authorities=frozenset({"gold-corpus"}))

    assert not sel.applies_language_gate(sample)


def test_normalized_empty_label_is_always_dropped(make_sample, tmp_path):
    store = CuratorStore(tmp_path / "store.sqlite")
    store.upsert([make_sample(language="eng_Latn", text="[silence] わかりません")])

    grouped, dropped = _normalize_and_filter(store, Selection(language_gate=False))
    store.close()

    assert grouped == {}
    assert dropped == {"empty_normalized_text": 1}


def test_normalized_descriptor_only_label_is_dropped_from_train(make_sample, tmp_path):
    store = CuratorStore(tmp_path / "store.sqlite")
    store.upsert([make_sample(language="eng_Latn", text="わかりません?")])

    grouped, dropped = _normalize_and_filter(store, Selection(language_gate=False))
    store.close()

    assert grouped == {}
    assert dropped == {"descriptor_only_normalized": 1}


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


def test_sample_provenance_round_trips_through_meta(make_sample):
    provenance = SourceProvenance(
        origin="youtube",
        authority="channel_registry",
        tool="yt-dlp",
        license=LicenseInfo("cc-by-4.0", commercial_use=True, authority="source-card"),
        transforms=(
            TransformStep("download", "yt-dlp", version="2026.03.17"),
            TransformStep("segment", "nemo-vad", parameters={"max_duration": 30.0}),
        ),
    )

    sample = make_sample().with_provenance(provenance)

    assert sample.provenance == provenance
    assert sample.meta["provenance"]["license"]["commercial_use"] is True


def test_license_registry_accepts_legacy_tuples_and_typed_values():
    registry = normalize_license_registry(
        {
            "fleurs": ("cc-by-4.0", True),
            "youtube": LicenseInfo("unknown", commercial_use=False),
            "hf": {"id": "cc0-1.0", "commercial_use": True, "authority": "dataset-card"},
        }
    )

    assert registry["fleurs"] == LicenseInfo("cc-by-4.0", commercial_use=True)
    assert registry["youtube"] == LicenseInfo("unknown", commercial_use=False)
    assert registry["hf"] == LicenseInfo("cc0-1.0", commercial_use=True, authority="dataset-card")


def test_youtube_category_taxonomy_normalizes_aliases():
    assert normalize_youtube_category("news") == "news"
    assert normalize_youtube_category("language learning") == "language_learning"
    assert normalize_youtube_category("kids") == "children"
    assert normalize_youtube_category("made-up") == "uncategorized"
    with pytest.raises(ValueError, match="dev_ratio"):
        YoutubeSplitPolicy(dev_ratio=0.7, test_ratio=0.4)


def test_youtube_split_policy_is_category_stratified_and_video_disjoint(make_sample, tmp_path):
    store = CuratorStore(tmp_path / "store.sqlite")
    samples = []
    for category in ("news", "education"):
        for video_idx in range(4):
            video_id = f"{category}_vid{video_idx}"
            samples.extend(
                [
                    make_sample(
                        id=f"{video_id}_{clip_idx:04d}",
                        source=f"youtube-{category}",
                        meta={"category": category},
                    )
                    for clip_idx in range(2)
                ]
            )
    store.upsert(samples)

    grouped, dropped = _normalize_and_filter(
        store,
        Selection(
            language_gate=False,
            youtube_split_policy=YoutubeSplitPolicy(
                dev_ratio=0.25, test_ratio=0.25, seed="test-seed"
            ),
        ),
    )
    store.close()

    assert dropped == {}
    video_splits: dict[str, set[str]] = defaultdict(set)
    split_counts_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for (_source, split, _language), rows in grouped.items():
        for sample, _norm_text in rows:
            video_id = sample.id.rsplit("_", 1)[0]
            category = str(sample.meta["category"])
            video_splits[video_id].add(split)
            split_counts_by_category[category][split] += 1

    assert all(len(splits) == 1 for splits in video_splits.values())
    for counts in split_counts_by_category.values():
        assert counts == {"train": 4, "dev": 2, "test": 2}


def test_export_writes_sample_metadata(make_sample, tmp_path):
    audio = tmp_path / "clip.flac"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, format="FLAC")
    sample = make_sample(
        audio_path=str(audio),
        meta={"category": "news", "tier": "clean", "title": "Video title"},
    )
    store = CuratorStore(tmp_path / "store.sqlite")
    store.upsert([sample])

    stats = export_dataset(
        store,
        tmp_path / "dataset",
        selection=Selection(language_gate=False),
        row_group_size=1,
    )
    store.close()

    assert stats.rows == 1
    parquet = next(
        (tmp_path / "dataset" / "version=0").glob("corpus=*/split=*/language=*/*.parquet")
    )
    table = pq.read_table(parquet)
    assert table.schema.names[:3] == ["text", "audio_bytes", "audio_size"]
    assert json.loads(table.column("metadata")[0].as_py()) == {
        "category": "news",
        "tier": "clean",
        "title": "Video title",
    }


def test_export_preserves_canonical_flac_bytes(make_sample, tmp_path):
    audio = tmp_path / "clip.flac"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, format="FLAC")
    expected = audio.read_bytes()
    store = CuratorStore(tmp_path / "store.sqlite")
    store.upsert([make_sample(audio_path=str(audio))])

    export_dataset(
        store,
        tmp_path / "dataset",
        selection=Selection(language_gate=False),
        row_group_size=1,
    )
    store.close()

    parquet = next(
        (tmp_path / "dataset" / "version=0").glob("corpus=*/split=*/language=*/*.parquet")
    )
    table = pq.read_table(parquet)
    actual = np.asarray(table.column("audio_bytes")[0].as_py(), dtype=np.int8).tobytes()
    assert actual == expected


def test_nemo_manifest_export_references_canonical_audio_without_copy(make_sample, tmp_path):
    audio = tmp_path / "clip.flac"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, format="FLAC")
    store = CuratorStore(tmp_path / "store.sqlite")
    store.upsert(
        [
            make_sample(
                audio_path=str(audio),
                language="eng_Latn",
                text="Hello, world!",
                meta={"category": "news"},
            )
        ]
    )

    output = tmp_path / "manifest-export"
    stats = export_nemo_manifests(
        store,
        output,
        selection=Selection(language_gate=False),
        licenses={"test": ("cc0-1.0", True)},
    )
    store.close()

    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert stats.rows == 1
    assert row["audio_filepath"] == str(audio.resolve())
    assert row["text"] == "Hello, world!"
    assert row["metadata"] == {"category": "news"}
    assert list(output.rglob("*.flac")) == []
    summary = json.loads((output / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["audio_mode"] == "reference"
    with pytest.raises(FileExistsError, match="immutable"):
        export_nemo_manifests(store, output)
