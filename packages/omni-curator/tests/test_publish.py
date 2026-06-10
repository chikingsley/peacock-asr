"""HF audio-dataset publisher: ungated scores ride along, junk and wrong-language drop."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from omni_curator.publish import export_hf_audio_dataset
from omni_curator.store import CuratorStore


@pytest.fixture
def store(tmp_path, make_sample):
    s = CuratorStore(tmp_path / "store.sqlite")
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"fLaC-fake-bytes")
    s.upsert(
        [
            make_sample(id="chan_vidA_0000", source="youtube-chan", text="ман китоб хондам",
                        audio_path=str(clip), scribe_wer=0.1, scribe_cer=0.05),
            # terrible score, but published anyway (scores are columns, not gates)
            make_sample(id="chan_vidA_0001", source="youtube-chan", text="хеле бад шуд ин ҷо",
                        audio_path=str(clip), scribe_wer=0.92, scribe_cer=0.5),
            make_sample(id="chan_vidA_0002", source="youtube-chan", text="[outro jingle]",
                        audio_path=str(clip), scribe_wer=0.0),
            make_sample(id="chan_vidA_0003", source="youtube-chan",
                        text="Это очень хорошо, мы согласны", audio_path=str(clip)),
            make_sample(id="x_0", source="fleurs", text="на гӯшт", audio_path=str(clip)),
        ]
    )
    yield s
    s.close()


def test_publishes_ungated_with_scores(store, tmp_path):
    stats = export_hf_audio_dataset(
        store, tmp_path / "out", language="tgk_Cyrl", rows_per_shard=10
    )
    assert stats.rows == 2  # the good clip AND the terrible-score clip
    assert stats.skipped == {"descriptor_only": 1, "language_gate": 1}
    assert stats.shards == 1
    shard = next((tmp_path / "out" / "data").glob("train-*.parquet"))
    assert shard.name == "train-00000-of-00001.parquet"
    table = pq.read_table(shard)
    assert table.column("scribe_wer").to_pylist() == [0.1, 0.92]
    assert table.column("channel").to_pylist() == ["chan", "chan"]
    assert table.column("video_id").to_pylist() == ["chan_vidA", "chan_vidA"]
    audio0 = table.column("audio").to_pylist()[0]
    assert audio0["bytes"].startswith(b"fLaC")
    assert audio0["path"] == "chan_vidA_0000.flac"


def test_refuses_to_mix_exports(store, tmp_path):
    export_hf_audio_dataset(store, tmp_path / "out", language="tgk_Cyrl")
    with pytest.raises(FileExistsError):
        export_hf_audio_dataset(store, tmp_path / "out", language="tgk_Cyrl")
