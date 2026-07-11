import io
import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from asr_benchmark_core.adapters import Adapter
from asr_benchmark_core.data import Example

from farsi_asr.quality_v4_full import score_asr

ROW_COUNT = 2


class FakeAdapter(Adapter):
    def transcribe(self, example: Example) -> str:
        return example.reference


def _flac() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(1600, dtype=np.float32), 16_000, format="FLAC")
    return stream.getvalue()


def test_full_asr_ledger_is_resumable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    shard = (
        root
        / "version=0"
        / "corpus=fleurs"
        / "split=train"
        / "language=fas_Arab"
        / "part-00000.parquet"
    )
    shard.parent.mkdir(parents=True)
    encoded = _flac()
    stored = np.frombuffer(encoded, dtype=np.int8).tolist()
    pq.write_table(
        pa.table(
            {
                "text": ["سلام", "درود"],
                "audio_bytes": [stored, stored],
                "audio_size": [1600, 1600],
            }
        ),
        shard,
    )
    database = tmp_path / "quality.sqlite3"
    args = Namespace(
        dataset_root=root,
        database=database,
        model=tmp_path / "model",
        model_sha256="abc123",
        adapter="whisper",
        language="Persian",
        device="cpu",
        batch_size=1,
        limit_shards=0,
        limit_rows=0,
        expected_shards=1,
        hub_repo="Peacockery/farsi-asr-corpus-v4",
        hub_revision="revision",
    )

    assert score_asr(args, adapter=FakeAdapter()) == 0
    assert score_asr(args, adapter=FakeAdapter()) == 0

    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT source, text, hypothesis, asr_agreement_json FROM quality_rows "
        "ORDER BY hub_row_index"
    ).fetchall()
    connection.close()
    assert len(rows) == ROW_COUNT
    assert rows[0][:3] == ("fleurs", "سلام", "سلام")
    assert json.loads(rows[0][3])["wer"] == 0.0
