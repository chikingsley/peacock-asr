"""Shared test fixtures for P011.

Data fixture reads P011_FEATURES_DIR from .env or environment, downloads data
if needed, and fails clearly if the path is not configured at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _sentinel(features_dir: Path) -> Path:
    return features_dir / "seq_data_librispeech_v4" / "tr_feat.npy"


@pytest.fixture(scope="session")
def features_dir() -> Path:
    """Return the SpeechOcean762 features directory, downloading if needed.

    Reads P011_FEATURES_DIR from the environment or .env file (via pydantic-settings).
    If data is not downloaded yet, triggers download automatically.
    Fails clearly if P011_FEATURES_DIR is not configured.
    """
    try:
        from p011.settings import Settings
        d = Settings().features_dir  # reads P011_FEATURES_DIR from .env
    except Exception:
        pytest.fail(
            "P011_FEATURES_DIR is not configured.\n"
            "Add it to .env or set the environment variable, then run:\n"
            "    uv run p011 download",
            pytrace=False,
        )

    if not _sentinel(d).exists():
        print(f"\nData not found at {d} — downloading...")
        from p011.data import download_features
        download_features(d)

    assert _sentinel(d).exists(), f"Download completed but sentinel not found: {_sentinel(d)}"
    return d


@pytest.fixture(scope="session")
def synthetic_features_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a tiny synthetic features dir with a frame store and pooled features."""
    root = tmp_path_factory.mktemp("p011-synth")
    data_dir = root / "seq_data_librispeech_v4"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_store_dir = root / "ssl_frame_store_v1"

    n_examples = 4
    max_phones = 50
    rng = np.random.default_rng(42)
    utterance_ids = [f"{idx + 1:09d}" for idx in range(n_examples)]
    speaker_dirs = [f"SPEAKER{idx + 1:04d}" for idx in range(n_examples)]
    split_name = {"tr": "train", "te": "test"}

    def pool_frames(frames: np.ndarray, durations: np.ndarray) -> np.ndarray:
        pooled = np.zeros((max_phones, 25, 1024), dtype=np.float32)
        counts = np.round(durations[durations[:, 0] > 0, 0] * 50.0).astype(np.int64)
        if len(counts):
            counts[-1] += frames.shape[0] - int(counts.sum())
        start = 0
        for phone_idx, count in enumerate(counts[:max_phones]):
            end = min(start + int(count), frames.shape[0])
            if end > start:
                pooled[phone_idx] = frames[start:end].mean(axis=0)
            start = end
        return pooled

    def make_split(prefix: str) -> None:
        gop = np.zeros((n_examples, max_phones, 84), dtype=np.float32)
        energy = np.zeros((n_examples, max_phones, 7), dtype=np.float32)
        dur = np.zeros((n_examples, max_phones, 1), dtype=np.float32)
        ssl_last: dict[str, np.ndarray] = {
            "hubert": np.zeros((n_examples, max_phones, 1024), dtype=np.float32),
            "w2v_300m": np.zeros((n_examples, max_phones, 1024), dtype=np.float32),
            "wavlm": np.zeros((n_examples, max_phones, 1024), dtype=np.float32),
        }
        ssl_all_layers: dict[str, np.ndarray] = {
            "hubert": np.zeros((n_examples, max_phones, 25, 1024), dtype=np.float16),
            "w2v_300m": np.zeros((n_examples, max_phones, 25, 1024), dtype=np.float16),
            "wavlm": np.zeros((n_examples, max_phones, 25, 1024), dtype=np.float16),
        }
        phn_label = np.full((n_examples, max_phones, 2), fill_value=-1.0, dtype=np.float32)
        utt_label = rng.uniform(0.0, 10.0, size=(n_examples, 5)).astype(np.float32)
        word_label = np.full((n_examples, max_phones, 4), fill_value=-1.0, dtype=np.float32)
        word_id = np.full((n_examples, max_phones), fill_value=-1.0, dtype=np.float32)
        diag_label = np.full((n_examples, max_phones), fill_value=-1, dtype=np.int64)

        for example_idx in range(n_examples):
            frame_budget = 8 + example_idx
            phone_counts = np.array([2, 3, frame_budget - 5], dtype=np.int64)
            for phone_idx, count in enumerate(phone_counts):
                gop[example_idx, phone_idx] = rng.normal(size=(84,)).astype(np.float32)
                energy[example_idx, phone_idx] = rng.normal(size=(7,)).astype(np.float32)
                dur[example_idx, phone_idx, 0] = float(count) / 50.0
                phn_label[example_idx, phone_idx, 0] = float((phone_idx + example_idx) % 39)
                phn_label[example_idx, phone_idx, 1] = 0.8 - 0.2 * phone_idx
                word_label[example_idx, phone_idx, 0:3] = np.array([5.0, 4.0, 4.5], dtype=np.float32)
                word_label[example_idx, phone_idx, 3] = float(phone_idx // 2)
                word_id[example_idx, phone_idx] = float(200 + 10 * example_idx + phone_idx)
                diag_label[example_idx, phone_idx] = int(phn_label[example_idx, phone_idx, 0])

        split_dir = frame_store_dir / split_name[prefix]
        split_dir.mkdir(parents=True, exist_ok=True)
        for model_key in ("hubert", "w2v_300m", "wavlm"):
            model_dir = split_dir / model_key
            model_dir.mkdir(parents=True, exist_ok=True)
            entries: list[dict[str, object]] = []
            for example_idx, (utterance_id, speaker_dir) in enumerate(zip(utterance_ids, speaker_dirs, strict=True)):
                frame_budget = int(round(float(dur[example_idx, :, 0].sum()) * 50.0))
                frames = rng.normal(size=(frame_budget, 25, 1024)).astype(np.float16)
                speaker_path = model_dir / speaker_dir
                speaker_path.mkdir(parents=True, exist_ok=True)
                np.save(speaker_path / f"{utterance_id}.npy", frames)

                pooled = pool_frames(frames.astype(np.float32), dur[example_idx])
                ssl_all_layers[model_key][example_idx] = pooled.astype(np.float16)
                ssl_last[model_key][example_idx] = pooled[:, 24, :]
                entries.append(
                    {
                        "utterance_id": utterance_id,
                        "speaker_dir": speaker_dir,
                        "rel_path": f"{speaker_dir}/{utterance_id}.npy",
                        "num_frames": int(frames.shape[0]),
                        "num_layers": 25,
                        "feat_dim": 1024,
                    }
                )
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "split": split_name[prefix],
                        "model_key": model_key,
                        "frame_rate_hz": 50,
                        "entries": entries,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        np.save(data_dir / f"{prefix}_feat.npy", gop)
        np.save(data_dir / f"{prefix}_energy_feat.npy", energy)
        np.save(data_dir / f"{prefix}_dur_feat.npy", dur)
        np.save(data_dir / f"{prefix}_hubert_feat_v2.npy", ssl_last["hubert"])
        np.save(data_dir / f"{prefix}_w2v_300m_feat_v2.npy", ssl_last["w2v_300m"])
        np.save(data_dir / f"{prefix}_wavlm_feat_v2.npy", ssl_last["wavlm"])
        np.save(data_dir / f"{prefix}_label_phn.npy", phn_label)
        np.save(data_dir / f"{prefix}_label_utt.npy", utt_label)
        np.save(data_dir / f"{prefix}_label_word.npy", word_label)
        np.save(data_dir / f"{prefix}_word_id.npy", word_id)
        np.save(data_dir / f"{prefix}_label_diag.npy", diag_label)
        for model_key in ("hubert", "w2v_300m", "wavlm"):
            np.save(data_dir / f"{prefix}_{model_key}_all_layers.npy", ssl_all_layers[model_key])

    make_split("tr")
    make_split("te")

    return root
