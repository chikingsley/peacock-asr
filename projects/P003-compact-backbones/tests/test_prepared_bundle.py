from __future__ import annotations

import mmap
from pathlib import Path

import numpy as np

from p003_compact.scoring.prepared_bundle import (
    load_prepared_bundle,
    save_prepared_bundle,
)


def test_prepared_bundle_round_trip_uses_memmap(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "prepared.bundle"
    prepared = {
        "test": [
            (
                np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
                [1, 2],
                ["AA", "B"],
                [3.0, 4.0],
            )
        ]
    }
    meta: dict[str, object] = {
        "backend": "hf-ctc (demo)",
        "dataset_revision": "demo-rev",
        "transport_dtype": "float32",
        "skipped": 0,
    }

    save_prepared_bundle(bundle_dir, prepared, meta)
    loaded = load_prepared_bundle(bundle_dir)

    assert loaded is not None
    loaded_meta, loaded_bundle = loaded
    assert loaded_meta["backend"] == "hf-ctc (demo)"
    assert (bundle_dir / "test" / "posteriors.npy").exists()

    posteriors, phone_indices, valid_phones, valid_scores = loaded_bundle["test"][0]
    root_base: object = posteriors
    while isinstance(root_base, np.ndarray):
        base = root_base.base
        if base is None:
            break
        root_base = base
    assert isinstance(root_base, mmap.mmap)
    assert phone_indices == [1, 2]
    assert valid_phones == ["AA", "B"]
    assert valid_scores == [3.0, 4.0]
    np.testing.assert_allclose(
        np.asarray(posteriors),
        np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
    )
