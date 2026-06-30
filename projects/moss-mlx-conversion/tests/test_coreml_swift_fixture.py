import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_exporter() -> ModuleType:
    exporter_path = (
        Path(__file__).resolve().parents[1] / "coreml" / "export_swift_fixture.py"
    )
    spec = importlib.util.spec_from_file_location("export_swift_fixture", exporter_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_prompt_extracts_moss_template_fields() -> None:
    exporter = load_exporter()
    input_ids = np.array(
        [[151644, 872, 198, 151669, 0, 0, 151670, 151645, 198, 151644, 77091, 198]],
        dtype=np.int32,
    )
    audio_input_mask = np.array(
        [[False, False, False, False, True, True, False, False, False, False, False, False]],
        dtype=np.bool_,
    )

    payload = exporter.split_prompt(input_ids, audio_input_mask)

    assert payload == {
        "prompt_prefix_ids": [151644, 872, 198, 151669],
        "prompt_suffix_ids": [151670, 151645, 198, 151644, 77091, 198],
        "audio_token_count": 2,
        "audio_placeholder_id": 0,
    }


def test_split_prompt_rejects_non_contiguous_audio_mask() -> None:
    exporter = load_exporter()
    input_ids = np.array([[151644, 0, 151670, 0]], dtype=np.int32)
    audio_input_mask = np.array([[False, True, False, True]], dtype=np.bool_)

    with pytest.raises(ValueError, match="not one contiguous audio block"):
        exporter.split_prompt(input_ids, audio_input_mask)
