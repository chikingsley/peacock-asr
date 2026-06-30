from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "artifacts/cache/huggingface/models--OpenMOSS-Team--MOSS-Transcribe-preview-2B"
    / "snapshots/c98175cb20e48bd9be4e95f6c85f2af18899f780"
)
DEFAULT_CONFIG = SNAPSHOT_DIR / "config.json"
DEFAULT_REFERENCE_TENSORS = (
    PROJECT_ROOT / "artifacts/reference/libri1-pytorch-bf16/reference_tensors.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/coreml/moss_swift_fixture.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the MOSS CoreML LibriSpeech fixture as Swift-readable JSON."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-tensors", type=Path, default=DEFAULT_REFERENCE_TENSORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def flatten_float32(array: np.ndarray) -> list[float]:
    return [float(value) for value in array.astype(np.float32).reshape(-1)]


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    config_data = load_json(args.config.resolve())
    language_config = config_data["language_config"]
    tensors = np.load(args.reference_tensors.resolve())
    input_ids = tensors["input_ids"].astype(np.int32)
    audio_input_mask = tensors["audio_input_mask"].astype(bool)
    audio_data = tensors["audio_data"].astype(np.float32)
    audio_data_seqlens = tensors["audio_data_seqlens"].astype(np.int32)
    generated_ids = tensors["generated_ids"].astype(np.int32)
    payload: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "reference_tensors": str(args.reference_tensors.resolve()),
        "prompt_len": int(input_ids.shape[1]),
        "hidden_size": int(language_config["hidden_size"]),
        "head_dim": int(language_config["head_dim"]),
        "rope_theta": float(language_config["rope_theta"]),
        "input_ids": input_ids.reshape(-1).tolist(),
        "audio_input_mask": audio_input_mask.reshape(-1).tolist(),
        "audio_data_shape": list(audio_data.shape),
        "audio_data": flatten_float32(audio_data),
        "audio_data_seqlens": audio_data_seqlens.reshape(-1).tolist(),
        "generated_ids": generated_ids.reshape(-1).tolist(),
    }
    write_json(output, payload)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
