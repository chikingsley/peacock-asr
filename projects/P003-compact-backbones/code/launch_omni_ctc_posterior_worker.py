#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.6,<2.9",
#     "torchaudio>=2.6,<2.9",
#     "fairseq2[arrow]>=0.5.2,<=0.6.0",
#     "pyarrow>=20.0.0",
#     "numpy>=1.23,<2",
#     "pandas>=2.2",
#     "polars>=1.29.0",
#     "soundfile>=0.13.1",
#     "pyyaml>=6.0.3",
#     "sentencepiece>=0.2.1",
# ]
# ///
"""Persistent OmniASR phoneme posterior worker for P003 scoring."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "projects" / "P003-compact-backbones"
OMNI_ROOT = (
    REPO_ROOT
    / "projects"
    / "P004-training-from-scratch"
    / "third_party"
    / "omnilingual-asr"
)
EXPECTED_SAMPLE_RATE = 16_000


def _inject_paths() -> None:
    for candidate in (OMNI_ROOT, OMNI_ROOT / "src", PROJECT_ROOT / "code"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _preload_tbb() -> None:
    try:
        tbb_dist = distribution("tbb")
    except PackageNotFoundError as exc:
        raise SystemExit(
            "Missing `tbb` runtime. Launch via `uv run --python 3.12 "
            "--with tbb>=2021.8 --with-editable <omnilingual-asr-path> ...`."
        ) from exc
    lib_path = tbb_dist.locate_file("../../libtbb.so.12")
    ctypes.CDLL(str(lib_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve_checkpoint_dir(run_dir: Path) -> Path:
    candidate = run_dir.expanduser().resolve()
    if (candidate / "checkpoints").is_dir():
        return candidate / "checkpoints"
    return candidate


def _resolve_latest_model_file(checkpoint_dir: Path) -> Path:
    step_dirs = sorted(
        (
            p
            for p in checkpoint_dir.glob("step_*")
            if p.is_dir() and p.name[5:].isdigit()
        ),
        key=lambda p: int(p.name[5:]),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No step_* dirs found in {checkpoint_dir}")
    model_files = sorted(step_dirs[-1].joinpath("model").rglob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint file found in {step_dirs[-1]}")
    return model_files[0]


def _load_pipeline(run_dir: Path, device: str) -> tuple[Any, list[str], int]:
    from fairseq2.composition.assets import register_file_assets  # noqa: PLC0415
    from fairseq2.data.tokenizers.hub import load_tokenizer  # noqa: PLC0415
    from fairseq2.models.hub import load_model  # noqa: PLC0415
    from fairseq2.nn.projection import Linear  # noqa: PLC0415
    from fairseq2.runtime.dependency import (  # noqa: PLC0415
        DependencyContainer,
        get_dependency_resolver,
    )
    from omnilingual_asr.models.inference.pipeline import (  # noqa: PLC0415
        ASRInferencePipeline,
    )

    from p003_compact.omni_assets import (  # noqa: PLC0415
        load_ordered_vocab,
        omni_phoneme_assets,
    )

    resolver = get_dependency_resolver()
    if not isinstance(resolver, DependencyContainer):
        raise SystemExit("fairseq2 dependency resolver is not mutable.")
    assets = omni_phoneme_assets()
    register_file_assets(resolver, assets.card_dir)

    vocab = load_ordered_vocab()
    target_vocab_size = len(vocab)
    blank_index = vocab.index("[PAD]")

    model = load_model("omniASR_CTC_300M_v2", device=torch.device("cpu"))
    final_proj = model.final_proj
    replacement = Linear(
        final_proj.input_dim,
        target_vocab_size,
        bias=final_proj.bias is not None,
        init_fn=getattr(final_proj, "init_fn", None),
        device=final_proj.weight.device,
        dtype=final_proj.weight.dtype,
    )
    model.final_proj = replacement

    checkpoint_file = _resolve_latest_model_file(_resolve_checkpoint_dir(run_dir))
    state_dict = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict, strict=True)

    tokenizer = load_tokenizer("omniASR_tokenizer_arpabet_41_v1")
    pipeline = ASRInferencePipeline(
        model_card=None,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,
    )
    return pipeline, vocab, blank_index


def _load_audio_batch(wav_paths: list[str]) -> list[tuple[torch.Tensor, None]]:
    import torchaudio.functional as audio_functional  # noqa: PLC0415

    wavs_langs: list[tuple[torch.Tensor, None]] = []
    for wav_path in wav_paths:
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=-1)
        waveform = torch.from_numpy(np.ascontiguousarray(array))
        if sample_rate != EXPECTED_SAMPLE_RATE:
            waveform = audio_functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                EXPECTED_SAMPLE_RATE,
            ).squeeze(0)
        wavs_langs.append((waveform, None))
    return wavs_langs


def _infer_to_npz(pipeline: Any, wav_paths: list[str], output_path: Path) -> None:
    from fairseq2.nn.batch_layout import BatchLayout  # noqa: PLC0415

    wavs_langs = _load_audio_batch(wav_paths)
    batch = pipeline._create_batch_simple(wavs_langs)  # noqa: SLF001

    with torch.inference_mode():
        batch_layout = BatchLayout(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )
        logits, output_layout = pipeline.model(batch.source_seqs, batch_layout)
        posteriors = logits.softmax(dim=-1).float().cpu().numpy(force=True)
        lengths = output_layout.seq_lens.to("cpu").tolist()

    arrays = {
        f"p{index}": posteriors[index, : int(length), :]
        for index, length in enumerate(lengths)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _validate_wav_paths(wav_paths: object) -> list[str]:
    if not isinstance(wav_paths, list) or not all(
        isinstance(item, str) for item in wav_paths
    ):
        raise TypeError("wav_paths must be a list[str]")
    return wav_paths


def main() -> int:
    args = parse_args()
    _inject_paths()
    _preload_tbb()

    pipeline, vocab, blank_index = _load_pipeline(args.run_dir, args.device)
    _emit(
        {
            "status": "ready",
            "backend": "omni-ctc",
            "device": str(pipeline.device),
            "vocab_size": len(vocab),
            "blank_index": blank_index,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        command = request.get("command")
        request_id = request.get("request_id")
        if command == "shutdown":
            _emit({"status": "bye", "request_id": request_id})
            return 0
        if command != "infer":
            _emit(
                {
                    "status": "error",
                    "request_id": request_id,
                    "message": f"unknown command: {command!r}",
                }
            )
            continue

        try:
            wav_paths = _validate_wav_paths(request["wav_paths"])
            output_path = Path(request["output_path"])
            _infer_to_npz(pipeline, wav_paths, output_path)
        except Exception as exc:  # noqa: BLE001
            _emit(
                {
                    "status": "error",
                    "request_id": request_id,
                    "message": repr(exc),
                }
            )
            continue

        _emit(
            {
                "status": "ok",
                "request_id": request_id,
                "output_path": str(output_path),
                "count": len(wav_paths),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
