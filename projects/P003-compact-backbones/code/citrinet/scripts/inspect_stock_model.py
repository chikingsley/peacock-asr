#!/usr/bin/env python3
"""Inspect the stock Citrinet checkpoint without touching the main P003 path."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _load_nemo_asr() -> Any:
    try:
        return importlib.import_module("nemo.collections.asr")
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        msg = (
            "NeMo ASR is not installed in this environment. "
            "Use the isolated Citrinet env under "
            "projects/P003-compact-backbones/env/citrinet/."
        )
        raise SystemExit(msg) from exc


def _safe_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return getattr(obj, name, None)


def _tokenizer_info(model: Any) -> dict[str, Any]:
    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_dir = None
    tokenizer_vocab_size = None
    if tokenizer is not None:
        artifact = _safe_attr(tokenizer, "tokenizer")
        tokenizer_dir = _safe_attr(artifact, "model_path")
        tokenizer_vocab_size = _safe_attr(tokenizer, "vocab_size")
        if tokenizer_vocab_size is None:
            vocab = _safe_attr(tokenizer, "vocab")
            if vocab is not None and hasattr(vocab, "__len__"):
                tokenizer_vocab_size = len(vocab)

    return {
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_dir": tokenizer_dir,
        "labels_len": len(getattr(model, "decoder", {}).vocabulary)
        if getattr(model, "decoder", None) is not None
        and hasattr(model.decoder, "vocabulary")
        else None,
    }


def _config_info(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "cfg", None)
    preprocessor = _safe_attr(cfg, "preprocessor")
    encoder = _safe_attr(cfg, "encoder")
    decoder = _safe_attr(cfg, "decoder")

    return {
        "model_class": type(model).__name__,
        "sample_rate": _safe_attr(preprocessor, "sample_rate"),
        "window_stride": _safe_attr(preprocessor, "window_stride"),
        "features": _safe_attr(preprocessor, "features"),
        "encoder_type": _safe_attr(encoder, "_target_"),
        "decoder_type": _safe_attr(decoder, "_target_"),
        "num_classes": _safe_attr(decoder, "num_classes"),
    }


def _inspect_audio(model: Any, audio_path: Path) -> dict[str, Any]:
    hyps = model.transcribe(
        [str(audio_path)],
        batch_size=1,
        return_hypotheses=True,
    )
    hyp = hyps[0]
    alignments = getattr(hyp, "alignments", None)

    result: dict[str, Any] = {
        "audio_path": str(audio_path),
        "transcript": getattr(hyp, "text", None),
        "alignment_len": None,
        "alignment_inner_len": None,
    }
    if alignments is not None:
        try:
            result["alignment_len"] = len(alignments)
            if len(alignments) > 0 and hasattr(alignments[0], "__len__"):
                result["alignment_inner_len"] = len(alignments[0])
        except TypeError:
            pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a stock NeMo Citrinet checkpoint and optional sample audio."
        )
    )
    parser.add_argument(
        "--model-name",
        default="nvidia/stt_en_citrinet_256_ls",
        help="Pretrained NeMo model name.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio file for a one-sample transcription/alignment check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON output.",
    )
    args = parser.parse_args()

    nemo_asr = _load_nemo_asr()
    model = nemo_asr.models.ASRModel.from_pretrained(args.model_name)
    model.eval()

    report: dict[str, Any] = {
        "model_name": args.model_name,
        "config": _config_info(model),
        "tokenizer": _tokenizer_info(model),
        "audio_probe": None,
    }
    if args.audio is not None:
        report["audio_probe"] = _inspect_audio(model, args.audio)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        sys.stdout.write(payload + "\n")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(f"{args.output}\n")


if __name__ == "__main__":
    main()
