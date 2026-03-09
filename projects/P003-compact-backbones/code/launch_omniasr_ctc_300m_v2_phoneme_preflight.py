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
# ]
# ///
"""Preflight the OmniASR 300M v2 phoneme adaptation assets."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import torch


def _inject_omnilingual_src(repo_root: Path) -> None:
    omni_root = (
        repo_root
        / "projects"
        / "P004-training-from-scratch"
        / "third_party"
        / "omnilingual-asr"
    )
    omni_src = omni_root / "src"
    for candidate in (omni_root, omni_src):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _preload_tbb() -> None:
    try:
        tbb_dist = distribution("tbb")
    except PackageNotFoundError as exc:
        raise SystemExit(
            "Missing `tbb` runtime. Launch via the wrapper or run with "
            "`uv run --python 3.12 --with tbb>=2021.8 --with-editable "
            "<omnilingual-asr-path> ...`."
        ) from exc

    lib_path = tbb_dist.locate_file("../../libtbb.so.12")
    ctypes.CDLL(str(lib_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(
            "/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/omni/cards"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path("/home/simon/github/peacock-asr")
    _inject_omnilingual_src(repo_root)
    _preload_tbb()

    from fairseq2.assets.card import AssetCardError  # noqa: PLC0415
    from fairseq2.composition.assets import register_file_assets  # noqa: PLC0415
    from fairseq2.data.tokenizers.hub import load_tokenizer  # noqa: PLC0415
    from fairseq2.model_checkpoint.loader import ModelCheckpointError  # noqa: PLC0415
    from fairseq2.models.hub import load_model  # noqa: PLC0415
    from fairseq2.nn.projection import Linear  # noqa: PLC0415
    from fairseq2.runtime.dependency import (  # noqa: PLC0415
        DependencyContainer,
        get_dependency_resolver,
    )
    from omnilingual_asr.models.inference.pipeline import (  # noqa: PLC0415
        ASRInferencePipeline,
    )

    resolver = get_dependency_resolver()
    if not isinstance(resolver, DependencyContainer):
        msg = "fairseq2 dependency resolver is not a mutable container."
        raise SystemExit(msg)

    register_file_assets(resolver, args.assets_dir)

    result: dict[str, Any] = {
        "model_name": "omniASR_CTC_300M_v2_arpabet_41",
        "assets_dir": str(args.assets_dir.resolve()),
    }

    tokenizer = load_tokenizer("omniASR_CTC_300M_v2_arpabet_41")
    result["tokenizer_loaded"] = True
    result["tokenizer_vocab_size"] = int(tokenizer.vocab_info.size)

    device = None if args.device is None else torch.device(args.device)
    try:
        model = load_model(
            "omniASR_CTC_300M_v2_arpabet_41",
            device=device,
        )
        result["load_mode"] = "custom-card"
    except (AssetCardError, ModelCheckpointError) as exc:
        result["custom_card_error"] = str(exc)
        model = load_model(
            "omniASR_CTC_300M_v2",
            device=device,
        )
        final_proj = model.final_proj
        replacement = Linear(
            final_proj.input_dim,
            int(tokenizer.vocab_info.size),
            bias=final_proj.bias is not None,
            init_fn=getattr(final_proj, "init_fn", None),
            device=final_proj.weight.device,
            dtype=final_proj.weight.dtype,
        )
        model.final_proj = replacement
        result["load_mode"] = "stock-plus-head-replacement"

    pipeline = ASRInferencePipeline(
        model_card=None,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    final_proj = getattr(model, "final_proj", None)
    result["model_loaded"] = True
    result["device"] = str(next(model.parameters()).device)
    result["final_proj_weight_shape"] = (
        list(final_proj.weight.shape) if final_proj is not None else None
    )
    result["pipeline_ready"] = True
    result["pipeline_device"] = str(pipeline.device)
    sys.stdout.write(f"{json.dumps(result, indent=2)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
