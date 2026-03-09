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
"""Launch OmniASR CTC 300M v2 phoneme fine-tuning with local assets."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from textwrap import dedent

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
            "Missing `tbb` runtime. Launch via the wrapper or run with "
            "`uv run --python 3.12 --with tbb>=2021.8 --with-editable "
            "<omnilingual-asr-path> ...`."
        ) from exc

    lib_path = tbb_dist.locate_file("../../libtbb.so.12")
    ctypes.CDLL(str(lib_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "checkpoints"
            / "omniasr-ctc-300m-v2-phoneme-en"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _run_builder(script: Path) -> None:
    subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
    )


def _write_config(path: Path, *, num_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_text = dedent(
        f"""
        model:
          name: "omniASR_CTC_300M_v2"

        dataset:
          name: "peacock_librispeech_phone_manifest_v1"
          train_split: "train"
          valid_split: "dev"
          storage_mode: "MANIFEST"
          task_mode: "ASR"
          manifest_storage_config:
            read_text: true
            cached_fd_count: 1000
          asr_task_config:
            min_audio_len: 32000
            max_audio_len: 960000
            max_num_elements: 960000
            batch_shuffle_window: 1
            example_shuffle_window: 1
            normalize_audio: true

        tokenizer:
          name: "omniASR_tokenizer_arpabet_41_v1"

        optimizer:
          config:
            lr: 1e-05

        trainer:
          freeze_encoder_for_n_steps: 0
          mixed_precision:
            dtype: "torch.bfloat16"
          grad_accumulation:
            num_batches: 4

        regime:
          num_steps: {num_steps}
          validate_every_n_steps: 500
          validate_after_n_steps: 500
          checkpoint_every_n_steps: 500
          checkpoint_after_n_steps: 500
          publish_metrics_every_n_steps: 500
          publish_metrics_after_n_steps: 500
        """
    ).strip()
    path.write_text(config_text + "\n", encoding="utf-8")


def _run_check() -> int:
    from fairseq2.composition.assets import register_file_assets  # noqa: PLC0415
    from fairseq2.data.tokenizers.hub import load_tokenizer  # noqa: PLC0415
    from fairseq2.models.hub import load_model  # noqa: PLC0415
    from fairseq2.nn.projection import Linear  # noqa: PLC0415
    from fairseq2.runtime.dependency import (  # noqa: PLC0415
        DependencyContainer,
        get_dependency_resolver,
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

    tokenizer = load_tokenizer("omniASR_tokenizer_arpabet_41_v1")
    model = load_model("omniASR_CTC_300M_v2", device=torch.device("cpu"))
    target_vocab_size = len(load_ordered_vocab())
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
    dataset_card = assets.card_dir / "dataset_librispeech_phone_manifest_v1.yaml"
    result = {
        "dataset_card": str(dataset_card.resolve()),
        "tokenizer_vocab_size": int(tokenizer.vocab_info.size),
        "target_vocab_size": target_vocab_size,
        "final_proj_weight_shape": list(model.final_proj.weight.shape),
        "pipeline_ready": True,
    }
    sys.stdout.write(f"{json.dumps(result, indent=2)}\n")
    return 0


def _run_train(output_dir: Path, config_path: Path) -> int:
    from fairseq2.composition.assets import register_file_assets  # noqa: PLC0415
    from fairseq2.nn.projection import Linear  # noqa: PLC0415
    from fairseq2.recipe.cli import train_main  # noqa: PLC0415
    from fairseq2.runtime.dependency import (  # noqa: PLC0415
        DependencyContainer,
        get_dependency_resolver,
    )
    from workflows.recipes.wav2vec2.asr.recipe import Wav2Vec2AsrRecipe  # noqa: PLC0415

    from p003_compact.omni_assets import (  # noqa: PLC0415
        load_ordered_vocab,
        omni_phoneme_assets,
    )

    class OmniPhonemeFineTuneRecipe(Wav2Vec2AsrRecipe):
        def prepare_model(self, context, model):  # type: ignore[override]
            model = super().prepare_model(context, model)
            asr_module = model.module
            target_vocab_size = len(load_ordered_vocab())
            final_proj = asr_module.final_proj
            if int(final_proj.weight.shape[0]) != target_vocab_size:
                replacement = Linear(
                    final_proj.input_dim,
                    target_vocab_size,
                    bias=final_proj.bias is not None,
                    init_fn=getattr(final_proj, "init_fn", None),
                    device=final_proj.weight.device,
                    dtype=final_proj.weight.dtype,
                )
                asr_module.final_proj = replacement
            return model

    resolver = get_dependency_resolver()
    if not isinstance(resolver, DependencyContainer):
        raise SystemExit("fairseq2 dependency resolver is not mutable.")
    assets = omni_phoneme_assets()
    register_file_assets(resolver, assets.card_dir)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "omni-phoneme-train",
            str(output_dir),
            "--config-file",
            str(config_path),
        ]
        train_main(OmniPhonemeFineTuneRecipe())
    finally:
        sys.argv = old_argv
    return 0


def main() -> int:
    args = parse_args()
    _inject_paths()
    _preload_tbb()

    _run_builder(PROJECT_ROOT / "code" / "build_omniasr_ctc_300m_v2_phoneme_assets.py")
    _run_builder(
        PROJECT_ROOT / "code" / "build_omniasr_ctc_300m_v2_phoneme_manifest.py"
    )

    config_path = (
        PROJECT_ROOT
        / "code"
        / "omni"
        / "configs"
        / "omniasr_ctc_300m_v2_phoneme_finetune_local.yaml"
    )
    _write_config(config_path, num_steps=args.num_steps)

    if args.check_only:
        return _run_check()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return _run_train(args.output_dir.resolve(), config_path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
