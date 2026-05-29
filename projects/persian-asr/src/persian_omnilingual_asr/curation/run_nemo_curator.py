from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from persian_asr_dataset.cli.export_nemo_manifest import NEMO_MODEL_CARD


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeMo Curator ASR/WER filtering.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=NEMO_MODEL_CARD)
    parser.add_argument("--decoder-type", choices=("ctc", "rnnt", "default"), default="ctc")
    parser.add_argument("--wer-threshold", type=float, default=35.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=float, default=1.0)
    return parser


def default_output_dir(manifest: Path) -> Path:
    if manifest.parent.name == "manifest":
        return manifest.parent.parent / "result"
    return manifest.parent / "result"


def run_pipeline(args: argparse.Namespace) -> None:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
    from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
    from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
    from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.stages.text.io.reader import JsonlReader
    from nemo_curator.stages.text.io.writer import JsonlWriter

    class DecoderInferenceAsrNemoStage(InferenceAsrNemoStage):
        def __init__(self, *stage_args, decoder_type: str = "ctc", **stage_kwargs) -> None:
            super().__init__(*stage_args, **stage_kwargs)
            self.decoder_type = decoder_type

        def setup(self, _worker_metadata: Any = None) -> None:
            super().setup(_worker_metadata)
            if self.decoder_type != "default":
                asr_model = cast("Any", self.asr_model)
                asr_model.change_decoding_strategy(
                    decoder_type=self.decoder_type,
                    verbose=False,
                )

    output_dir = args.output_dir or default_output_dir(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        name="nemo_fa_curation",
        description="Persian ASR curation with NVIDIA FastConformer and WER filtering",
    )
    pipeline.add_stage(
        JsonlReader(
            file_paths=str(args.manifest),
            fields=[
                "audio_filepath",
                "text",
                "duration",
                "language",
                "sample_rate",
                "sample_id",
                "source",
                "source_split",
                "model_card",
            ],
        )
    )
    pipeline.add_stage(
        DecoderInferenceAsrNemoStage(
            model_name=args.model_name,
            decoder_type=args.decoder_type,
            filepath_key="audio_filepath",
            pred_text_key="pred_text",
            resources=Resources(gpus=args.gpus),
            batch_size=args.batch_size,
        )
    )
    pipeline.add_stage(
        GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="wer")
    )
    pipeline.add_stage(
        GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
    )
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="duration", target_value=args.min_duration, operator="ge"
        )
    )
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="duration", target_value=args.max_duration, operator="le"
        )
    )
    pipeline.add_stage(
        PreserveByValueStage(input_value_key="wer", target_value=args.wer_threshold, operator="le")
    )
    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(path=str(output_dir), mode="overwrite"))
    pipeline.run()
    print(f"wrote {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
