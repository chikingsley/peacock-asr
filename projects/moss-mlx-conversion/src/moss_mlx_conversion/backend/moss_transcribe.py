from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.dump import write_json
from moss_mlx_conversion.paths import MLX_DIR
from moss_mlx_conversion.processor import MossProcessor
from moss_mlx_conversion.runtime.audio import load_waveform
from moss_mlx_conversion.runtime.transcribe import load_converted_model, transcribe_waveform


@dataclass(frozen=True)
class STTOutput:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = "English"
    total_time: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class MossTranscribeBackend:
    def __init__(
        self,
        *,
        model: Any,
        config: MossModelConfig,
        processor: MossProcessor,
        tokenizer: Any,
        model_dir: Path,
        max_new_tokens: int = 256,
        prefill_step_size: int = 512,
        generation_mode: str = "mlx-lm",
    ) -> None:
        self.model = model
        self.config = config
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.prefill_step_size = prefill_step_size
        self.generation_mode = generation_mode

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path = MLX_DIR / "MOSS-Transcribe-preview-2B-bf16",
        *,
        max_new_tokens: int = 256,
        prefill_step_size: int = 512,
        generation_mode: str = "mlx-lm",
    ) -> MossTranscribeBackend:
        resolved_model_dir = Path(model_dir).resolve()
        model, config = load_converted_model(resolved_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_dir), trust_remote_code=True)
        processor = MossProcessor(
            tokenizer,
            template_path=resolved_model_dir / "chat_template_default.py",
            enable_time_marker=False,
        )
        return cls(
            model=model,
            config=config,
            processor=processor,
            tokenizer=tokenizer,
            model_dir=resolved_model_dir,
            max_new_tokens=max_new_tokens,
            prefill_step_size=prefill_step_size,
            generation_mode=generation_mode,
        )

    def generate(
        self,
        audio: str | Path | np.ndarray,
        *,
        language: str = "English",
        max_new_tokens: int | None = None,
        verbose: bool = False,
    ) -> STTOutput:
        del verbose
        if isinstance(audio, np.ndarray):
            waveform = np.asarray(audio, dtype=np.float32)
        else:
            waveform, _audio_path = load_waveform(Path(audio), sample_rate=self.config.sample_rate)

        result = transcribe_waveform(
            model=self.model,
            config=self.config,
            processor=self.processor,
            tokenizer=self.tokenizer,
            waveform=waveform,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            prefill_step_size=self.prefill_step_size,
            generation_mode=self.generation_mode,
        )
        return STTOutput(
            text=result.transcript,
            language=language,
            total_time=result.elapsed_sec,
            prompt_tokens=result.prompt_length,
            generation_tokens=result.generated_token_count,
            timings=result.timings,
            raw={
                "generated_ids": result.generated_ids,
                "audio_placeholder_count": result.audio_placeholder_count,
                "generation_mode": result.generation_mode,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with the local MOSS MLX backend."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MLX_DIR / "MOSS-Transcribe-preview-2B-bf16",
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument(
        "--generation-mode",
        choices=["fast-greedy", "mlx-lm"],
        default="mlx-lm",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = MossTranscribeBackend.from_pretrained(
        args.model_dir,
        max_new_tokens=args.max_new_tokens,
        prefill_step_size=args.prefill_step_size,
        generation_mode=args.generation_mode,
    )
    output = backend.generate(args.audio)
    print(output.text)
    if args.output_json is not None:
        write_json(args.output_json, json.loads(json.dumps(asdict(output))))


if __name__ == "__main__":
    main()
