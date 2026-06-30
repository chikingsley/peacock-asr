from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer

from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.dump import ensure_dir, write_json
from moss_mlx_conversion.mlx_compat import mx, require_mlx
from moss_mlx_conversion.model.moss import MossMLXModel
from moss_mlx_conversion.paths import MLX_DIR
from moss_mlx_conversion.processor import MossProcessor
from moss_mlx_conversion.runtime.audio import load_waveform


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    generated_ids: list[int]
    prompt_length: int
    audio_placeholder_count: int
    elapsed_sec: float
    generation_elapsed_sec: float

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a converted MOSS MLX smoke transcription.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MLX_DIR / "MOSS-Transcribe-preview-2B-bf16",
    )
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/mlx-smoke/smoke-report.json"),
    )
    parser.add_argument("--reference-report", type=Path)
    return parser.parse_args()


def load_converted_model(model_dir: Path) -> tuple[MossMLXModel, MossModelConfig]:
    require_mlx()
    config = MossModelConfig.from_json(model_dir / "config.json")
    model = MossMLXModel(config)
    weights = mx.load(str(model_dir / "weights.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config


def _to_mx_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    input_ids = mx.array(batch["input_ids"].numpy(), dtype=mx.int32)
    audio_input_mask = mx.array(batch["audio_input_mask"].numpy(), dtype=mx.bool_)
    audio_data = mx.array(batch["audio_data"].float().numpy(), dtype=mx.bfloat16)
    if audio_data.ndim == 2:
        audio_data = audio_data[None, :, :]
    audio_data_seqlens = mx.array(batch["audio_data_seqlens"].numpy(), dtype=mx.int32)
    return {
        "input_ids": input_ids,
        "audio_input_mask": audio_input_mask,
        "audio_data": audio_data,
        "audio_data_seqlens": audio_data_seqlens,
    }


def build_prompt_embeddings(
    model: MossMLXModel,
    batch: dict[str, Any],
) -> tuple[Any, Any]:
    mlx_inputs = _to_mx_inputs(batch)
    audio_embeds = model.get_audio_features(
        mlx_inputs["audio_data"],
        mlx_inputs["audio_data_seqlens"],
    )
    mx.eval(audio_embeds)
    inputs_embeds = model.build_inputs_embeds(
        mlx_inputs["input_ids"],
        audio_embeds,
        mlx_inputs["audio_input_mask"],
    )
    mx.eval(inputs_embeds)
    return mlx_inputs["input_ids"][0], inputs_embeds[0]


def load_reference(reference_report: Path | None) -> dict[str, Any] | None:
    if reference_report is None or not reference_report.exists():
        return None
    return json.loads(reference_report.read_text(encoding="utf-8"))


def transcribe_waveform(
    *,
    model: MossMLXModel,
    config: MossModelConfig,
    processor: MossProcessor,
    tokenizer: Any,
    waveform: Any,
    max_new_tokens: int,
    prefill_step_size: int,
) -> TranscriptionResult:
    started = time.perf_counter()
    batch = cast("dict[str, Any]", dict(processor(audio=waveform, return_tensors="pt")))
    prompt, input_embeddings = build_prompt_embeddings(model, batch)

    generate_module: Any = import_module("mlx_lm.generate")
    generate_step = generate_module.generate_step

    generated_ids: list[int] = []
    eos_token_ids = {config.end_token_id, tokenizer.eos_token_id}
    generation_started = time.perf_counter()
    for token, _logprobs in generate_step(
        prompt=prompt,
        input_embeddings=input_embeddings,
        model=model,
        max_tokens=max_new_tokens,
        prefill_step_size=prefill_step_size,
    ):
        token_id = int(token)
        if token_id in eos_token_ids:
            break
        generated_ids.append(token_id)

    transcript = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return TranscriptionResult(
        transcript=transcript,
        generated_ids=generated_ids,
        prompt_length=int(batch["input_ids"].shape[1]),
        audio_placeholder_count=int(batch["audio_input_mask"].sum().item()),
        elapsed_sec=time.perf_counter() - started,
        generation_elapsed_sec=time.perf_counter() - generation_started,
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    model_dir = args.model_dir.resolve()
    report_path = args.report
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path

    model, config = load_converted_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    processor = MossProcessor(
        tokenizer,
        template_path=model_dir / "chat_template_default.py",
        enable_time_marker=False,
    )

    waveform, audio_path = load_waveform(args.audio, sample_rate=config.sample_rate)
    result = transcribe_waveform(
        model=model,
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        waveform=waveform,
        max_new_tokens=args.max_new_tokens,
        prefill_step_size=args.prefill_step_size,
    )
    reference = load_reference(args.reference_report)
    reference_generation = None if reference is None else reference.get("generation", {})
    report = {
        "model_dir": str(model_dir),
        "audio_path": str(audio_path),
        "prompt_length": result.prompt_length,
        "audio_placeholder_count": result.audio_placeholder_count,
        "max_new_tokens": args.max_new_tokens,
        "generated_token_count": result.generated_token_count,
        "first_5_new_ids": result.generated_ids[:5],
        "transcript": result.transcript,
        "elapsed_sec": time.perf_counter() - started,
        "generation_elapsed_sec": result.generation_elapsed_sec,
        "reference_transcript": None
        if reference_generation is None
        else reference_generation.get("transcript"),
        "reference_first_5_new_ids": None
        if reference_generation is None
        else reference_generation.get("first_5_new_ids"),
        "matches_reference_transcript": None
        if reference_generation is None
        else result.transcript == reference_generation.get("transcript"),
        "matches_reference_first_5_new_ids": None
        if reference_generation is None
        else result.generated_ids[:5] == reference_generation.get("first_5_new_ids"),
    }
    ensure_dir(report_path.parent)
    write_json(report_path, report)
    print(result.transcript)
    print(f"MLX smoke report: {report_path}")


if __name__ == "__main__":
    main()
