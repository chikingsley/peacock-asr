from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.dump import ensure_dir, save_npz, tensor_stats, topk_summary, write_json
from moss_mlx_conversion.paths import REFERENCE_DIR
from moss_mlx_conversion.reference.hf import (
    download_template,
    hf_cache_dir,
    load_remote_processor_classes,
    load_tokenizer,
)
from moss_mlx_conversion.runtime.audio import load_waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run upstream MOSS and dump parity tensors.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--dump-dir", type=Path, default=REFERENCE_DIR / "smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--save-large-tensors", action="store_true")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def pick_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    return torch.float32


def move_inputs_to_device(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def build_processor(
    model_id: str,
    *,
    revision: str,
    local_files_only: bool,
) -> tuple[Any, Any, Path]:
    tokenizer = load_tokenizer(model_id, revision=revision, local_files_only=local_files_only)
    processor_cls, mel_config_cls = load_remote_processor_classes(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    mel_cfg = mel_config_cls(
        mel_sr=16_000,
        mel_dim=128,
        mel_n_fft=400,
        mel_hop_length=160,
    )
    processor = processor_cls(tokenizer, config=mel_cfg, enable_time_marker=False)
    template_path = download_template(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    processor.load_template(str(template_path))
    return tokenizer, processor, template_path


def load_model(
    model_id: str,
    *,
    revision: str,
    dtype: torch.dtype,
    device: torch.device,
    local_files_only: bool,
) -> Any:
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
        cache_dir=hf_cache_dir(),
        local_files_only=local_files_only,
    )
    model = model.to(device)
    model.eval()
    return model


def score_summaries(
    scores: tuple[torch.Tensor, ...],
    *,
    max_steps: int = 5,
) -> list[dict[str, Any]]:
    summaries = []
    for step, score in enumerate(scores[:max_steps]):
        summaries.append({"step": step, "topk": topk_summary(score[0], k=10)})
    return summaries


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dump_dir = ensure_dir(args.dump_dir)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    waveform, audio_path = load_waveform(args.audio)
    tokenizer, processor, template_path = build_processor(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    del tokenizer

    inputs_cpu = processor(audio=waveform, return_tensors="pt")
    inputs = move_inputs_to_device(dict(inputs_cpu), device)
    model = load_model(
        args.model_id,
        revision=args.revision,
        dtype=dtype,
        device=device,
        local_files_only=args.local_files_only,
    )
    inputs["audio_data"] = inputs["audio_data"].to(model.dtype)

    report: dict[str, Any] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "audio_path": str(audio_path),
        "template_path": str(template_path),
        "device": str(device),
        "model_dtype": str(model.dtype),
        "prompt_length": int(inputs["input_ids"].shape[1]),
        "input_stats": {
            "waveform": tensor_stats(waveform),
            "input_ids": tensor_stats(inputs_cpu["input_ids"]),
            "attention_mask": tensor_stats(inputs_cpu["attention_mask"]),
            "audio_input_mask": tensor_stats(inputs_cpu["audio_input_mask"]),
            "audio_data": tensor_stats(inputs_cpu["audio_data"]),
            "audio_data_seqlens": tensor_stats(inputs_cpu["audio_data_seqlens"]),
        },
    }

    with torch.inference_mode():
        text_embeds = model.model.get_input_embeddings()(inputs["input_ids"])
        audio_hidden = model.model.get_audio_features(
            inputs["audio_data"],
            inputs["audio_data_seqlens"],
        )
        audio_embeds = model.model.audio_adapter(audio_hidden)
        merged_embeds = text_embeds.clone()
        mask_expanded = inputs["audio_input_mask"].unsqueeze(-1).expand_as(merged_embeds)
        merged_embeds.masked_scatter_(mask_expanded, audio_embeds)

        prefill = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
        prefill_last_logits = prefill.logits[:, -1, :].detach()

        report["component_stats"] = {
            "text_embeds": tensor_stats(text_embeds),
            "audio_hidden": tensor_stats(audio_hidden),
            "audio_embeds": tensor_stats(audio_embeds),
            "merged_embeds": tensor_stats(merged_embeds),
            "prefill_last_logits": tensor_stats(prefill_last_logits),
            "prefill_last_logits_topk": topk_summary(prefill_last_logits[0], k=10),
        }

        arrays_to_save: dict[str, torch.Tensor] = {
            "input_ids": inputs_cpu["input_ids"],
            "attention_mask": inputs_cpu["attention_mask"],
            "audio_input_mask": inputs_cpu["audio_input_mask"],
            "audio_data_seqlens": inputs_cpu["audio_data_seqlens"],
            "prefill_last_logits": prefill_last_logits.cpu(),
        }
        if args.save_large_tensors:
            arrays_to_save.update(
                {
                    "audio_data": inputs_cpu["audio_data"],
                    "audio_hidden": audio_hidden.cpu(),
                    "audio_embeds": audio_embeds.cpu(),
                    "merged_embeds": merged_embeds.cpu(),
                }
            )

        if not args.skip_generate:
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=[processor.end_token_id],
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = generated.sequences
            new_ids = sequences[:, inputs["input_ids"].shape[1] :]
            transcript = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            report["generation"] = {
                "new_token_count": int(new_ids.shape[1]),
                "new_ids": new_ids.detach().cpu().reshape(-1).tolist(),
                "first_5_new_ids": new_ids.detach().cpu().reshape(-1)[:5].tolist(),
                "score_topk_first_5": score_summaries(generated.scores, max_steps=5),
                "transcript": transcript,
            }
            arrays_to_save["generated_ids"] = new_ids.cpu()
            print(transcript)

        save_npz(dump_dir / "reference_tensors.npz", **arrays_to_save)

    report["elapsed_sec"] = time.perf_counter() - started
    write_json(dump_dir / "reference_report.json", report)
    print(f"reference report: {dump_dir / 'reference_report.json'}")


if __name__ == "__main__":
    main()
