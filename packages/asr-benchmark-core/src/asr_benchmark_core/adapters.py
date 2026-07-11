from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import soundfile as sf

if TYPE_CHECKING:
    from pathlib import Path

    from asr_benchmark_core.data import Example


class Adapter(ABC):
    @abstractmethod
    def transcribe(self, example: Example) -> str: ...

    def transcribe_batch(self, examples: list[Example]) -> list[str]:
        return [self.transcribe(example) for example in examples]


class WhisperAdapter(Adapter):
    def __init__(self, model_path: Path, *, language: str, device: str) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.language = language
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            use_safetensors=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = model.eval()
        self.processor = processor
        self.dtype = dtype

    def transcribe(self, example: Example) -> str:
        import torch

        inputs = self.processor(
            example.audio,
            sampling_rate=example.sampling_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device, self.dtype) for key, value in inputs.items()}
        language = "fa" if self.language.lower() in {"persian", "farsi", "fa"} else self.language
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, language=language, task="transcribe")
        return str(self.processor.batch_decode(output_ids, skip_special_tokens=True)[0])

    def transcribe_batch(self, examples: list[Example]) -> list[str]:
        import torch

        inputs = self.processor(
            [example.audio for example in examples],
            sampling_rate=examples[0].sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device, self.dtype) for key, value in inputs.items()}
        language = "fa" if self.language.lower() in {"persian", "farsi", "fa"} else self.language
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, language=language, task="transcribe")
        return [
            str(text)
            for text in self.processor.batch_decode(output_ids, skip_special_tokens=True)
        ]


class QwenAdapter(Adapter):
    def __init__(self, model_path: Path, *, language: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.language = language
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = (
            Qwen3ASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

    def transcribe(self, example: Example) -> str:
        import torch

        # Passing the decoded array avoids the optional TorchCodec/FFmpeg path used for filenames.
        inputs = self.processor.apply_transcription_request(
            audio=example.audio,
            language=self.language,
        ).to(self.device, self.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return str(self.processor.decode(generated, return_format="transcription_only")[0])

    def transcribe_batch(self, examples: list[Example]) -> list[str]:
        import torch

        inputs = self.processor.apply_transcription_request(
            audio=[example.audio for example in examples],
            language=[self.language] * len(examples),
        ).to(self.device, self.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return [
            str(text)
            for text in self.processor.decode(generated, return_format="transcription_only")
        ]


class OmniAdapter(Adapter):
    """Adapter for registered Peacock Farsi and reusable local Omni checkpoints."""

    def __init__(self, model_path: Path, *, language: str, device: str) -> None:
        # Importing the project extension registers the local fairseq2 model card.  The
        # benchmark path is retained in the run metadata; the card currently points at the
        # same verified checkpoint under projects/farsi-asr/data/benchmarks/model/model.pt.
        import farsi_asr.assets  # noqa: F401
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        local_cards = {
            "omniASR-CTC-300M-v2.pt": "peacock_omniASR_CTC_300M_v2",
            "omniASR-CTC-1B-v2.pt": "peacock_omniASR_CTC_1B_v2",
            "omniASR-CTC-3B-v2.pt": "peacock_omniASR_CTC_3B_v2",
            "omniASR-LLM-300M-v2.pt": "peacock_omniASR_LLM_300M_v2",
            "omniASR-LLM-1B-v2.pt": "peacock_omniASR_LLM_1B_v2",
            "omniASR-LLM-3B-v2.pt": "peacock_omniASR_LLM_3B_v2",
        }
        model_card = local_cards.get(model_path.name, "omni_ctc_300m_farsi_hf")
        self.language = language
        self.pipe = ASRInferencePipeline(
            model_card,
            device=device,
            dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )

    def transcribe(self, example: Example) -> str:
        stream = io.BytesIO()
        sf.write(stream, example.audio, example.sampling_rate, format="WAV")
        return str(
            self.pipe.transcribe(
                [stream.getvalue()],
                lang=[self.language if self.language != "Persian" else "fas_Arab"],
                batch_size=1,
            )[0]
        )

    def transcribe_batch(self, examples: list[Example]) -> list[str]:
        audio: list[bytes] = []
        for example in examples:
            stream = io.BytesIO()
            sf.write(stream, example.audio, example.sampling_rate, format="WAV")
            audio.append(stream.getvalue())
        return [
            str(text)
            for text in self.pipe.transcribe(
                audio,
                lang=[self.language if self.language != "Persian" else "fas_Arab"] * len(examples),
                batch_size=len(examples),
            )
        ]


def load_adapter(name: str, model_path: Path, *, language: str, device: str) -> Adapter:
    if name == "whisper":
        return WhisperAdapter(model_path, language=language, device=device)
    if name == "qwen":
        return QwenAdapter(model_path, language=language, device=device)
    if name == "omni":
        return OmniAdapter(model_path, language=language, device=device)
    raise ValueError(f"unknown adapter: {name}")
