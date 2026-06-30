from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import BatchEncoding
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor


@dataclass
class MelConfig:
    mel_sr: int = 16_000
    mel_dim: int = 128
    mel_n_fft: int = 400
    mel_hop_length: int = 160
    mel_dtype: torch.dtype = torch.bfloat16


def load_chat_template(template_path: str | Path) -> list[Any]:
    path = Path(template_path)
    spec = importlib.util.spec_from_file_location("moss_chat_template_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load chat template from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moss_chat_template_module"] = module
    spec.loader.exec_module(module)
    return module.chat_template


class MossProcessor:
    def __init__(
        self,
        tokenizer: Any,
        config: MelConfig | None = None,
        template_path: str | Path | None = None,
        *,
        enable_time_marker: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or MelConfig()
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=int(self.config.mel_dim),
            sampling_rate=int(self.config.mel_sr),
            hop_length=int(self.config.mel_hop_length),
            n_fft=int(self.config.mel_n_fft),
        )

        self.start_token_id = 151644
        self.end_token_id = 151645
        self.audio_start_token_id = 151669
        self.audio_end_token_id = 151670
        self.audio_placeholder_id = 0
        self.chat_template = None if template_path is None else load_chat_template(template_path)
        self.enable_time_marker = enable_time_marker

        self._digit_token_ids = {str(digit): 15 + digit for digit in range(10)}
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )

    def load_template(self, template_path: str | Path) -> MossProcessor:
        self.chat_template = load_chat_template(template_path)
        return self

    def _get_feat_extract_output_lengths(self, input_lengths: int) -> int:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def _get_time_marker_token_ids(self, second: int) -> list[int]:
        return [self._digit_token_ids[char] for char in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> list[int]:
        num_full_seconds = int(audio_seq_len / self.audio_tokens_per_second)
        tokens: list[int] = []
        audio_tokens_consumed = 0

        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            segment_len = marker_pos - audio_tokens_consumed
            if segment_len > 0:
                tokens.extend([self.audio_placeholder_id] * segment_len)
                audio_tokens_consumed += segment_len
            tokens.extend(self._get_time_marker_token_ids(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            tokens.extend([self.audio_placeholder_id] * remaining)
        return tokens

    def _build_input_from_template(self, num_audio_tokens: int) -> tuple[list[int], list[bool]]:
        if self.chat_template is None:
            raise ValueError("Chat template not loaded. Call load_template() first.")

        input_ids: list[int] = []
        audio_mask: list[bool] = []

        for segment in self.chat_template:
            seg_type = segment.type
            if seg_type == "constant_text_token":
                text_ids = segment.text_ids.tolist()
                input_ids.extend(text_ids)
                audio_mask.extend([False] * len(text_ids))
            elif seg_type in {"audio_contiguous", "audio_token"}:
                if self.enable_time_marker:
                    audio_ids = self._build_audio_tokens_with_time_markers(num_audio_tokens)
                    input_ids.extend(audio_ids)
                    audio_mask.extend([token == self.audio_placeholder_id for token in audio_ids])
                else:
                    input_ids.extend([self.audio_placeholder_id] * num_audio_tokens)
                    audio_mask.extend([True] * num_audio_tokens)
            elif seg_type == "text_token":
                break

        return input_ids, audio_mask

    def _build_input_legacy(self, num_audio_tokens: int) -> tuple[list[int], list[bool]]:
        if self.enable_time_marker:
            audio_ids = self._build_audio_tokens_with_time_markers(num_audio_tokens)
            ids = [
                self.start_token_id,
                self.audio_start_token_id,
                *audio_ids,
                self.audio_end_token_id,
            ]
            audio_mask = [token == self.audio_placeholder_id for token in audio_ids]
            return ids, [False, False, *audio_mask, False]

        ids = [
            self.start_token_id,
            self.audio_start_token_id,
            *([self.audio_placeholder_id] * num_audio_tokens),
            self.audio_end_token_id,
        ]
        return ids, [False, False, *([True] * num_audio_tokens), False]

    def __call__(
        self,
        audio: np.ndarray | torch.Tensor,
        return_tensors: str = "pt",
        **_: Any,
    ) -> BatchEncoding:
        if audio is None:
            raise ValueError("Audio input is required.")

        if isinstance(audio, torch.Tensor):
            waveform = audio.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform[0]

        mel = self.feature_extractor._np_extract_fbank_features(
            waveform[None, ...],
            device="cpu",
        )[0]

        input_features = torch.from_numpy(mel).to(self.config.mel_dtype)
        if input_features.dim() == 3:
            input_features = input_features.squeeze(0)

        raw_mel_len = input_features.shape[-1]
        num_audio_tokens = self._get_feat_extract_output_lengths(raw_mel_len)

        if self.chat_template is not None:
            ids, mask = self._build_input_from_template(num_audio_tokens)
        else:
            ids, mask = self._build_input_legacy(num_audio_tokens)

        input_ids_tensor = torch.tensor([ids], dtype=torch.long)
        audio_mask_tensor = torch.tensor([mask], dtype=torch.bool)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)
        seq_lens_tensor = torch.tensor([raw_mel_len], dtype=torch.long)

        return BatchEncoding(
            data={
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "audio_data": input_features,
                "audio_data_seqlens": seq_lens_tensor,
                "audio_input_mask": audio_mask_tensor,
            },
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)
