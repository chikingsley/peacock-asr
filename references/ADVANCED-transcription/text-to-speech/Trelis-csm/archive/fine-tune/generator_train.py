from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        # Remove setup_caches since we don't need KV caching for training
        # self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            
        Returns:
            (batch_size, seq_len, vocab_size) logits
        """
        dtype = next(self._model.parameters()).dtype
        b, s, _ = tokens.size()

        # Create causal masks for the current batch size
        # Shape: (batch_size, 3, num_heads, seq_len, seq_len)
        backbone_mask = torch.tril(torch.ones(b, 3, 32, s, s, dtype=torch.bool, device=tokens.device))  # 32 heads
        decoder_mask = torch.tril(torch.ones(b, 3, 8, s, s, dtype=torch.bool, device=tokens.device))     # 8 heads
        
        # Embed tokens
        embeds = self._model._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        
        # Backbone forward pass
        h = self._model.backbone(h, input_pos=input_pos, mask=backbone_mask).to(dtype=dtype)
        
        # Project to decoder dimension
        h = self._model.projection(h)
        
        # Decoder forward pass
        h = self._model.decoder(h, input_pos=input_pos, mask=decoder_mask).to(dtype=dtype)
        
        # Project to vocabulary
        logits = torch.matmul(h, self._model.audio_head.transpose(-2, -1))
        
        return logits


def load_csm_1b(device: str = "cuda") -> Generator:
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    print(f"Model: {model}")
    
    # Calculate and print model size in GB
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    print(f"Model size: {model_size_gb:.2f} GB")

    generator = Generator(model)
    return generator 