#!/usr/bin/env python3
"""
Kokoro Training Utilities

Provides a trainable forward pass for Kokoro TTS that allows gradients
to flow through the model. This is needed because the default
KModel.forward_with_tokens() has @torch.no_grad() decorator.
"""

import torch
from typing import Tuple


def trainable_forward(
    model,  # KModel instance
    input_ids: torch.LongTensor,
    ref_s: torch.FloatTensor,  # Voice embedding (batch, 256)
    speed: float = 1.0
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Forward pass WITH gradients (no @torch.no_grad).

    This mirrors KModel.forward_with_tokens() but without the decorator,
    allowing backpropagation through the voice embedding.

    Args:
        model: KModel instance with bert, bert_encoder, predictor, text_encoder, decoder
        input_ids: Phoneme token IDs (batch, seq_len)
        ref_s: Voice/style embedding (batch, 256)
               - First 128 dims: acoustic style (for decoder)
               - Second 128 dims: prosody style (for duration/F0)
        speed: Speech speed multiplier

    Returns:
        audio: Generated waveform
        pred_dur: Predicted durations per token
    """
    device = model.device

    # Input lengths
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long
    )

    # Create attention mask
    text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(
        input_lengths.shape[0], -1
    ).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

    # BERT encoding for duration
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    # Split style: prosody (second half) and acoustic (first half)
    s = ref_s[:, 128:]  # Prosody style for duration/F0

    # Duration prediction
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    # Handle single-element case
    if pred_dur.dim() == 0:
        pred_dur = pred_dur.unsqueeze(0)

    # Create alignment matrix
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=device), pred_dur
    )
    pred_aln_trg = torch.zeros(
        (input_ids.shape[1], indices.shape[0]), device=device
    )
    pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)

    # Expand duration-encoded features
    en = d.transpose(-1, -2) @ pred_aln_trg

    # F0 and noise prediction
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

    # Text encoding
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg

    # Decode to audio (use first 128 dims of style)
    audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

    return audio, pred_dur


def phonemes_to_ids(phonemes: str, vocab: dict) -> torch.LongTensor:
    """Convert phoneme string to token IDs.

    Args:
        phonemes: Phoneme string (e.g., "h ɛ l oʊ")
        vocab: Vocabulary dict mapping phoneme -> ID

    Returns:
        Token IDs with start/end tokens
    """
    ids = [vocab.get(p) for p in phonemes if vocab.get(p) is not None]
    # Add start and end tokens (both are 0)
    return torch.LongTensor([[0] + ids + [0]])


def get_g2p(lang_code: str = 'a'):
    """Get grapheme-to-phoneme converter for a language.

    Args:
        lang_code: 'a' for American English, 'b' for British, etc.

    Returns:
        G2P function that converts text to phonemes
    """
    from misaki import en
    g2p = en.G2P(trf=False, british=(lang_code == 'b'))

    def convert(text: str) -> str:
        _, tokens = g2p(text)
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()

    return convert


if __name__ == "__main__":
    # Test the trainable forward
    print("Testing trainable forward pass...")

    from kokoro import KModel
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading KModel...")
    model = KModel(repo_id='hexgrad/Kokoro-82M').to(device)
    model.eval()

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Create dummy voice embedding with gradients
    voice_emb = torch.randn(1, 256, device=device, requires_grad=True)

    # Test phonemes
    test_text = "Hello world"
    g2p = get_g2p('a')
    phonemes = g2p(test_text)
    print(f"Phonemes: {phonemes}")

    input_ids = phonemes_to_ids(phonemes, model.vocab).to(device)
    print(f"Input IDs shape: {input_ids.shape}")

    # Forward pass
    print("Running trainable forward...")
    audio, pred_dur = trainable_forward(model, input_ids, voice_emb)
    print(f"Audio shape: {audio.shape}")
    print(f"Durations: {pred_dur}")

    # Test backprop
    print("\nTesting backpropagation...")
    loss = audio.abs().mean()  # Dummy loss
    loss.backward()

    if voice_emb.grad is not None:
        grad_norm = voice_emb.grad.norm().item()
        print(f"Voice embedding gradient norm: {grad_norm:.6f}")
        if grad_norm > 0:
            print("SUCCESS: Gradients are flowing!")
        else:
            print("WARNING: Gradients are zero")
    else:
        print("ERROR: No gradients computed")
