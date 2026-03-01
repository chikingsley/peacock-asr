"""
Forced alignment using TorchAudio's MMS_FA model.
Maps ground truth text to audio to get word-level timestamps.

HOW IT WORKS:
=============

1. TEXT NORMALIZATION (with mapping):
   - MMS_FA only understands lowercase letters and apostrophes
   - "Phi-2 is fine-tuning!" → ["phi", "is", "fine", "tuning"]
   - We track mapping: original word → normalized word indices
     - "Phi-2" → [0]
     - "is" → [1]
     - "fine-tuning!" → [2, 3]  (hyphen splits into 2 words)

2. CHARACTER TOKENIZATION:
   - Each word is split into characters: "great" → ['g', 'r', 'e', 'a', 't']
   - The model aligns each character to audio frames

3. FRAME-LEVEL ALIGNMENT:
   - Audio is processed at 16kHz, with frames every 20ms (320 samples)
   - Model outputs "emissions" (probabilities for each character at each frame)
   - CTC aligner finds the best path matching characters to frames

4. REVERSE MAPPING:
   - Get timestamps for normalized words
   - Map back to original words by combining spans:
     - "fine": 1.1-1.3s, "tuning": 1.3-1.6s → "fine-tuning!": 1.1-1.6s

EXAMPLE:
   Audio: [someone saying "Phi-2 is great"]
   Text: "Phi-2 is great!"
   Normalized: ["phi", "is", "great"]
   Mapping: [[0], [1], [2]]
   Output: [
       {"word": "Phi-2", "start": 0.5, "end": 0.9},
       {"word": "is", "start": 1.0, "end": 1.1},
       {"word": "great!", "start": 1.2, "end": 1.5}
   ]
"""

import re
import torch
import torchaudio
from typing import List, Dict, Tuple, Optional

# Cache the model globally to avoid reloading
_fa_model = None
_fa_tokenizer = None
_fa_aligner = None
_fa_device = None


def _load_fa_model(device: str = "cpu"):
    """Load and cache the forced alignment model."""
    global _fa_model, _fa_tokenizer, _fa_aligner, _fa_device

    if _fa_model is not None and _fa_device == device:
        return _fa_model, _fa_tokenizer, _fa_aligner

    bundle = torchaudio.pipelines.MMS_FA
    _fa_model = bundle.get_model().to(device)
    _fa_tokenizer = bundle.get_tokenizer()
    _fa_aligner = bundle.get_aligner()
    _fa_device = device

    return _fa_model, _fa_tokenizer, _fa_aligner


_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "hundred",
}


def _number_to_words(text: str) -> str:
    """
    Convert numbers in text to words.
    Handles common patterns like "7B", "8x7B", "2.0", "10.7B".
    """
    # Handle decimal numbers like "2.0", "10.7"
    def replace_decimal(m):
        parts = m.group(0).split(".")
        left = _number_to_words(parts[0])
        right = " ".join(_NUMBER_WORDS.get(d, d) for d in parts[1])
        return f"{left} point {right}"

    text = re.sub(r"\d+\.\d+", replace_decimal, text)

    # Handle standalone numbers (longest match first)
    for num, word in sorted(_NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
        text = text.replace(num, f" {word} ")

    return text


def _normalize_word(word: str) -> List[str]:
    """
    Normalize a single word for MMS_FA alignment.
    Returns list of normalized parts (hyphens split into multiple words).
    """
    # Convert to lowercase
    word = word.lower()
    # Convert numbers to words BEFORE removing non-letters
    word = _number_to_words(word)
    # Replace hyphens with spaces (e.g., "fine-tune" -> "fine tune")
    word = word.replace("-", " ")
    # Keep only letters, apostrophes, and spaces
    word = re.sub(r"[^a-z' ]", "", word)
    # Collapse multiple spaces and split
    word = re.sub(r"\s+", " ", word).strip()
    # Split on spaces and filter empty
    parts = [p for p in word.split() if p]
    return parts


def normalize_with_mapping(text: str) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    Normalize text while tracking mapping from original to normalized words.

    Returns:
        original_words: List of original words (preserving case/punctuation)
        normalized_words: List of normalized words for alignment
        mapping: For each original word, list of indices into normalized_words

    Example:
        text = "Phi-2 is fine-tuning!"
        returns:
            original_words = ["Phi-2", "is", "fine-tuning!"]
            normalized_words = ["phi", "is", "fine", "tuning"]
            mapping = [[0], [1], [2, 3]]
    """
    original_words = text.split()
    normalized_words = []
    mapping = []

    for word in original_words:
        norm_parts = _normalize_word(word)

        if norm_parts:
            start_idx = len(normalized_words)
            normalized_words.extend(norm_parts)
            mapping.append(list(range(start_idx, len(normalized_words))))
        else:
            # Word normalized to nothing (e.g., just numbers/punctuation)
            # Map to empty list - will need special handling
            mapping.append([])

    return original_words, normalized_words, mapping


def reverse_timestamps(
    timestamps: List[Dict],
    original_words: List[str],
    mapping: List[List[int]]
) -> List[Dict]:
    """
    Map normalized word timestamps back to original words.

    Combines timestamps when one original word maps to multiple normalized words.
    """
    result = []

    for orig_idx, word in enumerate(original_words):
        norm_indices = mapping[orig_idx]

        if not norm_indices:
            # Original word had no normalized content (rare edge case)
            # Use previous word's end time or 0
            if result:
                t = result[-1]["end"]
            else:
                t = 0.0
            result.append({"word": word, "start": t, "end": t})
        elif norm_indices[0] >= len(timestamps):
            # Alignment didn't produce enough timestamps (shouldn't happen normally)
            break
        else:
            # Get start from first normalized word, end from last
            first_idx = norm_indices[0]
            last_idx = min(norm_indices[-1], len(timestamps) - 1)

            result.append({
                "word": word,
                "start": timestamps[first_idx]["start"],
                "end": timestamps[last_idx]["end"]
            })

    return result


def normalize_text_for_alignment(text: str) -> str:
    """
    Normalize text for MMS_FA alignment (simple version without mapping).
    MMS_FA expects lowercase text with only letters and apostrophes.
    """
    # Convert to lowercase
    text = text.lower()
    # Convert numbers to words
    text = _number_to_words(text)
    # Replace hyphens with spaces (e.g., "fine-tune" -> "fine tune")
    text = text.replace("-", " ")
    # Keep only letters, apostrophes, and spaces
    text = re.sub(r"[^a-z' ]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_word_timestamps(
    waveform: torch.Tensor,
    sample_rate: int,
    text: str,
    device: str = "cpu"
) -> List[Dict]:
    """
    Use TorchAudio MMS_FA to align text to audio and get word timestamps.

    Args:
        waveform: Audio tensor, shape [1, num_samples] or [num_samples]
        sample_rate: Sample rate of the audio
        text: Ground truth text to align
        device: Device to run model on

    Returns:
        List of dicts: [{"word": str, "start": float, "end": float}, ...]
    """
    model, tokenizer, aligner = _load_fa_model(device)
    bundle = torchaudio.pipelines.MMS_FA

    # Ensure waveform is 2D [1, num_samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to model's expected sample rate (16kHz)
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
        waveform = resampler(waveform)

    # Normalize text and split into words
    normalized_text = normalize_text_for_alignment(text)
    words = normalized_text.split()

    if not words:
        return []

    # Get frame-level emissions
    waveform = waveform.to(device)
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Tokenize words (character-level)
    tokens = tokenizer(words)

    # Align tokens to audio frames
    token_spans = aligner(emission[0], tokens)

    # MMS_FA uses 20ms frames = 320 samples at 16kHz
    hop_length = 320  # samples
    frames_per_second = bundle.sample_rate / hop_length

    # token_spans is grouped by word - each element is a list of character spans
    word_timestamps = []
    for i, word in enumerate(words):
        if i >= len(token_spans):
            break

        char_spans = token_spans[i]  # List of spans for this word's characters
        if not char_spans:
            continue

        # Get start of first char and end of last char
        start_frame = char_spans[0].start
        end_frame = char_spans[-1].end

        start_time = start_frame / frames_per_second
        end_time = end_frame / frames_per_second

        word_timestamps.append({
            "word": word,
            "start": start_time,
            "end": end_time
        })

    return word_timestamps


def get_word_timestamps_with_original(
    waveform: torch.Tensor,
    sample_rate: int,
    text: str,
    device: str = "cpu"
) -> List[Dict]:
    """
    Get word timestamps preserving original word forms (case, punctuation).

    Uses normalize_with_mapping to track how original words map to normalized
    words, then reverses the mapping after alignment.

    Example:
        text = "Phi-2 is fine-tuning!"
        Returns: [
            {"word": "Phi-2", "start": 0.5, "end": 0.9},
            {"word": "is", "start": 1.0, "end": 1.1},
            {"word": "fine-tuning!", "start": 1.2, "end": 1.6}
        ]
    """
    # Get mapping from original to normalized words
    original_words, normalized_words, mapping = normalize_with_mapping(text)

    if not normalized_words:
        return []

    # Get timestamps for normalized words
    model, tokenizer, aligner = _load_fa_model(device)
    bundle = torchaudio.pipelines.MMS_FA

    # Ensure waveform is 2D [1, num_samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to model's expected sample rate (16kHz)
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
        waveform = resampler(waveform)

    # Get frame-level emissions
    waveform = waveform.to(device)
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Tokenize and align
    tokens = tokenizer(normalized_words)
    token_spans = aligner(emission[0], tokens)

    # Convert to timestamps
    hop_length = 320
    frames_per_second = bundle.sample_rate / hop_length

    normalized_timestamps = []
    for i, word in enumerate(normalized_words):
        if i >= len(token_spans):
            break

        char_spans = token_spans[i]
        if not char_spans:
            # No alignment for this word - use interpolated time
            if normalized_timestamps:
                t = normalized_timestamps[-1]["end"]
            else:
                t = 0.0
            normalized_timestamps.append({"word": word, "start": t, "end": t})
            continue

        start_frame = char_spans[0].start
        end_frame = char_spans[-1].end

        normalized_timestamps.append({
            "word": word,
            "start": start_frame / frames_per_second,
            "end": end_frame / frames_per_second
        })

    # Reverse map to original words
    return reverse_timestamps(normalized_timestamps, original_words, mapping)
