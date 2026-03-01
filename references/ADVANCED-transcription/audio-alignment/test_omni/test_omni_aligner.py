"""
Test script for OmniASR CTC forced alignment.

This script demonstrates how to extract CTC emissions from the OmniASR model
and use Viterbi alignment to get word-level timestamps.

Run with: uv run python test_omni_aligner.py <audio_file> "<transcript>"

Example:
  uv run python test_omni_aligner.py test.wav "Hello world how are you"
"""

import sys
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


@dataclass
class AlignmentResult:
    """Result of forced alignment."""
    word: str
    start: float
    end: float


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Load and preprocess audio file to mono 16kHz."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


class OmniASRAligner:
    """
    Forced aligner using OmniASR CTC model.

    Uses the CTC emissions from the model and Viterbi algorithm to align
    known text to audio.
    """

    def __init__(self, model_card: str = "omniASR_CTC_300M", device: str = DEVICE):
        self.device = device
        self.model_card = model_card
        self._pipeline = None
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the model lazily."""
        if self._pipeline is not None:
            return

        print(f"Loading {self.model_card} model...")
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self._pipeline = ASRInferencePipeline(model_card=self.model_card)
        self._model = self._pipeline.model
        self._tokenizer = self._pipeline.tokenizer
        self._encoder = self._tokenizer.create_encoder()

        # Get vocab info
        self._vocab_info = self._tokenizer.vocab_info
        # CTC blank is typically the pad token in fairseq2
        self._blank_idx = self._vocab_info.pad_idx  # Usually 1

        # Move model to device if needed
        if self.device != "cpu":
            self._model = self._model.to(self.device)

        print(f"Model loaded on {self.device}")
        print(f"Vocab size: {self._vocab_info.size}, blank_idx: {self._blank_idx}")

    def _get_emissions(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Extract CTC emissions (log probabilities) from audio.

        Returns:
            emissions: Tensor of shape [T, V] where T is time frames, V is vocab size
            stride: Number of audio samples per emission frame
        """
        self._load_model()

        # Ensure waveform is on correct device and has batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            # Multiple channels, take first
            waveform = waveform[:1]

        # Match model dtype and device
        model_dtype = next(self._model.parameters()).dtype
        waveform = waveform.to(device=self.device, dtype=model_dtype)

        # Create batch layout for fairseq2
        from fairseq2.nn import BatchLayout
        seq_len = waveform.shape[1]
        layout = BatchLayout.of(waveform, seq_lens=[seq_len])

        # Run forward pass to get logits
        with torch.inference_mode():
            # The model forward returns logits when return_logits=True
            output = self._model(waveform, layout, return_logits=True)

            if isinstance(output, tuple):
                logits = output[0]  # [B, T, V]
            else:
                logits = output

            # Apply log_softmax to get log probabilities
            emissions = torch.log_softmax(logits, dim=-1)

        # Remove batch dimension
        emissions = emissions.squeeze(0)  # [T, V]

        # Calculate stride (samples per frame)
        # OmniASR uses 20ms frames at 16kHz = 320 samples per frame
        stride = seq_len // emissions.shape[0]

        return emissions, stride

    def _tokenize_text(self, text: str, lang: str = "eng_Latn") -> list[int]:
        """Tokenize text using the model's tokenizer."""
        self._load_model()

        # Use the encoder from the tokenizer
        tokens = self._encoder(text)
        return tokens.tolist()

    def _get_blank_token_id(self) -> int:
        """Get the blank token ID used by CTC."""
        self._load_model()
        return self._blank_idx

    def _viterbi_align(
        self,
        emissions: torch.Tensor,
        tokens: list[int],
        blank_id: int = 0
    ) -> list[tuple[int, int]]:
        """
        Viterbi alignment to find optimal token-to-frame mapping.

        This is a simplified CTC alignment that finds the best path through
        the emission matrix for the given token sequence.

        Args:
            emissions: [T, V] log probabilities
            tokens: List of token IDs to align
            blank_id: ID of the blank token

        Returns:
            List of (start_frame, end_frame) for each token
        """
        emissions = emissions.cpu().float().numpy()
        T, V = emissions.shape
        n_tokens = len(tokens)

        if n_tokens == 0:
            return []

        # Create extended token sequence with blanks: blank, t1, blank, t2, blank, ...
        # This allows the model to output blanks between tokens
        extended_tokens = [blank_id]
        for t in tokens:
            extended_tokens.append(t)
            extended_tokens.append(blank_id)
        n_extended = len(extended_tokens)

        # Viterbi DP
        # dp[t][s] = best log probability to be at state s at time t
        NEG_INF = float('-inf')
        dp = np.full((T, n_extended), NEG_INF)
        backptr = np.zeros((T, n_extended), dtype=np.int32)

        # Initialize first frame
        dp[0, 0] = emissions[0, extended_tokens[0]]
        if n_extended > 1:
            dp[0, 1] = emissions[0, extended_tokens[1]]

        # Fill DP table
        for t in range(1, T):
            for s in range(n_extended):
                token_id = extended_tokens[s]
                emit_prob = emissions[t, token_id]

                # Can stay in same state
                candidates = [(dp[t-1, s], s)]

                # Can come from previous state
                if s > 0:
                    candidates.append((dp[t-1, s-1], s-1))

                # Can skip a blank (non-blank -> non-blank transition through blank)
                if s > 1 and extended_tokens[s-2] != extended_tokens[s]:
                    candidates.append((dp[t-1, s-2], s-2))

                best_prob, best_state = max(candidates, key=lambda x: x[0])
                dp[t, s] = best_prob + emit_prob
                backptr[t, s] = best_state

        # Backtrack to find alignment
        # Find best final state (must end at last token or last blank)
        final_states = [n_extended - 1, n_extended - 2] if n_extended > 1 else [0]
        best_final = max(final_states, key=lambda s: dp[T-1, s])

        # Backtrack
        path = []
        s = best_final
        for t in range(T-1, -1, -1):
            path.append((t, s))
            s = backptr[t, s]
        path.reverse()

        # Convert path to token alignments (skip blanks)
        alignments = []
        current_token_idx = -1
        start_frame = 0

        for t, s in path:
            token_id = extended_tokens[s]
            # Map extended index back to original token index
            orig_idx = (s - 1) // 2 if s > 0 and s % 2 == 1 else -1

            if orig_idx != current_token_idx:
                if current_token_idx >= 0:
                    # End previous token
                    alignments.append((start_frame, t))
                if orig_idx >= 0:
                    # Start new token
                    start_frame = t
                    current_token_idx = orig_idx

        # Close final token
        if current_token_idx >= 0:
            alignments.append((start_frame, T))

        return alignments

    def align(
        self,
        audio_path: str,
        text: str,
        lang: str = "eng_Latn",
        sample_rate: int = 16000
    ) -> tuple[list[AlignmentResult], float]:
        """
        Align text to audio and return word-level timestamps.

        Args:
            audio_path: Path to audio file
            text: Text to align
            lang: Language code (e.g., "eng_Latn")
            sample_rate: Expected sample rate (will resample if different)

        Returns:
            List of AlignmentResult with word timestamps
            Alignment time in milliseconds
        """
        start_time = time.perf_counter()

        # Load audio
        waveform, sr = load_audio(audio_path, sample_rate)
        duration = waveform.shape[1] / sr

        print(f"Audio duration: {duration:.2f}s")

        # Get emissions
        print("Extracting emissions...")
        emissions, stride = self._get_emissions(waveform)
        print(f"Emissions shape: {emissions.shape}, stride: {stride}")

        # Tokenize text
        words = text.split()
        print(f"Words to align: {words}")

        # For word-level alignment, we tokenize each word separately
        # and then align them sequentially
        results = []

        # Simple approach: tokenize full text and align
        # Then split by word boundaries
        full_tokens = self._tokenize_text(text, lang)
        print(f"Tokens: {full_tokens[:20]}..." if len(full_tokens) > 20 else f"Tokens: {full_tokens}")

        # Get blank token
        blank_id = self._get_blank_token_id()

        # Optimize: only keep needed token columns
        needed_tokens = set([blank_id] + full_tokens)
        token_map = {t: i for i, t in enumerate(sorted(needed_tokens))}

        # Create sparse emissions matrix
        sparse_emissions = emissions[:, sorted(needed_tokens)]
        mapped_tokens = [token_map[t] for t in full_tokens]
        mapped_blank = token_map[blank_id]

        print(f"Sparse emissions shape: {sparse_emissions.shape} (reduced from {emissions.shape[1]} to {len(needed_tokens)} tokens)")

        # Run Viterbi alignment
        print("Running Viterbi alignment...")
        alignments = self._viterbi_align(sparse_emissions, mapped_tokens, mapped_blank)

        # Convert frame indices to timestamps
        frames_per_second = sr / stride

        # Map token alignments back to words
        # This is simplified - ideally we'd track word boundaries through tokenization
        word_tokens = []
        for word in words:
            toks = self._tokenize_text(word, lang)
            word_tokens.append(len(toks))

        # Distribute alignments to words
        token_idx = 0
        for i, word in enumerate(words):
            n_toks = word_tokens[i]
            if token_idx + n_toks <= len(alignments):
                word_alignments = alignments[token_idx:token_idx + n_toks]
                if word_alignments:
                    start_frame = word_alignments[0][0]
                    end_frame = word_alignments[-1][1]
                    results.append(AlignmentResult(
                        word=word,
                        start=round(start_frame / frames_per_second, 3),
                        end=round(end_frame / frames_per_second, 3)
                    ))
            token_idx += n_toks

        align_time = (time.perf_counter() - start_time) * 1000

        return results, align_time

    def align_from_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        text: str,
        lang: str = "eng_Latn"
    ) -> tuple[list[dict], float, float]:
        """
        Align text to waveform tensor (for integration with server).

        Returns:
            timestamps: List of {"word": str, "start": float, "end": float}
            duration: Audio duration in seconds
            align_time_ms: Alignment time in milliseconds
        """
        start_time = time.perf_counter()

        # Ensure 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        duration = waveform.shape[-1] / sample_rate

        # Ensure mono
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Get emissions
        emissions, stride = self._get_emissions(waveform)

        # Tokenize text
        words = text.split()
        full_tokens = self._tokenize_text(text, lang)

        # Get blank token
        blank_id = self._get_blank_token_id()

        # Optimize: only keep needed token columns
        needed_tokens = set([blank_id] + full_tokens)
        token_map = {t: i for i, t in enumerate(sorted(needed_tokens))}
        sparse_emissions = emissions[:, sorted(needed_tokens)]
        mapped_tokens = [token_map[t] for t in full_tokens]
        mapped_blank = token_map[blank_id]

        # Run Viterbi alignment
        alignments = self._viterbi_align(sparse_emissions, mapped_tokens, mapped_blank)

        # Convert to timestamps
        frames_per_second = sample_rate / stride

        # Map to words
        word_tokens = [len(self._tokenize_text(w, lang)) for w in words]

        timestamps = []
        token_idx = 0
        for i, word in enumerate(words):
            n_toks = word_tokens[i]
            if token_idx + n_toks <= len(alignments):
                word_alignments = alignments[token_idx:token_idx + n_toks]
                if word_alignments:
                    start_frame = word_alignments[0][0]
                    end_frame = word_alignments[-1][1]
                    timestamps.append({
                        "word": word,
                        "start": round(start_frame / frames_per_second, 3),
                        "end": round(end_frame / frames_per_second, 3)
                    })
            token_idx += n_toks

        align_time_ms = (time.perf_counter() - start_time) * 1000

        return timestamps, duration, align_time_ms


def main():
    """Main entry point for testing."""
    if len(sys.argv) < 3:
        print("Usage: uv run python test_omni_aligner.py <audio_file> \"<transcript>\"")
        print("\nExample:")
        print("  uv run python test_omni_aligner.py test.wav \"Hello world how are you\"")

        # Run basic exploration instead
        print("\n--- Running basic model exploration ---\n")
        aligner = OmniASRAligner()
        aligner._load_model()

        # Show tokenizer info
        print("\n--- Tokenizer test ---")
        test_texts = ["Hello", "world", "Hello world"]
        for text in test_texts:
            tokens = aligner._tokenize_text(text)
            print(f"'{text}' -> {len(tokens)} tokens: {tokens}")

        blank_id = aligner._get_blank_token_id()
        print(f"\nBlank token ID: {blank_id}")

        return

    audio_path = sys.argv[1]
    text = sys.argv[2]

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    print(f"\nAligning: {audio_path}")
    print(f"Text: {text}\n")

    aligner = OmniASRAligner()
    results, align_time = aligner.align(audio_path, text)

    print(f"\n=== Alignment Results ===")
    print(f"Alignment time: {align_time:.1f}ms\n")

    for r in results:
        print(f"  {r.start:6.3f} - {r.end:6.3f}  {r.word}")

    print("\nDone!")


if __name__ == "__main__":
    main()
