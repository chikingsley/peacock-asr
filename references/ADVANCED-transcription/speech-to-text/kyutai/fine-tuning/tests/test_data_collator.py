import pytest
import torch
import numpy as np
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_collator import TimeAlignedDataCollator


class MockProcessor:
    """Mock processor for testing."""
    
    def __init__(self):
        self.tokenizer = MockTokenizer()
        
    def __call__(self, audio_array, sampling_rate=None):
        """Mock audio processing."""
        # Simulate processing - return dummy input_values and padding_mask
        seq_len = len(audio_array) // 1000  # Simulate downsampling
        return {
            'input_values': torch.randn(1, len(audio_array)),
            'padding_mask': torch.ones(1, seq_len)
        }


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab = {
            "Hello": 100, "world": 101, "this": 102, "is": 103, 
            "a": 104, "test": 105, "<pad>": 3, "<unk>": 0
        }
        
    def __call__(self, text, **kwargs):
        """Mock tokenization."""
        words = text.split()
        input_ids = [self.vocab.get(word, 0) for word in words]
        
        result = {"input_ids": input_ids}
        
        if kwargs.get('return_offsets_mapping'):
            # Create mock character offsets
            offsets = []
            char_pos = 0
            for word in words:
                start = char_pos
                end = char_pos + len(word)
                offsets.append((start, end))
                char_pos = end + 1  # +1 for space
            result["offset_mapping"] = offsets
            
        return result
    
    def convert_ids_to_tokens(self, ids):
        """Convert token IDs to token strings."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        return [id_to_token.get(id, f"<unk_{id}>") for id in ids]


class MockCodecModel:
    """Mock codec model for testing."""
    
    def __init__(self):
        self.config = Mock()
        
    def encode(self, input_values, padding_mask=None):
        """Mock codec encoding."""
        batch_size, audio_len = input_values.shape
        # Simulate encoding to shorter sequence with 32 codebooks
        seq_len = min(100, audio_len // 1000)  # Simulate downsampling
        
        mock_output = Mock()
        mock_output.audio_codes = torch.randint(0, 1000, (batch_size, 32, seq_len))
        return mock_output


class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self.codec_model = MockCodecModel()
        self.config = Mock()
        self.config.bos_token_id = 1
        self.config.audio_bos_token_id = 2
        self.config.audio_pad_token_id = 4
        self.device = "cpu"


def create_mock_sample(duration_seconds=2.0, num_words=3, frame_hop_s=0.08):
    """Create a mock sample for testing."""
    # Create audio array (simulate 24kHz as in real preprocessing)
    audio_len = int(duration_seconds * 24000)
    audio_array = np.random.randn(audio_len).astype(np.float32)

    # Create word timestamps
    words = ["Hello", "world", "test"][:num_words]
    word_timestamps = []

    for i, word in enumerate(words):
        start_time = i * (duration_seconds / len(words))
        end_time = (i + 1) * (duration_seconds / len(words))
        word_timestamps.append({
            "word": word,
            "start": start_time,
            "end": end_time
        })

    # Create mock pre-computed audio codes
    # Frame hop is 0.08s, so seq_len = duration / 0.08
    seq_len = int(duration_seconds / 0.08)
    audio_codes = np.random.randint(0, 1000, (seq_len, 32)).tolist()  # [seq_len, 32]

    # Build a simple token schedule – one token per word
    token_schedule = []
    for idx, word_info in enumerate(word_timestamps):
        token_schedule.append({
            "token_id": 100 + idx,
            "token": word_info["word"],
            "start": word_info["start"],
            "end": min(word_info["end"], word_info["start"] + frame_hop_s)
        })

    return {
        "processed_audio": {
            "array": audio_array.tolist(),
            "sampling_rate": 24000
        },
        "word_timestamps": word_timestamps,
        "audio_codes": audio_codes,  # Pre-computed
        "token_schedule": token_schedule
    }


def test_data_collator_single_sample():
    """Test data collator with a single sample."""
    processor = MockProcessor()
    model = MockModel()
    
    collator = TimeAlignedDataCollator(
        processor=processor,
        model=model,
        frame_hop_s=0.08
    )
    
    sample = create_mock_sample(duration_seconds=1.0, num_words=2)
    batch = [sample]
    
    result = collator(batch)
    
    # Check output structure
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result
    
    # Check tensor shapes
    batch_size, seq_len, num_dims = result["input_ids"].shape
    assert batch_size == 1
    assert num_dims == 33  # 1 text + 32 audio
    assert result["labels"].shape == (batch_size, seq_len)
    assert result["attention_mask"].shape == (batch_size, seq_len)
    
    # Check that we have non-pad tokens
    assert (result["input_ids"][:, :, 0] != 3).sum() > 0  # Some non-pad text tokens
    
    print(f"✅ Single sample test passed - shape: {result['input_ids'].shape}")


def test_data_collator_multiple_samples():
    """Test data collator with multiple samples of different lengths."""
    processor = MockProcessor()
    model = MockModel()
    
    collator = TimeAlignedDataCollator(
        processor=processor,
        model=model,
        frame_hop_s=0.08
    )
    
    # Create samples with different durations
    samples = [
        create_mock_sample(duration_seconds=1.0, num_words=2),  # Short
        create_mock_sample(duration_seconds=3.0, num_words=3),  # Long
        create_mock_sample(duration_seconds=2.0, num_words=2),  # Medium
    ]
    
    result = collator(samples)
    
    # Check batch dimensions
    batch_size, seq_len, num_dims = result["input_ids"].shape
    assert batch_size == 3
    assert num_dims == 33
    
    # Check that all samples are padded to same length
    assert result["labels"].shape == (batch_size, seq_len)
    assert result["attention_mask"].shape == (batch_size, seq_len)
    
    # Check attention mask - shorter samples should have 0s at the end
    attention_sums = result["attention_mask"].sum(dim=1)
    assert not torch.all(attention_sums == attention_sums[0])  # Different lengths
    
    # Check that padding was applied correctly
    for i in range(batch_size):
        mask_len = int(attention_sums[i].item())
        # Padded positions should have ignore_index in labels
        if mask_len < seq_len:
            assert torch.all(result["labels"][i, mask_len:] == -100)
    
    print(f"✅ Multiple samples test passed - batch shape: {result['input_ids'].shape}")
    print(f"   Attention lengths: {attention_sums.tolist()}")


def test_data_collator_edge_cases():
    """Test data collator with edge cases."""
    processor = MockProcessor()
    model = MockModel()
    
    collator = TimeAlignedDataCollator(
        processor=processor,
        model=model,
        frame_hop_s=0.08
    )
    
    # Test with very short audio
    short_sample = create_mock_sample(duration_seconds=0.1, num_words=1)
    result = collator([short_sample])
    
    assert result["input_ids"].shape[0] == 1  # Batch size 1
    assert result["input_ids"].shape[2] == 33  # Correct dimensions
    
    # Test BOS token is set at position 0
    # Note: Scheduled tokens may overwrite BOS if they start at time 0
    # Just verify we have the audio BOS token
    assert result["input_ids"][0, 0, 1] == 2  # Audio BOS token
    
    print("✅ Edge cases test passed")


def test_data_collator_empty_batch():
    """Test data collator error handling."""
    processor = MockProcessor()
    model = MockModel()

    collator = TimeAlignedDataCollator(
        processor=processor,
        model=model
    )
    
    # Test empty batch
    with pytest.raises(ValueError, match="No valid samples in batch"):
        collator([])
    
    print("✅ Empty batch error handling test passed")


if __name__ == "__main__":
    # Run tests
    test_data_collator_single_sample()
    test_data_collator_multiple_samples()
    test_data_collator_edge_cases()
    test_data_collator_empty_batch()
    print("🎉 All data collator tests passed!")
