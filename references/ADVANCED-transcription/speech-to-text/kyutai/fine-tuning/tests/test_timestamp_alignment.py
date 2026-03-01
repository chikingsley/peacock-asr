import pytest
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from timestamp_alignment import schedule_tokens_discrete, build_word_spans, assign_tokens_to_words


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        # Simple word-based tokenization for testing
        self.token_map = {
            "Hello": [100, 101],  # "Hel" + "lo" 
            "world": [200],       # "world"
            "this": [300],        # "this"
            "is": [400],          # "is"
            "a": [500],           # "a"
            "test": [600, 601],   # "te" + "st"
        }
        self.id_to_token = {
            0: "<unk>",
            3: "<pad>",
            100: "Hel", 101: "lo",
            200: "world",
            300: "this",
            400: "is", 
            500: "a",
            600: "te", 601: "st"
        }
    
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        """Mock tokenization."""
        words = text.split()
        input_ids = []
        offset_mapping = []
        
        char_pos = 0
        for word in words:
            # Skip leading spaces
            while char_pos < len(text) and text[char_pos] == ' ':
                char_pos += 1
                
            word_start = char_pos
            if word in self.token_map:
                token_ids = self.token_map[word]
                # Distribute character positions across tokens
                chars_per_token = len(word) // len(token_ids)
                for i, token_id in enumerate(token_ids):
                    start_char = word_start + i * chars_per_token
                    end_char = word_start + (i + 1) * chars_per_token
                    if i == len(token_ids) - 1:  # Last token gets remaining chars
                        end_char = word_start + len(word)
                    
                    input_ids.append(token_id)
                    offset_mapping.append((start_char, end_char))
            
            char_pos += len(word)
        
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offset_mapping
        return result
    
    def convert_ids_to_tokens(self, ids):
        """Convert token IDs to token strings."""
        return [self.id_to_token.get(id, f"<unk_{id}>") for id in ids]


def test_build_word_spans():
    """Test building word character spans."""
    text = "Hello world this is a test"
    words = ["Hello", "world", "this", "is", "a", "test"]
    
    spans = build_word_spans(text, words)
    
    expected = [(0, 5), (6, 11), (12, 16), (17, 19), (20, 21), (22, 26)]
    assert spans == expected


def test_assign_tokens_to_words():
    """Test assigning tokens to words."""
    tokenizer = MockTokenizer()
    text = "Hello world this"
    words = ["Hello", "world", "this"]
    spans = [(0, 5), (6, 11), (12, 16)]
    
    tokens_by_word = assign_tokens_to_words(text, tokenizer, spans)
    
    # Hello -> [100, 101], world -> [200], this -> [300]
    assert len(tokens_by_word) == 3
    assert tokens_by_word[0] == [(100, "Hel"), (101, "lo")]
    assert tokens_by_word[1] == [(200, "world")]
    assert tokens_by_word[2] == [(300, "this")]


def test_schedule_tokens_discrete_basic():
    """Test basic token scheduling."""
    tokenizer = MockTokenizer()
    
    # Sample with word timestamps
    sample = {
        "text": "Hello world",
        "start_time": "00:00:00",
        "end_time": "00:00:02", 
        "word_timestamps": [
            {"word": "Hello", "start": 0.1},
            {"word": "world", "start": 0.8}
        ]
    }
    
    result = schedule_tokens_discrete(
        sample, 
        tokenizer,
        stt_delay=0.5,
        frame_hop_s=0.02,
        pad_to_segment_end=True
    )
    
    # Check structure
    assert len(result) > 0
    assert all("token_id" in item for item in result)
    assert all("token" in item for item in result)
    assert all("start" in item for item in result)
    assert all("end" in item for item in result)
    
    # Check timing progression
    times = [item["start"] for item in result]
    assert all(times[i] <= times[i+1] for i in range(len(times)-1))
    
    # Check that we have UNK tokens at word boundaries
    unk_positions = [i for i, item in enumerate(result) if item["token_id"] == 0]
    assert len(unk_positions) == 2  # One UNK per word


def test_schedule_tokens_discrete_with_padding():
    """Test token scheduling with padding to segment end."""
    tokenizer = MockTokenizer()
    
    sample = {
        "text": "Hello world",
        "start_time": "00:00:00",
        "end_time": "00:00:03",
        "word_timestamps": [
            {"word": "Hello", "start": 0.5},
            {"word": "world", "start": 1.5}
        ]
    }
    
    result = schedule_tokens_discrete(
        sample,
        tokenizer,
        stt_delay=0.0,
        frame_hop_s=0.1,
        pad_to_segment_end=True
    )
    
    # Should have padding tokens before first word and at the end
    pad_tokens = [item for item in result if item["token_id"] == 3]
    assert len(pad_tokens) > 0
    
    # Last token should be close to segment end time
    last_time = result[-1]["end"]
    assert abs(last_time - 3.0) < 0.2  # Within frame tolerance


def test_spillover_truncate():
    """Test truncate spillover policy.""" 
    tokenizer = MockTokenizer()
    
    sample = {
        "text": "Hello world",
        "start_time": "00:00:00",
        "word_timestamps": [
            {"word": "Hello", "start": 0.0},
            {"word": "world", "start": 0.1}  # Very close timing
        ]
    }
    
    result = schedule_tokens_discrete(
        sample,
        tokenizer,
        stt_delay=0.0,
        frame_hop_s=0.1,
        spillover="truncate"
    )
    
    # Should handle the tight timing by truncating tokens
    assert len(result) > 0


def test_spillover_shift():
    """Test shift spillover policy."""
    tokenizer = MockTokenizer()
    
    sample = {
        "text": "Hello world",
        "start_time": "00:00:00", 
        "word_timestamps": [
            {"word": "Hello", "start": 0.0},
            {"word": "world", "start": 0.1}  # Very close timing
        ]
    }
    
    result = schedule_tokens_discrete(
        sample,
        tokenizer,
        stt_delay=0.0,
        frame_hop_s=0.1,
        spillover="shift"
    )
    
    # Should handle the tight timing by shifting subsequent timestamps
    assert len(result) > 0
    # Times should still be monotonically increasing
    times = [item["start"] for item in result]
    assert all(times[i] <= times[i+1] for i in range(len(times)-1))


if __name__ == "__main__":
    # Run tests
    test_build_word_spans()
    test_assign_tokens_to_words()
    test_schedule_tokens_discrete_basic()
    test_schedule_tokens_discrete_with_padding()
    test_spillover_truncate()
    test_spillover_shift()
    print("All tests passed!")