import torch
from typing import Dict, List, Any


class TimeAlignedDataCollator:
    """
    Data collator for time-aligned speech-to-text training with Kyutai models.
    
    Handles:
    - Consuming pre-computed codec codes and token schedules
    - Batch padding for variable sequence lengths
    - Proper masking for training
    """
    
    def __init__(
        self,
        processor,
        model,
        stt_delay: float = 0.5,
        frame_hop_s: float = 0.08,
        pad_token_id: int = 3,
        ignore_index: int = -100,
        pad_weight: float = 0.25,  # Weight for PAD tokens (0.25 works well empirically)
    ):
        self.processor = processor
        self.model = model
        self.device = model.device  # Use model's device
        self.stt_delay = stt_delay
        self.frame_hop_s = frame_hop_s
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.pad_weight = pad_weight
        
    def _process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample: audio encoding + token scheduling."""

        # 1. Pre-computed audio codes and token schedule should already exist
        audio_codes_list = sample['audio_codes']  # [seq_len, 32]
        audio_codes = torch.tensor(audio_codes_list, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len, 32]

        audio_seq_len = audio_codes.shape[1]

        token_schedule = sample.get('token_schedule')
        if not token_schedule:
            raise ValueError("Sample is missing precomputed token_schedule")

        # 2. Create training tensors
        bos_token_id = self.model.config.bos_token_id

        # Initialize text tokens with pad (on self.device)
        text_input_tokens = torch.full((audio_seq_len,), self.pad_token_id, dtype=torch.long, device=self.device)

        # Labels for next-token prediction (on self.device)
        labels = torch.full((audio_seq_len,), self.ignore_index, dtype=torch.long, device=self.device)

        # Map scheduled tokens to frames
        for token_info in token_schedule:
            # Clamp frame position to avoid OOB access
            frame_pos = int(token_info['start'] / self.frame_hop_s)
            frame_pos = max(0, min(audio_seq_len - 1, frame_pos))

            token_id = token_info['token_id']

            # Place token in input sequence
            if token_id == 0:  # UNK token - mark word boundaries
                text_input_tokens[frame_pos] = token_id
            elif token_id != self.pad_token_id:  # Real text tokens
                text_input_tokens[frame_pos] = token_id

        # Overwrite position 0 with BOS (takes precedence over any scheduled token)
        text_input_tokens[0] = bos_token_id

        # The forward pass and generator handle shifting of labels
        labels = text_input_tokens.clone()
        labels[0] = self.ignore_index   # Never predict the initial token
        labels[-1] = self.ignore_index  # No target for last position

        # Create loss weights: PAD tokens get reduced weight, others get full weight
        # This replaces the previous approach of masking consecutive PADs
        loss_weights = torch.ones(audio_seq_len, dtype=torch.float32, device=self.device)
        is_pad = (labels == self.pad_token_id)
        loss_weights[is_pad] = self.pad_weight
        loss_weights[0] = 0.0   # BOS position - no loss
        loss_weights[-1] = 0.0  # Last position - no loss

        # 6. Combine text and audio tokens
        text_tokens = text_input_tokens.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
        input_ids = torch.cat([text_tokens, audio_codes], dim=2)    # [1, seq_len, 33]
        
        # Fix audio BOS at position 0
        if hasattr(self.model.config, 'audio_bos_token_id'):
            audio_bos_id = int(self.model.config.audio_bos_token_id)  # Ensure it's a Python int, not a tensor
            input_ids[0, 0, 1:] = audio_bos_id
        
        return {
            'input_ids': input_ids.squeeze(0),  # [seq_len, 33]
            'labels': labels,                    # [seq_len]
            'attention_mask': torch.ones(audio_seq_len, dtype=torch.long, device=self.device),  # [seq_len]
            'loss_weights': loss_weights,        # [seq_len]
        }
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples and pad to uniform length.
        
        Returns:
            Dict with 'input_ids', 'labels', 'attention_mask' tensors
        """
        
        # Process each sample
        processed_samples = []
        for sample in batch:
            try:
                processed = self._process_single_sample(sample)
                processed_samples.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                # Skip problematic samples or handle gracefully
                continue
        
        if not processed_samples:
            raise ValueError("No valid samples in batch")
        
        # Find max sequence length in batch
        max_seq_len = max(sample['input_ids'].shape[0] for sample in processed_samples)
        batch_size = len(processed_samples)

        # Initialize padded tensors (on self.device)
        padded_input_ids = torch.full(
            (batch_size, max_seq_len, 33),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        padded_labels = torch.full(
            (batch_size, max_seq_len),
            self.ignore_index,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.long,
            device=self.device
        )
        padded_loss_weights = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float32,
            device=self.device
        )
        
        # Fill padded tensors
        for i, sample in enumerate(processed_samples):
            seq_len = sample['input_ids'].shape[0]

            # Copy actual data
            padded_input_ids[i, :seq_len] = sample['input_ids']
            padded_labels[i, :seq_len] = sample['labels']
            attention_mask[i, :seq_len] = sample['attention_mask']

            # Copy loss weights
            padded_loss_weights[i, :seq_len] = sample['loss_weights']

            # Audio padding for the extra dimensions
            if seq_len < max_seq_len:
                # Explicitly: all 33 channels already filled with pad_token_id above
                # Now overwrite the 32 audio channels (1:33) with audio_pad_token_id
                audio_pad_id = getattr(self.model.config, 'audio_pad_token_id', self.pad_token_id)
                padded_input_ids[i, seq_len:, 1:] = audio_pad_id

        # Already on self.device, no need to move
        return {
            'input_ids': padded_input_ids,
            'labels': padded_labels,
            'attention_mask': attention_mask,
            'loss_weights': padded_loss_weights,
        }
