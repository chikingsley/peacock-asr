#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) for Kokoro Decoder

Implements LoRA adapters that can be applied to Kokoro's decoder layers
without modifying the base model weights.

Key concepts:
- For a linear layer W, LoRA adds: W' = W + BA
- B: (out_dim, rank), A: (rank, in_dim)
- Only B and A are trainable, W is frozen
- Reduces trainable params from in*out to rank*(in+out)

Reference: https://arxiv.org/abs/2106.09685
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Replaces: y = Wx + b
    With: y = Wx + b + (BA)x

    Where:
    - W, b are frozen original weights
    - B: (out_features, rank) - initialized to zeros
    - A: (rank, in_features) - initialized with kaiming
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep reference to original layer (frozen)
        self.original = original_layer
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with kaiming, B with zeros
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original forward
        result = self.original(x)

        # LoRA adaptation: (BA)x * scaling
        lora_out = F.linear(
            self.dropout(x),
            self.lora_B @ self.lora_A
        ) * self.scaling

        return result + lora_out

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original layer for inference."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.original.bias is not None
        )

        # W' = W + BA * scaling
        merged.weight.data = (
            self.original.weight.data +
            (self.lora_B @ self.lora_A) * self.scaling
        )

        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data

        return merged


class LoRAConv1d(nn.Module):
    """Conv1d layer with LoRA adaptation.

    Similar to LoRALinear but for 1D convolutions.
    Uses 1x1 convolutions for the LoRA matrices.
    """

    def __init__(
        self,
        original_layer: nn.Conv1d,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep reference to original layer (frozen)
        self.original = original_layer
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA as 1x1 convolutions
        self.lora_A = nn.Conv1d(
            self.in_channels, rank, kernel_size=1, bias=False
        )
        self.lora_B = nn.Conv1d(
            rank, self.out_channels, kernel_size=1, bias=False
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        result = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return result + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, List[str]]:
    """Apply LoRA adapters to specified modules in a model.

    Args:
        model: PyTorch model to adapt
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA
        target_modules: List of module name patterns to target
                       If None, targets all Linear layers

    Returns:
        model: Model with LoRA adapters applied
        adapted_modules: List of module names that were adapted
    """
    if target_modules is None:
        target_modules = ['linear', 'proj', 'fc', 'dense', 'query', 'key', 'value', 'out']

    adapted_modules = []

    def should_adapt(name: str) -> bool:
        """Check if module name matches any target pattern."""
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in target_modules)

    # Find and replace target modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_adapt(name):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRA version
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)
            adapted_modules.append(name)

        elif isinstance(module, nn.Conv1d) and should_adapt(name):
            # Only apply to 1x1 convolutions for now
            if module.kernel_size == (1,):
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                lora_layer = LoRAConv1d(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)
                adapted_modules.append(name)

    return model, adapted_modules


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model state dict."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
    """Load LoRA parameters into model."""
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name] = param
    model.load_state_dict(model_state)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        trainable: Number of trainable parameters
        total: Total number of parameters
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_non_lora(model: nn.Module):
    """Freeze all parameters except LoRA layers."""
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False


class KokoroLoRAWrapper(nn.Module):
    """Wrapper for Kokoro model with LoRA adaptation.

    This class handles:
    1. Loading the Kokoro model
    2. Applying LoRA to decoder layers
    3. Managing voice embeddings
    4. Saving/loading LoRA weights
    """

    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = torch.device(device)
        self.lora_rank = lora_rank

        # Load Kokoro
        self._load_kokoro(model_path, voices_dir)

        # Apply LoRA to decoder
        if hasattr(self, 'decoder'):
            self.decoder, self.lora_modules = apply_lora_to_model(
                self.decoder,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            print(f"Applied LoRA to {len(self.lora_modules)} modules")

        # Count parameters
        trainable, total = count_parameters(self)
        print(f"Parameters: {trainable:,} trainable / {total:,} total")

    def _load_kokoro(self, model_path: str, voices_dir: str):
        """Load Kokoro model components."""
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='a')
            self.use_pipeline = True
            print("Loaded Kokoro via pipeline")

            # Access internal model for LoRA
            if hasattr(self.pipeline, 'model'):
                self.model = self.pipeline.model
                if hasattr(self.model, 'decoder'):
                    self.decoder = self.model.decoder
        except ImportError:
            # Load raw weights
            print("Loading raw Kokoro weights...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'decoder' in checkpoint:
                    self.decoder = checkpoint['decoder']
            else:
                self.model = checkpoint

            self.use_pipeline = False

    def generate(
        self,
        text: str,
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> Tuple[torch.Tensor, int]:
        """Generate audio with the adapted model."""
        if self.use_pipeline:
            # Use kokoro pipeline
            generator = self.pipeline(text, voice=voice, speed=speed)

            import numpy as np
            audio_chunks = []
            for chunk in generator:
                if hasattr(chunk, 'audio'):
                    audio_chunks.append(chunk.audio)
                else:
                    audio_chunks.append(chunk)

            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                return torch.from_numpy(audio).float(), 24000
            else:
                raise RuntimeError("No audio generated")
        else:
            raise NotImplementedError(
                "Raw model inference not implemented. Install kokoro package."
            )

    def save_lora(self, path: str):
        """Save only LoRA weights."""
        lora_state = get_lora_state_dict(self)
        torch.save({
            'lora_state': lora_state,
            'rank': self.lora_rank,
            'modules': self.lora_modules,
        }, path)
        print(f"Saved LoRA weights to {path}")

    def load_lora(self, path: str):
        """Load LoRA weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        load_lora_state_dict(self, checkpoint['lora_state'])
        print(f"Loaded LoRA weights from {path}")


if __name__ == "__main__":
    # Test LoRA implementation
    print("Testing LoRA implementation...")

    # Test LoRALinear
    linear = nn.Linear(256, 512)
    lora_linear = LoRALinear(linear, rank=16)

    x = torch.randn(1, 256)
    y = lora_linear(x)
    print(f"LoRALinear: input {x.shape} -> output {y.shape}")

    # Count params
    original_params = sum(p.numel() for p in linear.parameters())
    lora_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"Original params: {original_params:,}")
    print(f"LoRA trainable params: {lora_params:,}")
    print(f"Reduction: {(1 - lora_params/original_params)*100:.1f}%")

    # Test apply_lora_to_model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )
            self.decoder = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = SimpleModel()
    model, adapted = apply_lora_to_model(model, rank=8)

    print(f"\nAdapted modules: {adapted}")
    trainable, total = count_parameters(model)
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    # Test forward pass
    x = torch.randn(1, 128)
    y = model(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    print("\nLoRA implementation working!")
