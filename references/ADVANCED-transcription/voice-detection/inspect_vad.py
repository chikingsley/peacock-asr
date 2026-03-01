#!/usr/bin/env python3

import torch
import os
import json
from typing import Dict, Any

def save_model_info(model: torch.nn.Module, utils: Dict[str, Any], output_dir: str = "vad_model_info"):
    """Save detailed information about the Silero VAD model."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
        f.write("=== Silero VAD Model Information ===\n\n")
        
        # Basic model information
        f.write("Model Structure:\n")
        f.write(str(model) + "\n\n")
        
        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write("Parameters:\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")
        
        # Detailed layer information
        f.write("Layer Details:\n")
        for name, module in model.named_modules():
            if len(name) > 0:  # Skip the root module
                f.write(f"{name}: {module.__class__.__name__}\n")
                # Get input/output features for linear layers
                if isinstance(module, torch.nn.Linear):
                    f.write(f"    Input features: {module.in_features}\n")
                    f.write(f"    Output features: {module.out_features}\n")
                # Get conv layer details
                elif isinstance(module, torch.nn.Conv1d):
                    f.write(f"    In channels: {module.in_channels}\n")
                    f.write(f"    Out channels: {module.out_channels}\n")
                    f.write(f"    Kernel size: {module.kernel_size}\n")
                    f.write(f"    Stride: {module.stride}\n")
                    f.write(f"    Padding: {module.padding}\n")
        f.write("\n")
        
        # Available utility functions
        f.write("Utility Functions:\n")
        for i, util in enumerate(utils):
            if callable(util):
                f.write(f"{i}: {util.__name__}\n")
            else:
                f.write(f"{i}: {type(util).__name__}\n")
        f.write("\n")
        
        # Default configurations
        f.write("Default Configurations:\n")
        f.write("- window_size_samples: 512 (32ms at 16kHz)\n")
        f.write("- threshold: 0.5\n")
        f.write("- min_speech_duration_ms: 250\n")
        f.write("- min_silence_duration_ms: 100\n")
        f.write("- sampling_rate: 16000\n")

def main():
    print("Loading Silero VAD model (this may take a moment)...")
    
    try:
        # Load the model
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True,
                                    trust_repo=True)
        
        # Save model information
        save_model_info(model, utils)
        print("✓ Model information has been saved to the 'vad_model_info' directory")
        
        # Print available utility functions
        print("\nAvailable utility functions:")
        for i, util in enumerate(utils):
            if callable(util):
                print(f"{i}: {util.__name__}")
            else:
                print(f"{i}: {type(util).__name__}")
        
    except Exception as e:
        print(f"Error loading or inspecting model: {str(e)}")

if __name__ == "__main__":
    main() 