#!/usr/bin/env python3

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Model
import torch
import os
import json

def save_model_info(model, name, output_dir="pyannote_model_info"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"{name}.txt"), "w") as f:
        f.write(f"=== {name} Model Information ===\n\n")
        
        # Basic model string representation
        f.write("Model Structure:\n")
        if hasattr(model, "model_"):
            f.write(str(model.model_) + "\n\n")
        else:
            f.write(str(model) + "\n\n")
        
        # Try to get model parameters if it's a PyTorch model
        if isinstance(model, torch.nn.Module):
            f.write("Parameters:\n")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n\n")
            
            # Detailed layer information
            f.write("Layer Details:\n")
            for name, module in model.named_modules():
                if len(name) > 0:  # Skip the root module
                    f.write(f"{name}: {module.__class__.__name__}\n")
                    if hasattr(module, "in_features") and hasattr(module, "out_features"):
                        f.write(f"    Input features: {module.in_features}\n")
                        f.write(f"    Output features: {module.out_features}\n")
            f.write("\n")
        
        # Try to get specifications
        if hasattr(model, "specifications"):
            f.write("Model Specifications:\n")
            f.write(str(model.specifications) + "\n\n")
        
        # For SpeechBrain models
        if hasattr(model, "embedding_model"):
            f.write("SpeechBrain Embedding Model Details:\n")
            f.write(str(model.embedding_model) + "\n\n")
            
            if hasattr(model.embedding_model, "encoder"):
                f.write("Encoder Architecture:\n")
                f.write(str(model.embedding_model.encoder) + "\n\n")
            
            if hasattr(model.embedding_model, "classifier"):
                f.write("Classifier Architecture:\n")
                f.write(str(model.embedding_model.classifier) + "\n\n")
        
        # For Pipeline
        if isinstance(model, Pipeline):
            f.write("Pipeline Components:\n")
            # Try to access pipeline attributes safely
            for attr in ["segmentation", "embedding", "clustering"]:
                if hasattr(model, attr):
                    f.write(f"{attr}: {getattr(model, attr)}\n")
            
            if hasattr(model, "_segmentation"):
                f.write("\nSegmentation Model Details:\n")
                f.write(str(model._segmentation.model_) + "\n")
            
            if hasattr(model, "_embedding"):
                f.write("\nEmbedding Model Details:\n")
                f.write(str(model._embedding.embedding_model) + "\n")

def main():
    output_dir = "pyannote_model_info"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models (this may take a few minutes)...")
    
    try:
        # 1. Load and inspect Segmentation Model
        print("\nLoading segmentation model...")
        segmentation = Model.from_pretrained("pyannote/segmentation-3.0")
        save_model_info(segmentation, "segmentation", output_dir)
        print("✓ Saved segmentation model info")
    except Exception as e:
        print(f"Error loading segmentation model: {str(e)}")
    
    try:
        # 2. Load and inspect Embedding Model
        print("\nLoading embedding model...")
        embedding = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
        save_model_info(embedding, "embedding", output_dir)
        print("✓ Saved embedding model info")
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
    
    try:
        # 3. Load and inspect Full Pipeline
        print("\nLoading pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        save_model_info(pipeline, "pipeline", output_dir)
        print("✓ Saved pipeline info")
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
    
    print(f"\nModel information has been saved to the '{output_dir}' directory")

if __name__ == "__main__":
    main() 