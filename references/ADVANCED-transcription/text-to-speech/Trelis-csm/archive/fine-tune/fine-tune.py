import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import torchaudio
import sys

# Add parent directory to Python path to allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Model, ModelArgs
from generator_train import Generator, Segment, load_csm_1b

def collate_fn(batch):
    # Find max sequence length in this batch
    max_len = max(item["tokens"].size(0) for item in batch)
    
    # Pad all sequences to max_len
    padded_batch = {
        "tokens": [],
        "tokens_mask": [],
        "input_pos": [],
        "target_tokens": []
    }
    
    for item in batch:
        # Pad tokens and mask
        tokens = item["tokens"]
        tokens_mask = item["tokens_mask"]
        input_pos = item["input_pos"]
        target_tokens = item["target_tokens"]
        
        pad_len = max_len - tokens.size(0)
        if pad_len > 0:
            # Pad tokens with zeros
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad_len))
            # Pad mask with False
            tokens_mask = torch.nn.functional.pad(tokens_mask, (0, 0, 0, pad_len))
            # Pad input_pos with sequential numbers
            input_pos = torch.nn.functional.pad(input_pos, (0, pad_len))
            # Pad target_tokens with zeros
            target_tokens = torch.nn.functional.pad(target_tokens, (0, 0, 0, pad_len))
        
        padded_batch["tokens"].append(tokens)
        padded_batch["tokens_mask"].append(tokens_mask)
        padded_batch["input_pos"].append(input_pos)
        padded_batch["target_tokens"].append(target_tokens)
    
    # Stack all tensors
    return {
        "tokens": torch.stack(padded_batch["tokens"]),
        "tokens_mask": torch.stack(padded_batch["tokens_mask"]),
        "input_pos": torch.stack(padded_batch["input_pos"]),
        "target_tokens": torch.stack(padded_batch["target_tokens"])
    }

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, generator: Generator):
        self.generator = generator
        self.data_dir = Path(data_dir)
        
        # Load metadata.csv
        self.metadata = []
        metadata_path = self.data_dir / "metadata.csv"
        print(f"Loading metadata from: {metadata_path}")
        
        try:
            with open(metadata_path, "r") as f:
                # Skip header line
                next(f)
                
                for i, line in enumerate(f, 1):
                    try:
                        filename, text, speaker_id = line.strip().split("|")
                        self.metadata.append({
                            "filename": filename,
                            "text": text,
                            "speaker_id": int(speaker_id)
                        })
                    except ValueError as e:
                        print(f"Error parsing line {i}: {line.strip()}")
                        print(f"Error details: {e}")
                        raise
        except Exception as e:
            print(f"Error loading metadata file: {e}")
            raise
        
        print(f"Loaded {len(self.metadata)} items from metadata")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_dir / "wavs" / item["filename"]
        print(f"Loading audio from: {audio_path}")
        
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.generator.sample_rate:
                audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=self.generator.sample_rate)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            raise
        
        # Create segment
        segment = Segment(
            speaker=item["speaker_id"],
            text=item["text"],
            audio=audio.squeeze(0)
        )
        
        # Tokenize text and audio
        text_tokens, text_masks = self.generator._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self.generator._tokenize_audio(segment.audio)
        
        # Combine tokens and masks
        tokens = torch.cat([text_tokens, audio_tokens], dim=0)
        tokens_mask = torch.cat([text_masks, audio_masks], dim=0)
        
        # Create input positions
        input_pos = torch.arange(0, tokens.size(0), device=tokens.device)
        
        # Create target tokens (shifted by 1 position)
        target_tokens = tokens[1:, :-1]  # Remove first position and text token
        
        return {
            "tokens": tokens,
            "tokens_mask": tokens_mask,
            "input_pos": input_pos,
            "target_tokens": target_tokens
        }

def train(
    model: Model,
    train_loader: DataLoader,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    device: str = "cuda",
    save_dir: str = "checkpoints",
):
    # Freeze all parameters except projection layers (w1, w2, w3)
    for name, param in model.named_parameters():
        if not any(x in name for x in ["projection", "codebook0_head", "audio_head"]):
            param.requires_grad = False
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Initialize tensorboard
    writer = SummaryWriter("runs/csm-fine-tuning")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create generator for forward pass
    generator = Generator(model)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tokens = batch["tokens"].to(device)
            tokens_mask = batch["tokens_mask"].to(device)
            input_pos = batch["input_pos"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            
            # Forward pass using generator
            logits = generator.forward(tokens, tokens_mask, input_pos)
            
            # Compute loss (only on audio tokens)
            loss = nn.CrossEntropyLoss()(
                logits[:, :-1, :-1].reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Log metrics to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    writer.close()

def main():
    # Initialize model and generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_csm_1b(device=device)
    model = generator._model
    
    # Create dataset
    train_dataset = AudioDataset("dataset", generator)
    
    # Create dataloader with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        num_epochs=1,
        learning_rate=1e-5,
        device=device
    )

if __name__ == "__main__":
    main() 