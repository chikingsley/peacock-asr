#!/usr/bin/env python3
"""
Orpheus TTS Fine-Tuning on Modal

Trains Orpheus TTS with LoRA using:
- transformers + PEFT for efficient LoRA fine-tuning
- SNAC for audio tokenization

Usage:
    # Run training on Modal
    cd modal-training && uv run modal run orpheus_modal_train.py

    # With custom parameters
    uv run modal run orpheus_modal_train.py --epochs 3 --model-name unsloth/orpheus-3b-0.1-ft

    # Using pretrained base model
    uv run modal run orpheus_modal_train.py --model-name canopylabs/orpheus-tts-0.1-pretrained
"""

import modal
from pathlib import Path

# Modal app configuration
app = modal.App("orpheus-tts-training")

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40.0",
        "datasets>=3.4.1",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "snac",
        "soundfile",
        "huggingface_hub",
        "torchaudio",
        "torchcodec",  # Required for audio decoding in datasets
        "numpy",
        "tqdm",
        "python-dotenv",
        "wandb",  # Experiment tracking
    )
)

# Volume for persistent storage (models + outputs)
volume = modal.Volume.from_name("orpheus-training-data", create_if_missing=True)

# Constants
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SNAC_SAMPLE_RATE = 24000
HF_CACHE_DIR = "/data/hf_cache"

# Special token constants for Orpheus TTS
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
END_OF_TEXT = 128009
AUDIO_TOKEN_OFFSET = 128266


# ============================================================================
# Audio Tokenization (from tokenise_speech_dataset.py)
# ============================================================================

def tokenize_audio(snac_model, waveform, sample_rate, device):
    """Tokenize audio using SNAC model."""
    import torch
    import torchaudio.transforms as T
    import numpy as np

    # Ensure waveform is a numpy array
    if isinstance(waveform, list):
        waveform = np.array(waveform, dtype=np.float32)

    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    # Resample to 24kHz if needed
    if sample_rate != 24000:
        resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to(device)

    # Generate the codes from SNAC
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Process codes into interleaved format
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))

    return all_codes


def remove_duplicate_frames(codes_list):
    """Remove duplicate frames from the codes list."""
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = codes_list[:7]

    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(codes_list[i:i+7])

    return result


def redistribute_codes(code_list, snac_model, device):
    """Convert token codes back to audio using SNAC decoder."""
    import torch

    # Initialize layers for audio code reconstruction
    layer_1 = []
    layer_2 = []
    layer_3 = []

    # Reorganize codes into specific layers
    for i in range((len(code_list)+1)//7):
        if 7*i < len(code_list):
            layer_1.append(code_list[7*i])

            if 7*i+1 < len(code_list) and 7*i+4 < len(code_list):
                layer_2.append(code_list[7*i+1] - 4096)
                layer_2.append(code_list[7*i+4] - (4*4096))

            if 7*i+2 < len(code_list) and 7*i+3 < len(code_list) and 7*i+5 < len(code_list) and 7*i+6 < len(code_list):
                layer_3.append(code_list[7*i+2] - (2*4096))
                layer_3.append(code_list[7*i+3] - (3*4096))
                layer_3.append(code_list[7*i+5] - (5*4096))
                layer_3.append(code_list[7*i+6] - (6*4096))

    # Convert layers to tensors
    codes = [
        torch.tensor(layer_1, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(layer_2, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(layer_3, dtype=torch.long).unsqueeze(0).to(device)
    ]

    # Decode audio
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    return audio_hat


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_dataset_from_hf(
    dataset_name: str,
    tokenizer,
    snac_model,
    device: str,
    hf_token: str = None,
    max_samples: int = None,
):
    """Load and tokenize a HuggingFace dataset with text + audio columns."""
    import torch
    import numpy as np
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train", token=hf_token)

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        print(f"Using {max_samples} samples for training")

    # Check if dataset is already tokenized (has input_ids)
    if "input_ids" in ds.column_names:
        print("Dataset already tokenized, using as-is")
        return ds

    # Process the dataset
    processed_data = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }

    print("Tokenizing audio samples...")
    for idx, example in enumerate(tqdm(ds)):
        try:
            text = example.get("text", "")
            source = example.get("source", "")

            # Format text with speaker name if available
            if source:
                formatted_text = f"{source}: {text}"
            else:
                formatted_text = text

            # Extract audio - handle different formats
            audio_data = example.get("audio")
            if audio_data is None:
                print(f"  Sample {idx}: no audio field")
                continue

            # Handle torchcodec AudioDecoder
            if hasattr(audio_data, "get_all_samples"):
                # torchcodec AudioDecoder - decode to get samples
                audio_samples = audio_data.get_all_samples()
                if hasattr(audio_samples, "data"):
                    audio_tensor = audio_samples.data.squeeze()
                    # Convert to mono if stereo
                    if audio_tensor.dim() > 1:
                        audio_tensor = audio_tensor.mean(dim=0)
                    audio_array = audio_tensor.numpy()
                else:
                    audio_array = np.array(audio_samples).squeeze()
                sample_rate = audio_data.metadata.sample_rate if hasattr(audio_data, "metadata") else 24000
            elif isinstance(audio_data, dict) and "array" in audio_data:
                audio_array = audio_data["array"]
                sample_rate = audio_data.get("sampling_rate", 24000)
            elif hasattr(audio_data, "array"):
                # Handle Audio feature object
                audio_array = audio_data.array
                sample_rate = getattr(audio_data, "sampling_rate", 24000)
            else:
                print(f"  Sample {idx}: unexpected audio format {type(audio_data)}")
                continue

            # Tokenize audio
            codes_list = tokenize_audio(snac_model, audio_array, sample_rate, device)
            codes_list = remove_duplicate_frames(codes_list)

            # Tokenize text
            text_tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
            text_tokens.append(END_OF_TEXT)

            # Create the complete input_ids sequence
            input_ids = (
                [START_OF_HUMAN]
                + text_tokens
                + [END_OF_HUMAN]
                + [START_OF_AI]
                + [START_OF_SPEECH]
                + codes_list
                + [END_OF_SPEECH]
                + [END_OF_AI]
            )

            attention_mask = [1] * len(input_ids)

            processed_data["input_ids"].append(input_ids)
            processed_data["labels"].append(input_ids)
            processed_data["attention_mask"].append(attention_mask)

        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    print(f"Processed {len(processed_data['input_ids'])} examples")

    from datasets import Dataset, Features, Value, Sequence
    features = Features({
        'input_ids': Sequence(Value('int64')),
        'labels': Sequence(Value('int64')),
        'attention_mask': Sequence(Value('int8'))
    })

    return Dataset.from_dict(processed_data, features=features)


# ============================================================================
# Inference
# ============================================================================

def extract_audio_from_generated_ids(generated_ids, snac_model, device):
    """Extract audio from generated token IDs containing speech tokens.

    Finds speech tokens between START_OF_SPEECH and END_OF_SPEECH,
    converts them to SNAC codes, and decodes to audio.
    """
    import torch

    # Find speech tokens (after start_of_speech, before end_of_speech)
    token_indices = (generated_ids == START_OF_SPEECH).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_idx = token_indices[1][-1].item()
        cropped = generated_ids[:, last_idx + 1:]
    else:
        cropped = generated_ids

    # Remove end_of_speech tokens and decode
    for row in cropped:
        masked = row[row != END_OF_SPEECH]
        new_len = (masked.size(0) // 7) * 7
        trimmed = masked[:new_len]
        code_list = [t.item() - AUDIO_TOKEN_OFFSET for t in trimmed]

        if len(code_list) > 0:
            audio = redistribute_codes(code_list, snac_model, device)
            return audio.detach().squeeze().cpu().numpy()

    return None


def generate_speech_from_tokens(model, input_ids, snac_model, device: str, temperature: float = 0.6, top_p: float = 0.95):
    """Generate speech from pre-tokenized input_ids (validation samples)."""
    import torch

    input_tensor = torch.tensor([input_ids], dtype=torch.int64).to(device)

    # Find where speech starts and truncate there
    speech_start_indices = (input_tensor == START_OF_SPEECH).nonzero(as_tuple=True)
    if len(speech_start_indices[1]) > 0:
        # Truncate just before speech tokens - include the start_of_speech token
        truncate_idx = speech_start_indices[1][0].item() + 1
        prompt_ids = input_tensor[:, :truncate_idx]
    else:
        prompt_ids = input_tensor

    attention_mask = torch.ones(prompt_ids.shape, dtype=torch.int64).to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=END_OF_SPEECH,
        )

    return extract_audio_from_generated_ids(generated_ids, snac_model, device)


# ============================================================================
# Training Function
# ============================================================================

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_orpheus(
    dataset_name: str = "Trelis/ronan_tts_medium_clean",
    model_name: str = "unsloth/orpheus-3b-0.1-ft",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    hub_repo_id: str = None,
    hf_token: str = None,
    max_samples: int = None,
    save_steps: int = None,
):
    """Train Orpheus TTS with LoRA on Modal."""
    import os
    import torch
    import math
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
    from huggingface_hub import login

    # Setup HF cache on volume
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/hf_cache/hub"

    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace Hub")

    # Setup wandb (from Modal secret)
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        import wandb
        wandb.login(key=wandb_api_key)
        os.environ["WANDB_PROJECT"] = "trelis-orpheus"
        print("WandB logging enabled (project: trelis-orpheus)")
    else:
        print("WandB logging disabled (no WANDB_API_KEY in secret)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path("/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/data/hf_cache",
    )

    # Load SNAC model for audio tokenization
    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained(SNAC_MODEL_ID)
    snac_model = snac_model.to(device)
    snac_model.eval()

    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset_from_hf(
        dataset_name,
        tokenizer,
        snac_model,
        device,
        hf_token=hf_token,
        max_samples=max_samples,
    )

    # Create train/validation split
    total_samples = len(dataset)
    validation_size = max(1, min(25, math.ceil(total_samples * 0.1)))
    train_size = total_samples - validation_size

    print(f"Splitting dataset: {train_size} training samples, {validation_size} validation samples")
    split_dataset = dataset.train_test_split(test_size=validation_size, seed=42)
    train_ds = split_dataset["train"]
    validation_ds = split_dataset["test"]

    # Load model
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir="/data/hf_cache",
    )

    # Define LoRA configuration (same as train-lora.py)
    print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Calculate save steps if not provided
    if save_steps is None:
        save_steps = max(1, len(train_ds) // 4)  # Save 4 times per epoch

    # Training arguments
    log_dir = str(output_dir / "logs")
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="steps",
        save_steps=save_steps,
        bf16=True,
        output_dir=str(output_dir / "checkpoints"),
        remove_unused_columns=True,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if wandb_api_key else "none",
        run_name=f"orpheus-lora-r{lora_rank}-lr{learning_rate}-ep{epochs}",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
    )

    # Train
    print("Starting LoRA training...")
    trainer.train()

    # Merge and save
    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    merged_output_dir = output_dir / "orpheus-ft"
    print(f"Saving merged model to {merged_output_dir}")
    merged_model.save_pretrained(str(merged_output_dir))
    tokenizer.save_pretrained(str(merged_output_dir))

    volume.commit()

    # Generate audio samples from validation set
    print("\nGenerating audio samples from validation set...")
    merged_model = merged_model.to(device)
    merged_model.eval()

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    num_samples = min(5, len(validation_ds))
    for i in range(num_samples):
        sample = validation_ds[i]
        print(f"Generating sample {i+1}/{num_samples}...")
        try:
            audio = generate_speech_from_tokens(
                merged_model, sample["input_ids"], snac_model, device
            )
            if audio is not None:
                sample_path = samples_dir / f"sample_{i}.wav"
                sf.write(str(sample_path), audio, 24000)
                print(f"  Saved: {sample_path}")
            else:
                print(f"  Failed to generate audio for sample {i}")
        except Exception as e:
            print(f"  Error generating sample {i}: {e}")

    volume.commit()

    # Push to HuggingFace Hub (private)
    if hub_repo_id and hf_token:
        print(f"\nPushing model to HuggingFace Hub: {hub_repo_id}")
        try:
            merged_model.push_to_hub(hub_repo_id, private=True, token=hf_token)
            tokenizer.push_to_hub(hub_repo_id, private=True, token=hf_token)
            print(f"Model pushed to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {merged_output_dir}")
    print(f"Test samples saved to: {samples_dir}")
    if hub_repo_id:
        print(f"Hub repo: https://huggingface.co/{hub_repo_id}")

    return {
        "output_dir": str(merged_output_dir),
        "samples_dir": str(samples_dir),
        "hub_repo": hub_repo_id,
    }


@app.function(image=image, volumes={"/data": volume})
def download_wav_files(subdir: str = "samples") -> list[tuple[str, bytes]]:
    """Download WAV files from a subdirectory in the Modal volume."""
    samples_dir = Path(f"/data/output/{subdir}")
    files = []
    if samples_dir.exists():
        for f in sorted(samples_dir.iterdir()):
            if f.is_file() and f.suffix == ".wav":
                files.append((f.name, f.read_bytes()))
                print(f"Prepared: {f.name}")
    return files


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={"/data": volume},
)
def generate_novel_samples():
    """Generate audio from novel test phrases using the saved model."""
    import os
    import torch
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC

    os.environ["HF_HOME"] = "/data/hf_cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = Path("/data/output/orpheus-ft")
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run training first.")

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained(SNAC_MODEL_ID).to(device)
    snac_model.eval()

    # Novel test phrases - completely different from training data
    novel_phrases = [
        "The mysterious lighthouse keeper had never seen such a peculiar storm approaching from the west.",
        "Scientists discovered a new species of butterfly in the remote mountains of Papua New Guinea.",
        "My grandmother's secret recipe for apple pie has been passed down for seven generations.",
        "The quantum computer solved the equation in three milliseconds flat.",
        "Please remember to water the plants while I'm away on my business trip to Tokyo.",
    ]

    samples_dir = Path("/data/output/novel_samples")
    samples_dir.mkdir(exist_ok=True)

    for i, text in enumerate(novel_phrases):
        print(f"Generating novel sample {i+1}: {text[:50]}...")
        try:
            audio = generate_speech_from_text(model, tokenizer, snac_model, text, device)
            if audio is not None:
                sample_path = samples_dir / f"novel_{i}.wav"
                sf.write(str(sample_path), audio, 24000)
                print(f"  Saved: {sample_path}")
            else:
                print(f"  Failed to generate audio")
        except Exception as e:
            print(f"  Error: {e}")

    volume.commit()
    return {"samples_dir": str(samples_dir), "count": len(novel_phrases)}


def generate_speech_from_text(model, tokenizer, snac_model, text: str, device: str, temperature: float = 0.6, top_p: float = 0.95):
    """Generate speech from raw text."""
    import torch

    # Encode text with special tokens
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]], dtype=torch.int64)
    prompt_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to(device)
    attention_mask = torch.ones(prompt_ids.shape, dtype=torch.int64).to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=END_OF_SPEECH,
        )

    return extract_audio_from_generated_ids(generated_ids, snac_model, device)


@app.local_entrypoint()
def main(
    dataset: str = "Trelis/ronan_tts_medium_clean",
    model_name: str = "unsloth/orpheus-3b-0.1-ft",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    hub_repo: str = "Trelis/orpheus-ft",
    max_samples: int = None,
    download: bool = True,
    generate_novel: bool = False,
):
    """Run Orpheus TTS training from local machine.

    Args:
        dataset: HuggingFace dataset with text + audio columns
        model_name: Base model (unsloth/orpheus-3b-0.1-ft or canopylabs/orpheus-tts-0.1-pretrained)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for LoRA fine-tuning
        lora_rank: LoRA rank (default: 32)
        lora_alpha: LoRA alpha (default: 64)
        hub_repo: HuggingFace Hub repo to push model (e.g., 'username/orpheus-ft-custom')
        max_samples: Limit number of training samples (for testing)
        download: Download results after training
    """
    from dotenv import load_dotenv
    import os

    # Load .env file for HF token
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print(f"HF token found: {hf_token[:10]}...")
    else:
        print("Warning: No HF_TOKEN found in .env - private datasets/repos may not work")

    print("\n" + "="*60)
    print("ORPHEUS TTS MODAL TRAINING")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print("WandB: using Modal secret 'wandb-secret'")
    if hub_repo:
        print(f"Hub repo: {hub_repo}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("="*60 + "\n")

    print("Starting training on Modal...")
    result = train_orpheus.remote(
        dataset_name=dataset,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        hub_repo_id=hub_repo,
        hf_token=hf_token,
        max_samples=max_samples,
    )
    print(f"\nTraining result: {result}")

    if download:
        print("\nDownloading audio samples...")
        samples_dir = Path("./output/samples")
        samples_dir.mkdir(parents=True, exist_ok=True)

        files = download_wav_files.remote("samples")
        for filename, content in files:
            (samples_dir / filename).write_bytes(content)
            print(f"Downloaded: samples/{filename}")

    if generate_novel:
        print("\nGenerating novel test samples...")
        generate_novel_samples.remote()

        print("Downloading novel samples...")
        novel_dir = Path("./output/novel_samples")
        novel_dir.mkdir(parents=True, exist_ok=True)

        files = download_wav_files.remote("novel_samples")
        for filename, content in files:
            (novel_dir / filename).write_bytes(content)
            print(f"Downloaded: novel_samples/{filename}")

    print("\nDone!")
