#!/usr/bin/env python3
"""
Voxtral STT Fine-Tuning on Modal

Trains Voxtral-Mini-3B for transcription with LoRA via PEFT.

Usage:
    # Run training on Modal
    cd modal-training && uv run modal run voxtral_modal_train.py

    # With custom parameters
    uv run modal run voxtral_modal_train.py --epochs 3 --max-samples 5

    # Quick test run
    uv run modal run voxtral_modal_train.py --max-samples 3 --epochs 1
"""

import modal
from pathlib import Path

app = modal.App("voxtral-stt-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.52.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "datasets[audio]>=3.4.1",
        "mistral-common[audio]>=1.8.1",
        "jiwer",
        "evaluate",
        "torchcodec",
        "huggingface_hub",
        "soundfile",
        "librosa",
        "numpy",
        "tqdm",
        "python-dotenv",
        "wandb",
    )
)

volume = modal.Volume.from_name("voxtral-training-data", create_if_missing=True)

DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
HF_CACHE_DIR = "/data/hf_cache"


def _find_transcription_method(processor):
    """Find the correct transcription method on the processor.

    Some transformers versions have a typo in the method name.
    """
    for method_name in [
        "apply_transcription_request",
        "apply_transcrition_request",
    ]:
        if hasattr(processor, method_name):
            return getattr(processor, method_name), method_name
    available = [m for m in dir(processor) if "transcri" in m.lower()]
    raise AttributeError(
        f"Processor has no transcription method. Available: {available}"
    )


class VoxtralDataCollator:
    """Collator that builds transcription prompts with mel spectrograms.

    For each sample:
    1. Uses processor.apply_transcription_request to build prompt tokens + mel specs
    2. Tokenizes ground-truth text separately
    3. Concatenates: prompt_ids + text_ids + EOS
    4. Labels: -100 for prompt positions, real IDs for text + EOS
    5. Pads batch to max length
    """

    def __init__(self, processor, tokenizer, model_id, language="en", max_text_length=256):
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.language = language
        self.max_text_length = max_text_length

        self.transcription_method, method_name = _find_transcription_method(processor)
        print(f"Using transcription method: {method_name}")

    def _extract_audio(self, audio_data):
        """Extract audio array and sample rate from various formats.

        Handles: torchcodec AudioDecoder, HF datasets dict, legacy objects.
        """
        import numpy as np

        # torchcodec AudioDecoder (from datasets with torchcodec backend)
        if hasattr(audio_data, "get_all_samples"):
            audio_samples = audio_data.get_all_samples()
            if hasattr(audio_samples, "data"):
                audio_tensor = audio_samples.data.squeeze()
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.mean(dim=0)  # stereo → mono
                audio_array = audio_tensor.numpy()
            else:
                audio_array = np.array(audio_samples).squeeze()
            sr = audio_data.metadata.sample_rate if hasattr(audio_data, "metadata") else 16000
            return audio_array, sr

        # Standard HF datasets dict format
        if isinstance(audio_data, dict) and "array" in audio_data:
            return audio_data["array"], audio_data.get("sampling_rate", 16000)

        # Legacy object format
        if hasattr(audio_data, "array"):
            return audio_data.array, getattr(audio_data, "sampling_rate", 16000)

        return None, None

    def _ensure_numpy(self, audio_array):
        """Convert to float32 numpy array if needed."""
        import numpy as np
        if isinstance(audio_array, list):
            return np.array(audio_array, dtype=np.float32)
        if hasattr(audio_array, 'numpy'):
            return audio_array.numpy().astype('float32')
        if hasattr(audio_array, 'astype'):
            return audio_array.astype('float32')
        return audio_array

    def _resample_if_needed(self, audio_array, sr):
        """Resample to 16kHz if needed (Voxtral expects 16kHz)."""
        if sr == 16000:
            return audio_array
        import librosa
        import numpy as np
        audio_array = self._ensure_numpy(audio_array)
        return librosa.resample(
            audio_array.astype(np.float32), orig_sr=sr, target_sr=16000
        )

    def __call__(self, features):
        import torch
        import soundfile as sf

        batch_input_ids = []
        batch_labels = []
        batch_input_features = []

        for i, feature in enumerate(features):
            audio_data = feature.get("audio")
            text = feature.get("text", feature.get("transcription", ""))

            if audio_data is None:
                continue

            audio_array, sr = self._extract_audio(audio_data)
            if audio_array is None:
                continue

            audio_array = self._resample_if_needed(audio_array, sr)
            audio_array = self._ensure_numpy(audio_array)

            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, audio_array, 16000)
                    prompt_inputs = self.transcription_method(
                        language=self.language,
                        audio=tmp.name,
                        model_id=self.model_id,
                    )
            except Exception as e:
                print(f"Collator: skipping sample {i}: {e}")
                continue

            prompt_ids = prompt_inputs.input_ids[0]  # [seq_len]
            input_features = prompt_inputs.input_features[0]  # mel spectrogram

            # Tokenize ground-truth text (no special tokens)
            text_encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=self.max_text_length,
                truncation=True,
                return_tensors="pt",
            )
            text_ids = text_encoding.input_ids[0]  # [text_len]

            # EOS token
            eos_id = self.tokenizer.eos_token_id
            eos_tensor = torch.tensor([eos_id], dtype=text_ids.dtype)

            # Concatenate: prompt + text + EOS
            full_ids = torch.cat([prompt_ids, text_ids, eos_tensor], dim=0)

            # Labels: -100 for prompt, real IDs for text + EOS
            prompt_labels = torch.full_like(prompt_ids, -100)
            full_labels = torch.cat([prompt_labels, text_ids, eos_tensor], dim=0)

            batch_input_ids.append(full_ids)
            batch_labels.append(full_labels)
            batch_input_features.append(input_features)

        if not batch_input_ids:
            raise ValueError("All samples in batch failed collation")

        # Pad to max length in batch
        max_len = max(ids.size(0) for ids in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for ids, labels in zip(batch_input_ids, batch_labels):
            pad_len = max_len - ids.size(0)
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
            )
            attention_masks.append(
                torch.cat([torch.ones(ids.size(0), dtype=torch.long),
                           torch.zeros(pad_len, dtype=torch.long)])
            )

        # Pad input_features along time dimension
        max_feat_len = max(f.size(-1) for f in batch_input_features)
        padded_features = [
            torch.nn.functional.pad(f, (0, max_feat_len - f.size(-1)), value=0.0)
            for f in batch_input_features
        ]

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "input_features": torch.stack(padded_features),
        }


def _create_trainer(model, training_args, train_dataset, eval_dataset, data_collator):
    """Create Trainer with workaround for _signature_columns being None in transformers v5+."""
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if trainer._signature_columns is None:
        trainer._signature_columns = []
    return trainer


def run_wer_evaluation(model, processor, tokenizer, val_ds, device, model_id, collator, label=""):
    import torch
    import tempfile
    import soundfile as sf

    transcription_method, _ = _find_transcription_method(processor)
    model_on_device = model.to(device)
    model_on_device.eval()
    predictions = []
    references = []

    for i in range(len(val_ds)):
        sample = val_ds[i]
        text = sample.get("text", sample.get("transcription", ""))
        audio_data = sample.get("audio")

        if audio_data is None:
            continue

        audio_array, sr = collator._extract_audio(audio_data)
        if audio_array is None:
            continue
        audio_array = collator._resample_if_needed(audio_array, sr)
        audio_array = collator._ensure_numpy(audio_array)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_array, 16000)
                inputs = transcription_method(
                    language="en",
                    audio=tmp.name,
                    model_id=model_id,
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model_on_device.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                )

            prompt_len = inputs["input_ids"].shape[1]
            new_ids = generated_ids[0, prompt_len:]
            prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            predictions.append(prediction)
            references.append(text)
            print(f"  [{i}] REF: {text}")
            print(f"       HYP: {prediction}")

        except Exception as e:
            print(f"  [{i}] Error: {e}")
            continue

    if predictions:
        from jiwer import wer
        word_error_rate = wer(references, predictions)
        print(f"\n{label}WER: {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
        print(f"Evaluated {len(predictions)} / {len(val_ds)} samples")
        return word_error_rate

    print("No predictions generated")
    return None


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
)
def train_voxtral(
    dataset_name: str = "Trelis/llm-lingo",
    model_name: str = DEFAULT_MODEL_ID,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    hub_repo_id: str = None,
    hf_token: str = None,
    max_samples: int = None,
):
    """Train Voxtral STT with LoRA on Modal."""
    import os
    import torch
    from transformers import (
        VoxtralForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from huggingface_hub import login

    # Setup HF cache on volume
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_DIR}/hub"

    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace Hub")

    # Setup wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        import wandb
        wandb.login(key=wandb_api_key)
        os.environ["WANDB_PROJECT"] = "voxtral-stt"
        print("WandB logging enabled (project: voxtral-stt)")
    else:
        print("WandB logging disabled (no WANDB_API_KEY)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path("/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processor and tokenizer
    print(f"Loading processor from {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    tokenizer = processor.tokenizer

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, token=hf_token, cache_dir=HF_CACHE_DIR)

    # Use predefined splits if available
    if "train" in ds and "validation" in ds:
        train_ds = ds["train"]
        val_ds = ds["validation"]
        print(f"Using dataset splits: {len(train_ds)} train, {len(val_ds)} val")
    elif "train" in ds:
        split = ds["train"].train_test_split(test_size=0.3, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
        print(f"Split dataset: {len(train_ds)} train, {len(val_ds)} val")
    else:
        raise ValueError(f"Unexpected dataset splits: {list(ds.keys())}")

    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(max_samples, len(val_ds))))
        print(f"Limited to: {len(train_ds)} train, {len(val_ds)} val")

    # Load model
    print(f"Loading model: {model_name}...")
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
    )

    # Create collator early to reuse its audio helpers
    collator = VoxtralDataCollator(
        processor=processor,
        tokenizer=tokenizer,
        model_id=model_name,
    )

    # --- Baseline WER (before training) ---
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (before training)")
    print("=" * 60)

    baseline_wer = run_wer_evaluation(
        model, processor, tokenizer, val_ds, device, model_name, collator, label="BASELINE "
    )
    model = model.cpu()

    # Freeze audio encoder (direct attribute on VoxtralForConditionalGeneration)
    print("\nFreezing audio tower...")
    for param in model.audio_tower.parameters():
        param.requires_grad = False

    # Apply LoRA
    print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        output_dir=str(output_dir / "checkpoints"),
        remove_unused_columns=False,  # Critical: collator needs raw columns
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        load_best_model_at_end=False,
        report_to="wandb" if wandb_api_key else "none",
        run_name=f"voxtral-lora-r{lora_rank}-lr{learning_rate}-ep{epochs}",
        dataloader_pin_memory=False,
    )

    trainer = _create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # Train
    print("Starting LoRA training...")
    trainer.train()

    # Merge and save
    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    merged_output_dir = output_dir / "voxtral-ft"
    print(f"Saving merged model to {merged_output_dir}")
    merged_model.save_pretrained(str(merged_output_dir))
    processor.save_pretrained(str(merged_output_dir))
    volume.commit()

    # Post-training WER evaluation
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    word_error_rate = run_wer_evaluation(
        merged_model, processor, tokenizer, val_ds, device, model_name, collator, label="FINE-TUNED "
    )

    volume.commit()

    # Push to HuggingFace Hub
    if hub_repo_id and hf_token:
        print(f"\nPushing model to HuggingFace Hub: {hub_repo_id}")
        try:
            merged_model.push_to_hub(hub_repo_id, private=True, token=hf_token)
            processor.push_to_hub(hub_repo_id, private=True, token=hf_token)
            print(f"Model pushed to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {merged_output_dir}")
    if baseline_wer is not None:
        print(f"Baseline WER (before): {baseline_wer:.4f} ({baseline_wer*100:.1f}%)")
    if word_error_rate is not None:
        print(f"Fine-tuned WER (after): {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
    if baseline_wer is not None and word_error_rate is not None:
        improvement = baseline_wer - word_error_rate
        print(f"Improvement: {improvement:.4f} ({improvement*100:.1f}pp)")
    if hub_repo_id:
        print(f"Hub repo: https://huggingface.co/{hub_repo_id}")

    return {
        "output_dir": str(merged_output_dir),
        "hub_repo": hub_repo_id,
        "baseline_wer": baseline_wer,
        "finetuned_wer": word_error_rate,
    }


@app.function(image=image, volumes={"/data": volume})
def download_files(subdir: str = "voxtral-ft") -> list[tuple[str, bytes]]:
    """Download files from a subdirectory in the Modal volume."""
    target_dir = Path(f"/data/output/{subdir}")
    if not target_dir.exists():
        return []
    files = [
        (f.name, f.read_bytes())
        for f in sorted(target_dir.iterdir())
        if f.is_file() and f.stat().st_size < 50_000_000
    ]
    for name, _ in files:
        print(f"Prepared: {name}")
    return files


@app.local_entrypoint()
def main(
    dataset: str = "Trelis/llm-lingo",
    model_name: str = DEFAULT_MODEL_ID,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    hub_repo: str = "Trelis/voxtral-ft",
    max_samples: int = None,
    download: bool = False,
):
    """Run Voxtral STT training from local machine.

    Args:
        dataset: HuggingFace dataset with audio + text columns
        model_name: Base model ID
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for LoRA
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA alpha (default: 32)
        hub_repo: HuggingFace Hub repo for model push
        max_samples: Limit samples (for testing)
        download: Download model files after training
    """
    from dotenv import load_dotenv
    import os

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print("HF token found")
    else:
        print("Warning: No HF_TOKEN found in .env")

    print("\n" + "=" * 60)
    print("VOXTRAL STT MODAL TRAINING")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print("WandB: enabled if 'wandb-secret' Modal secret is configured")
    if hub_repo:
        print(f"Hub repo: {hub_repo}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("=" * 60 + "\n")

    print("Starting training on Modal...")
    result = train_voxtral.remote(
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
        print("\nDownloading model files...")
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        files = download_files.remote("voxtral-ft")
        for filename, content in files:
            (output_dir / filename).write_bytes(content)
            print(f"Downloaded: {filename}")

    print("\nDone!")
