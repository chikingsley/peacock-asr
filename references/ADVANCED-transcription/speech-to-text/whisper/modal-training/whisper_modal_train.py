#!/usr/bin/env python3
"""
Whisper STT Fine-Tuning on Modal

Trains Whisper-large-v3-turbo with LoRA via PEFT + Seq2SeqTrainer.
Measures WER before and after training for direct comparison with Voxtral.

Usage:
    # Run training on Modal
    cd modal-training && uv run modal run whisper_modal_train.py

    # With custom parameters
    uv run modal run whisper_modal_train.py --epochs 2 --max-samples 5
"""

import modal
from pathlib import Path

app = modal.App("whisper-stt-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.52.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "datasets[audio]>=3.4.1",
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

volume = modal.Volume.from_name("whisper-training-data", create_if_missing=True)

DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"
HF_CACHE_DIR = "/data/hf_cache"


def _extract_audio(audio_data):
    """Extract audio array and sample rate from various formats."""
    import numpy as np

    # torchcodec AudioDecoder
    if hasattr(audio_data, "get_all_samples"):
        audio_samples = audio_data.get_all_samples()
        if hasattr(audio_samples, "data"):
            audio_tensor = audio_samples.data.squeeze()
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            audio_array = audio_tensor.numpy()
        else:
            audio_array = np.array(audio_samples).squeeze()
        sr = audio_data.metadata.sample_rate if hasattr(audio_data, "metadata") else 16000
        return audio_array, sr

    # Standard HF datasets dict
    if isinstance(audio_data, dict) and "array" in audio_data:
        return audio_data["array"], audio_data.get("sampling_rate", 16000)

    # Legacy object
    if hasattr(audio_data, "array"):
        return audio_data.array, getattr(audio_data, "sampling_rate", 16000)

    return None, None


def _ensure_numpy(audio_array):
    """Convert to float32 numpy array if needed."""
    import numpy as np
    if isinstance(audio_array, list):
        return np.array(audio_array, dtype=np.float32)
    if hasattr(audio_array, 'numpy'):
        return audio_array.numpy().astype('float32')
    if hasattr(audio_array, 'astype'):
        return audio_array.astype('float32')
    return audio_array


def _resample_if_needed(audio_array, sr, target_sr=16000):
    """Resample to target sample rate if needed."""
    if sr == target_sr:
        return audio_array
    import librosa
    audio_array = _ensure_numpy(audio_array)
    return librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)


def prepare_dataset(ds_split, processor):
    """Prepare dataset split into input_features + labels format for Whisper training."""
    import tqdm

    prepared = []
    for sample in tqdm.tqdm(ds_split, desc="Preparing data"):
        audio_data = sample.get("audio")
        text = sample.get("text", sample.get("transcription", ""))

        if audio_data is None:
            continue

        audio_array, sr = _extract_audio(audio_data)
        if audio_array is None:
            continue

        audio_array = _resample_if_needed(audio_array, sr, 16000)
        audio_array = _ensure_numpy(audio_array)

        features = processor.feature_extractor(
            audio_array, sampling_rate=16000
        )
        tokenized_text = processor.tokenizer(text)

        prepared.append({
            "input_features": features.input_features[0],
            "labels": tokenized_text.input_ids,
        })

    return prepared


def run_wer_evaluation(model, processor, val_ds, device, label=""):
    """Run WER evaluation on validation set using generate."""
    import torch

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

        audio_array, sr = _extract_audio(audio_data)
        if audio_array is None:
            continue

        audio_array = _resample_if_needed(audio_array, sr, 16000)
        audio_array = _ensure_numpy(audio_array)

        try:
            inputs = processor.feature_extractor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                generated_ids = model_on_device.generate(
                    input_features=input_features,
                    max_new_tokens=256,
                    language="en",
                    task="transcribe",
                )

            prediction = processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

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
    timeout=7200,
    volumes={"/data": volume},
)
def train_whisper(
    dataset_name: str = "Trelis/llm-lingo",
    model_name: str = DEFAULT_MODEL_ID,
    epochs: int = 2,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    hub_repo_id: str = None,
    hf_token: str = None,
    max_samples: int = None,
):
    """Train Whisper STT with LoRA on Modal."""
    import os
    import torch
    import numpy as np
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
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
        os.environ["WANDB_PROJECT"] = "whisper-stt"
        print("WandB logging enabled (project: whisper-stt)")
    else:
        print("WandB logging disabled (no WANDB_API_KEY)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path("/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processor
    print(f"Loading processor from {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, token=hf_token, cache_dir=HF_CACHE_DIR)

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
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
    )

    # Configure generation
    model.generation_config.language = "<|en|>"
    model.generation_config.task = "transcribe"
    model.generation_config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    # --- Baseline WER (before training) ---
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (before training)")
    print("=" * 60)

    baseline_wer = run_wer_evaluation(model, processor, val_ds, device, label="BASELINE ")
    model = model.cpu()

    # Apply LoRA
    print(f"\nApplying LoRA (rank={lora_rank}, alpha={lora_alpha}, RSLoRA=True)...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # Prepare datasets
    print("\nPreparing training data...")
    train_prepared = prepare_dataset(train_ds, processor)
    val_prepared = prepare_dataset(val_ds, processor)
    print(f"Prepared: {len(train_prepared)} train, {len(val_prepared)} val samples")

    # Data collator
    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            batch["input_features"] = batch["input_features"].to(torch.bfloat16)

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # compute_metrics
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if len(pred_ids.shape) == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        from jiwer import wer
        return {"wer": wer(label_str, pred_str)}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        bf16=True,
        logging_strategy="steps",
        logging_steps=1,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        save_strategy="epoch",
        lr_scheduler_type="constant",
        report_to="wandb" if wandb_api_key else "none",
        run_name=f"whisper-lora-r{lora_rank}-lr{learning_rate}-ep{epochs}",
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_prepared,
        eval_dataset=val_prepared,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\nStarting LoRA training...")
    trainer.train()

    # Merge and save
    print("Merging LoRA weights with base model...")
    model.config.use_cache = True
    merged_model = model.merge_and_unload()

    merged_output_dir = output_dir / "whisper-ft"
    print(f"Saving merged model to {merged_output_dir}")
    merged_model.save_pretrained(str(merged_output_dir))
    processor.save_pretrained(str(merged_output_dir))
    volume.commit()

    # Post-training WER evaluation
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    finetuned_wer = run_wer_evaluation(merged_model, processor, val_ds, device, label="FINE-TUNED ")

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
    if finetuned_wer is not None:
        print(f"Fine-tuned WER (after): {finetuned_wer:.4f} ({finetuned_wer*100:.1f}%)")
    if baseline_wer is not None and finetuned_wer is not None:
        improvement = baseline_wer - finetuned_wer
        print(f"Improvement: {improvement:.4f} ({improvement*100:.1f}pp)")
    if hub_repo_id:
        print(f"Hub repo: https://huggingface.co/{hub_repo_id}")

    return {
        "output_dir": str(merged_output_dir),
        "hub_repo": hub_repo_id,
        "baseline_wer": baseline_wer,
        "finetuned_wer": finetuned_wer,
    }


@app.function(image=image, volumes={"/data": volume})
def download_files(subdir: str = "whisper-ft") -> list[tuple[str, bytes]]:
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
    epochs: int = 2,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    hub_repo: str = "Trelis/whisper-ft",
    max_samples: int = None,
    download: bool = False,
):
    """Run Whisper STT training from local machine."""
    from dotenv import load_dotenv
    import os

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print("HF token found")
    else:
        print("Warning: No HF_TOKEN found in .env")

    print("\n" + "=" * 60)
    print("WHISPER STT MODAL TRAINING")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}, RSLoRA: True")
    if hub_repo:
        print(f"Hub repo: {hub_repo}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("=" * 60 + "\n")

    print("Starting training on Modal...")
    result = train_whisper.remote(
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

        files = download_files.remote("whisper-ft")
        for filename, content in files:
            (output_dir / filename).write_bytes(content)
            print(f"Downloaded: {filename}")

    print("\nDone!")
