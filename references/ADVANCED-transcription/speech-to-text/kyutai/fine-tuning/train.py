#!/usr/bin/env python3
"""
Kyutai STT Fine-tuning Script

Usage:
    uv run train.py --config config.yaml
"""

# Disable XET before any HuggingFace imports (must be first)
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import gc
import pickle
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch
import wandb
import yaml
from datasets import Audio, load_dataset
from jiwer import wer
from torch.utils.data import DataLoader
from transformers import (
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from data_collator import TimeAlignedDataCollator
from forced_alignment import get_word_timestamps_with_original
from timestamp_alignment import schedule_tokens_discrete


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_config: str) -> str:
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_config


def format_seconds(seconds: float) -> str:
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}:{seconds % 60:06.3f}"


def process_sample(sample, processor, model, config, fa_device="cpu"):
    """Process a single sample: use ground truth text + forced alignment."""
    try:
        audio_data = sample["audio"]
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_array, sample_rate = sf.read(BytesIO(audio_data["bytes"]))
        else:
            return None

        # Convert to mono float32
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        # Use ground truth text from dataset
        text = sample["text"]

        # Resample for forced alignment (16kHz) and Kyutai (24kHz)
        if sample_rate != 16000:
            audio_16k = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_array.copy()

        if sample_rate != 24000:
            audio_24k = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=24000)
        else:
            audio_24k = audio_array

        # Get word timestamps using forced alignment (preserves original words)
        print("  Running forced alignment...")
        waveform = torch.tensor(audio_16k, dtype=torch.float32)
        word_timestamps = get_word_timestamps_with_original(
            waveform, 16000, text, device=fa_device
        )

        # Encode audio with codec
        print("  Encoding with codec...")
        audio_inputs = processor(audio_24k, sampling_rate=24000)
        input_values = audio_inputs['input_values'].to(model.device)
        padding_mask = audio_inputs.get('padding_mask')
        if padding_mask is not None:
            padding_mask = padding_mask.to(model.device)

        with torch.no_grad():
            codec_output = model.codec_model.encode(input_values, padding_mask=padding_mask)
            audio_codes = codec_output.audio_codes.transpose(1, 2).cpu()

        # NOTE: Codec encoding CAN be batched for ~10-20% speedup. The processor and codec
        # both support batching with proper padding masks. However, since forced alignment
        # (MMS_FA) is inherently sequential and dominates preprocessing time, batching the
        # codec step provides limited benefit. See KYUTAI_FINETUNING_EXPLAINED.md for details.
        #
        # Batched alternative (would require restructuring the preprocessing loop):
        #
        # batch_audios = [sample['audio_24k'] for sample in aligned_samples]
        # batch_inputs = processor(batch_audios, sampling_rate=24000, padding=True, return_tensors="pt")
        # input_values = batch_inputs['input_values'].to(model.device)
        # padding_mask = batch_inputs.get('padding_mask')
        # if padding_mask is not None:
        #     padding_mask = padding_mask.to(model.device)
        #
        # with torch.no_grad():
        #     codec_output = model.codec_model.encode(input_values, padding_mask=padding_mask)
        #     # codec_output.audio_codes: [batch_size, 32, max_seq_len]
        #
        # # Strip padding to restore natural lengths before saving:
        # for i, sample in enumerate(aligned_samples):
        #     real_frames = calculate_real_frame_count(padding_mask[i], hop_samples)
        #     sample['audio_codes'] = codec_output.audio_codes[i, :, :real_frames].transpose(0, 1)

        # Generate token schedule (using original text with proper word timestamps)
        seq_len = audio_codes.shape[1]
        frame_hop_s = config["processing"]["frame_hop_s"]
        token_schedule = schedule_tokens_discrete(
            {"text": text, "word_timestamps": word_timestamps,
             "start_time": "00:00:00", "end_time": format_seconds(seq_len * frame_hop_s)},
            processor.tokenizer,
            stt_delay=config["processing"]["stt_delay"],
            frame_hop_s=frame_hop_s,
            pad_to_segment_end=True, spillover="shift"
        )

        # Cleanup GPU
        del input_values, padding_mask, codec_output
        if model.device.type in ("mps", "cuda"):
            getattr(torch, model.device.type).empty_cache()

        return {
            "text": text,  # Original text with proper casing/punctuation
            "word_timestamps": word_timestamps,
            "token_schedule": token_schedule,
            "processed_audio": {"array": audio_24k.tolist(), "sampling_rate": 24000},
            "audio_codes": audio_codes.squeeze(0).tolist(),
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_or_process_data(config, processor, model, split=None):
    """Load processed data from cache or process from scratch."""
    dataset_config = config["dataset"]
    split = split or dataset_config["split"]
    cache_path = dataset_config.get("processed_data_path", "./processed_data.pkl")
    # Use different cache for validation
    if split != dataset_config["split"]:
        cache_path = cache_path.replace(".pkl", f"_{split}.pkl")
    force_reprocess = dataset_config.get("force_reprocess", False)

    # Try loading from cache
    if not force_reprocess and Path(cache_path).exists():
        print(f"Loading processed {split} data from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Process from scratch using ground truth + forced alignment
    print(f"Processing {split} dataset from scratch...")

    # Forced alignment device (MMS_FA works on CPU)
    fa_device = "cpu"
    print(f"Forced alignment running on: {fa_device}")

    dataset = load_dataset(dataset_config["name"], split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    max_samples = dataset_config.get("max_samples") or len(dataset)
    num_samples = min(max_samples, len(dataset))

    processed = []
    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}...")
        result = process_sample(dataset[i], processor, model, config, fa_device)
        if result:
            processed.append(result)
        gc.collect()

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(processed, f)

    return processed


class KyutaiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract loss_weights before passing to model
        loss_weights = inputs.pop('loss_weights', None)

        outputs = model(**inputs)

        if loss_weights is not None:
            # Compute weighted loss manually
            # Model uses internal shifting: logits[t] predicts labels[t+1]
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            labels = inputs['labels']  # [batch, seq_len]

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_weights = loss_weights[:, 1:].contiguous()

            # Flatten
            batch_size, seq_len, vocab_size = shift_logits.shape
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_weights = shift_weights.view(-1)

            # Compute per-token loss (no reduction)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(flat_logits, flat_labels)

            # Apply weights and average
            weighted_loss = per_token_loss * flat_weights
            # Only count positions with non-zero weight
            num_active = (flat_weights > 0).sum().float()
            loss = weighted_loss.sum() / num_active.clamp(min=1.0)

            # Replace the loss in outputs
            outputs.loss = loss

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.per_device_train_batch_size,
            shuffle=True, collate_fn=self.data_collator, num_workers=0
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset, batch_size=self.args.per_device_eval_batch_size,
            shuffle=False, collate_fn=self.data_collator, num_workers=0
        )


class KyutaiTrainerWithWER(KyutaiTrainer):
    """Trainer that computes teacher-forced WER during evaluation."""

    def __init__(self, processor=None, raw_eval_samples=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.raw_eval_samples = raw_eval_samples or []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Run standard evaluation first
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Compute teacher-forced WER
        if self.raw_eval_samples:
            self.model.eval()
            wer_scores = []

            for sample in self.raw_eval_samples:
                ground_truth = sample["text"]
                batch = self.data_collator([sample])
                # Remove loss_weights before passing to model
                batch.pop('loss_weights', None)

                with torch.no_grad():
                    outputs = self.model.forward(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                pred_text = self.processor.tokenizer.decode(predictions[0], skip_special_tokens=True)

                sample_wer = wer(ground_truth, pred_text) if ground_truth and pred_text else 1.0
                wer_scores.append(sample_wer)

            avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0
            metrics[f"{metric_key_prefix}_teacher_forced_wer"] = avg_wer
            self.log(metrics)

        return metrics


def setup_model(model, config):
    """Setup model with LoRA or full fine-tuning."""
    lora_config = config.get("lora", {})
    use_lora = lora_config.get("use_lora", False)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=lora_config.get("rank", 32),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get("target_modules", ["fc1", "fc2", "k_proj", "q_proj", "v_proj"]),
            lora_dropout=lora_config.get("dropout", 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=lora_config.get("use_rslora", True)
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("LoRA adapters applied!")
    else:
        # Full fine-tuning - all parameters trainable
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable:,} trainable parameters")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["model"]["device"])
    print(f"Device: {device}")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Load model
    model_id = config["model"]["model_id"]
    dataset_name = config["dataset"]["name"]
    print(f"Loading model: {model_id}")
    processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
    model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=device)

    # Initialize wandb
    model_short = model_id.split("/")[-1]
    dataset_short = dataset_name.split("/")[-1]
    lora_config = config.get("lora", {})
    lora_str = f"lora-r{lora_config.get('rank', 32)}" if lora_config.get("use_lora") else "full"
    run_name = f"{model_short}-{dataset_short}-{lora_str}"

    wandb.init(
        project="kyutai-stt",
        name=run_name,
        config={
            "model_id": model_id,
            "dataset": dataset_name,
            "learning_rate": config["training"]["learning_rate"],
            "num_epochs": config["training"]["num_epochs"],
            "batch_size": config["training"]["batch_size"],
            "lora_enabled": lora_config.get("use_lora", False),
            "lora_rank": lora_config.get("rank", 32),
            "lora_alpha": lora_config.get("alpha", 32),
            "stt_delay": config["processing"]["stt_delay"],
        }
    )

    # Load or process train data
    processed_samples = load_or_process_data(config, processor, model)
    print(f"Loaded {len(processed_samples)} train samples")

    # Load or process validation data (same pipeline as train)
    val_samples = load_or_process_data(config, processor, model, split="validation")
    print(f"Loaded {len(val_samples)} validation samples")

    # Setup model for training
    model = setup_model(model, config)

    # Create collator
    collator = TimeAlignedDataCollator(
        processor=processor, model=model,
        stt_delay=config["processing"]["stt_delay"],
        frame_hop_s=config["processing"]["frame_hop_s"],
        pad_weight=config["processing"].get("pad_weight", 0.5),
    )

    # Training
    tc = config["training"]
    training_args = TrainingArguments(
        output_dir=tc["output_dir"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        num_train_epochs=tc["num_epochs"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc.get("lr_scheduler_type", "linear"),
        logging_steps=tc["logging_steps"],
        eval_strategy=tc.get("eval_strategy", "no"),
        save_strategy="no",  # Don't save checkpoints, only final model
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = KyutaiTrainerWithWER(
        model=model, args=training_args,
        train_dataset=processed_samples,
        eval_dataset=val_samples if val_samples else None,
        data_collator=collator,
        processor=processor,
        raw_eval_samples=val_samples,
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete!")

    wandb.finish()

    # Merge LoRA weights if used
    if config.get("lora", {}).get("use_lora", False):
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    # Save final model
    print(f"Saving model to {tc['output_dir']}...")
    model.save_pretrained(tc["output_dir"])
    processor.save_pretrained(tc["output_dir"])
    print("Done!")


if __name__ == "__main__":
    main()
