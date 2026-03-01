#!/usr/bin/env python3
"""
Chatterbox TTS Fine-Tuning on Modal

Trains Chatterbox TTS T3 model with LoRA using:
- chatterbox-tts for model loading, preprocessing, and inference
- PEFT for efficient LoRA fine-tuning
- HF Trainer for the training loop

Only T3 (text -> speech tokens) is trained. S3Gen and VoiceEncoder are frozen.
Voice cloning is zero-shot (built into Chatterbox inference, no training needed).

Supports two model variants:
- ResembleAI/chatterbox-turbo (350M, GPT-2 backbone) — default, smaller/faster
- ResembleAI/chatterbox (500M, Llama backbone) — multilingual

Usage:
    cd text-to-speech/chatterbox/modal-training
    uv run modal run --env=dev chatterbox_modal_train.py --max-samples 5 --epochs 3

    # Full training
    uv run modal run chatterbox_modal_train.py

    # With custom parameters
    uv run modal run chatterbox_modal_train.py --model-name ResembleAI/chatterbox --language fr
"""

import modal
from pathlib import Path

# Modal app configuration
app = modal.App("chatterbox-tts-training")

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "chatterbox-tts",
        "torch>=2.1",
        "torchaudio>=2.1",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "datasets==3.3.2",
        "wandb",
        "soundfile",
        "numpy",
        "tqdm",
        "python-dotenv",
        "huggingface_hub",
    )
    # Avoid SDPA bug with voice references in some transformers versions
    .env({"TRANSFORMERS_ATTN_IMPLEMENTATION": "eager"})
)

# Volume for persistent storage (models + outputs + cache)
volume = modal.Volume.from_name("chatterbox-training-data", create_if_missing=True)

# Constants
SPEAKER_EMB_DROPOUT_PROB = 0.2
CB_SAMPLE_RATE = 24000
HF_CACHE_DIR = "/data/hf_cache"


# ============================================================================
# Model Inspection
# ============================================================================

def inspect_model(cb_model):
    """Print essential Chatterbox model info."""
    print("\n" + "=" * 60)
    print("CHATTERBOX MODEL INSPECTION")
    print("=" * 60)

    print(f"\nModel type: {type(cb_model).__name__}")
    print(f"Sample rate: {getattr(cb_model, 'sr', 'unknown')}")

    for name, label in [("t3", "T3"), ("ve", "VoiceEncoder"), ("s3gen", "S3Gen")]:
        component = getattr(cb_model, name, None)
        if component:
            params = sum(p.numel() for p in component.parameters())
            extra = f" type: {type(component).__name__}," if name == "t3" else ""
            print(f"{label}{extra} parameters: {params:,}")
        elif name == "t3":
            print("WARNING: No 't3' attribute found!")

    print("=" * 60 + "\n")


# ============================================================================
# Preprocessing: HF dataset -> tensors for T3 training
# ============================================================================

def extract_speaker_embedding(cb_model, audio_tensor, sample_rate):
    """Extract speaker embedding from audio using ChatterboxTTS.

    The VoiceEncoder (cb_model.ve) expects mel spectrograms, not raw audio.
    Strategy:
    1. Try prepare_conditionals (saves to temp file) and check cb_model.conds
    2. If that fails, compute mel spectrograms and call ve directly
    """
    import torch
    import torchaudio
    import tempfile
    import soundfile as sf

    device = cb_model.device
    ve = cb_model.ve

    # Strategy 1: Use prepare_conditionals and inspect cb_model.conds
    try:
        audio_np = audio_tensor.squeeze().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, sample_rate)
            temp_path = Path(f.name)

        try:
            cb_model.prepare_conditionals(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

        # After prepare_conditionals, check cb_model.conds for speaker embedding
        conds = cb_model.conds
        if conds is not None:
            if isinstance(conds, torch.Tensor):
                return conds.squeeze().cpu().float()
            elif isinstance(conds, dict):
                for key in ['speaker_emb', 'spk_emb', 've', 'cond']:
                    if key in conds and isinstance(conds[key], torch.Tensor):
                        return conds[key].squeeze().cpu().float()
            elif hasattr(conds, '__dict__'):
                for val in conds.__dict__.values():
                    if isinstance(val, torch.Tensor) and val.dim() <= 2:
                        emb = val.squeeze().cpu().float()
                        if emb.dim() == 1 and 128 <= emb.shape[0] <= 512:
                            return emb
    except Exception as e:
        print(f"  prepare_conditionals failed: {e}, falling back to manual mel extraction")

    # Strategy 2: Compute mel spectrograms and call VE directly
    # VoiceEncoder LSTM expects (batch, time, 40) - 40 mel bands at 16kHz
    VE_SAMPLE_RATE = 16000
    VE_N_MELS = 40

    # Resample to 16kHz (standard for Resemblyzer-style voice encoders)
    if sample_rate != VE_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, VE_SAMPLE_RATE)
        audio_16k = resampler(audio_tensor.cpu()).to(device)
    else:
        audio_16k = audio_tensor.to(device)

    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=VE_SAMPLE_RATE,
        n_mels=VE_N_MELS,
        n_fft=400,       # 25ms window
        hop_length=160,  # 10ms hop
        win_length=400,
    ).to(device)

    mels = mel_transform(audio_16k)  # (1, 40, T)
    # Log scale
    mels = torch.log(mels.clamp(min=1e-5))
    # Transpose to (1, T, 40) for LSTM
    mels = mels.transpose(1, 2)

    with torch.no_grad():
        emb = ve(mels)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = emb.squeeze().cpu().float()

    return emb


def extract_speech_tokens(cb_model, audio_tensor, sample_rate):
    """Extract acoustic tokens from audio using S3Gen's tokenizer."""
    import torch

    device = cb_model.device

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        tokens = cb_model.s3gen.tokenizer(audio_tensor)

    if isinstance(tokens, tuple):
        tokens = tokens[0]
    return tokens.squeeze().cpu().long()


def tokenize_text(cb_model, text, max_text_tokens=256):
    """Tokenize text using Chatterbox's text tokenizer (cb_model.tokenizer)."""
    import torch

    tokens = cb_model.tokenizer.encode(text)
    if isinstance(tokens, (list, tuple)):
        tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.squeeze().cpu().long()

    return torch.cat([
        torch.tensor([255], dtype=torch.long),  # start token
        tokens[:max_text_tokens - 2],
        torch.tensor([0], dtype=torch.long),     # stop token
    ])


def preprocess_dataset(cb_model, dataset_name, device, hf_token=None, max_samples=None):
    """Load HF dataset and preprocess into tensors for T3 training.

    For each sample:
    1. Extract speaker embedding (256-dim) via VoiceEncoder from audio
    2. Extract acoustic tokens via S3Gen encoder from audio
    3. Tokenize text with start/stop tokens
    4. Cache results to volume for re-use
    """
    import torch
    import traceback
    import numpy as np
    from datasets import load_dataset
    from tqdm import tqdm
    import torchaudio

    # Check for cached data (v2 = with speaker embeddings via mel extraction)
    CACHE_VERSION = "v2"
    safe_name = dataset_name.replace("/", "_")
    cache_dir = Path(f"/data/cache/{safe_name}")
    if max_samples:
        cache_dir = cache_dir / f"n{max_samples}"
    cache_file = cache_dir / f"preprocessed_{CACHE_VERSION}.pt"

    if cache_file.exists():
        print(f"Loading cached preprocessed data from {cache_file}")
        data = torch.load(cache_file, weights_only=False)
        print(f"Loaded {len(data)} preprocessed samples from cache")
        return data

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train", token=hf_token)

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        print(f"Using {max_samples} samples")

    model_sr = getattr(cb_model, 'sr', CB_SAMPLE_RATE)
    processed = []

    print(f"Preprocessing {len(ds)} samples (extracting embeddings + tokens)...")
    for idx, example in enumerate(tqdm(ds)):
        try:
            text = example.get("text", "")
            audio_data = example.get("audio")

            if audio_data is None or not text:
                print(f"  Sample {idx}: missing audio or text, skipping")
                continue

            # Extract audio array and sample rate (supports dict or object)
            raw_array = audio_data.get("array") if isinstance(audio_data, dict) else getattr(audio_data, "array", None)
            if raw_array is None:
                print(f"  Sample {idx}: unexpected audio format {type(audio_data)}, skipping")
                continue
            audio_array = np.array(raw_array, dtype=np.float32)
            sample_rate = (audio_data.get("sampling_rate", CB_SAMPLE_RATE) if isinstance(audio_data, dict)
                           else getattr(audio_data, "sampling_rate", CB_SAMPLE_RATE))

            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)

            if sample_rate != model_sr:
                audio_tensor = torchaudio.transforms.Resample(sample_rate, model_sr)(audio_tensor)

            speaker_emb = extract_speaker_embedding(cb_model, audio_tensor, model_sr)
            speech_tokens = extract_speech_tokens(cb_model, audio_tensor, model_sr)
            text_tokens = tokenize_text(cb_model, text)

            processed.append({
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "text": text,
            })

        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            traceback.print_exc()
            continue

    print(f"Successfully preprocessed {len(processed)} / {len(ds)} samples")

    if not processed:
        raise RuntimeError(
            "No samples were successfully preprocessed! "
            "Check the model inspection output above for API issues."
        )

    # Cache to volume
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(processed, cache_file)
    print(f"Cached to {cache_file}")

    return processed


# ============================================================================
# Dataset & Data Collator
# ============================================================================

class ChatterboxDataset:
    """Dataset wrapper for preprocessed Chatterbox tensors."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ChatterboxCollator:
    """Collates preprocessed samples into dynamically padded batches.

    Pads text_tokens and speech_tokens to the max length in each batch.
    With 20% probability, speaker_emb is zeroed out (helps generalization).
    """

    def __init__(self, speaker_emb_dropout=SPEAKER_EMB_DROPOUT_PROB):
        self.speaker_emb_dropout = speaker_emb_dropout

    def __call__(self, batch):
        import torch
        import random

        text_tokens_list = [s["text_tokens"] for s in batch]
        speech_tokens_list = [s["speech_tokens"] for s in batch]
        speaker_embs = [s["speaker_emb"] for s in batch]

        B = len(batch)
        text_lens = [len(t) for t in text_tokens_list]
        speech_lens = [len(s) for s in speech_tokens_list]
        max_text = max(text_lens)
        max_speech = max(speech_lens)

        text_padded = torch.zeros(B, max_text, dtype=torch.long)
        for i, t in enumerate(text_tokens_list):
            text_padded[i, :len(t)] = t

        speech_padded = torch.zeros(B, max_speech, dtype=torch.long)
        for i, s in enumerate(speech_tokens_list):
            speech_padded[i, :len(s)] = s

        speaker_emb = torch.stack(speaker_embs)
        for i in range(B):
            if random.random() < self.speaker_emb_dropout:
                speaker_emb[i] = 0.0

        return {
            "text_tokens": text_padded,
            "text_token_lens": torch.tensor(text_lens, dtype=torch.long),
            "speech_tokens": speech_padded,
            "speech_token_lens": torch.tensor(speech_lens, dtype=torch.long),
            "speaker_emb": speaker_emb,
        }


# ============================================================================
# Find LoRA target modules
# ============================================================================

def find_lora_targets(t3_model):
    """Find attention projection layers in T3 for LoRA targeting.

    Supports Llama-style (q_proj, k_proj, v_proj, o_proj) and
    GPT-2-style (c_attn, c_proj) backbones.
    """
    standard_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
    gpt2_names = {"c_attn", "c_proj"}

    found_standard = set()
    found_gpt2 = set()

    for name, _ in t3_model.named_modules():
        last = name.split(".")[-1]
        if last in standard_names:
            found_standard.add(last)
        elif last in gpt2_names:
            found_gpt2.add(last)

    if found_standard:
        targets = list(found_standard)
    elif found_gpt2:
        targets = list(found_gpt2)
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print("WARNING: No attention layers found, using default targets. LoRA may not be applied.")

    print(f"LoRA target modules: {targets}")
    return targets


# ============================================================================
# Training
# ============================================================================

@app.function(
    image=image,
    gpu="H100",
    timeout=10800,  # 3 hours (preprocessing + training)
    volumes={"/data": volume},
    # WandB key passed via function arg from local .env (no secret required)
)
def train_chatterbox(
    dataset_name: str = "Trelis/ronan_tts_medium_clean",
    model_name: str = "ResembleAI/chatterbox-turbo",
    epochs: int = 50,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    hub_repo_id: str = None,
    hf_token: str = None,
    wandb_api_key: str = None,
    max_samples: int = None,
    language: str = "en",
):
    """Train Chatterbox TTS T3 model with LoRA on Modal."""
    import os
    import torch
    import torchaudio
    import math
    import soundfile as sf
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import login

    # Setup HF cache on volume
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_DIR}/hub"

    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace Hub")

    # WandB setup (passed from local .env)
    if wandb_api_key:
        import wandb
        wandb.login(key=wandb_api_key)
        os.environ["WANDB_PROJECT"] = "chatterbox-tts"
        print("WandB logging enabled (project: chatterbox-tts)")
    else:
        print("WandB logging disabled (no WANDB_API_KEY in secret)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path("/data/output/chatterbox-ft")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Chatterbox model (API varies across versions)
    print(f"\nLoading Chatterbox model: {model_name}")
    from chatterbox.tts import ChatterboxTTS

    for loader in [
        lambda: ChatterboxTTS.from_pretrained(device=device, model_id=model_name),
        lambda: ChatterboxTTS.from_pretrained(model_name, device=device),
        lambda: ChatterboxTTS.from_pretrained(device=device),
    ]:
        try:
            cb_model = loader()
            break
        except TypeError:
            continue
    else:
        raise RuntimeError(f"Could not load ChatterboxTTS model: {model_name}")

    inspect_model(cb_model)

    # ----------------------------------------------------------------
    # Preprocess dataset
    # ----------------------------------------------------------------
    print("\nPreprocessing dataset...")
    processed_data = preprocess_dataset(
        cb_model, dataset_name, device,
        hf_token=hf_token, max_samples=max_samples,
    )

    # Debug: print shape/type of first preprocessed sample
    print("\nPreprocessed sample 0 shapes:")
    for k, v in processed_data[0].items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} len={len(v) if isinstance(v, str) else v}")

    # Train/validation split
    total = len(processed_data)
    val_size = max(1, min(25, math.ceil(total * 0.1)))
    train_size = total - val_size

    print(f"Split: {train_size} train, {val_size} validation")
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]

    train_ds = ChatterboxDataset(train_data)
    val_ds = ChatterboxDataset(val_data)

    # ----------------------------------------------------------------
    # Apply LoRA to T3 model
    # ----------------------------------------------------------------
    t3 = cb_model.t3

    # Freeze T3 base weights (LoRA will add trainable adapters)
    for param in t3.parameters():
        param.requires_grad = False

    # Find LoRA target modules
    target_modules = find_lora_targets(t3)

    print(f"\nApplying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
    )

    t3_lora = get_peft_model(t3, lora_config)
    t3_lora.print_trainable_parameters()

    # ----------------------------------------------------------------
    # Custom Trainer for Chatterbox's non-standard forward pass
    # ----------------------------------------------------------------
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    class ChatterboxTrainer(Trainer):
        """Trainer that handles Chatterbox T3's custom forward pass.

        T3.loss() signature:
            t3_cond: T3Cond, text_tokens: LongTensor, text_token_lens: LongTensor,
            speech_tokens: LongTensor, speech_token_lens: LongTensor
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch.nn.functional as F
            IGNORE_ID = -100

            text_tokens = inputs["text_tokens"].to(device)
            text_token_lens = inputs["text_token_lens"].to(device)
            speech_tokens = inputs["speech_tokens"].to(device)
            speech_token_lens = inputs["speech_token_lens"].to(device)
            speaker_emb = inputs["speaker_emb"].to(device)

            B = speaker_emb.shape[0]
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                emotion_adv=torch.full((B, 1), 0.5, device=device),
            )

            t3_base = model.base_model.model if hasattr(model, 'base_model') else model
            t3_base.prepare_conditioning(t3_cond)

            out = t3_base.forward(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                text_token_lens=text_token_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_token_lens,
                training=True,
            )

            # Mask padding positions for loss computation
            mask_text = torch.arange(text_tokens.size(1), device=device)[None] >= text_token_lens[:, None]
            mask_speech = torch.arange(speech_tokens.size(1), device=device)[None] >= speech_token_lens[:, None]
            masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
            masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

            # Transpose logits from [B, seq, vocab] to [B, vocab, seq] for cross_entropy
            loss_text = F.cross_entropy(out.text_logits.transpose(1, 2), masked_text, ignore_index=IGNORE_ID)
            loss_speech = F.cross_entropy(out.speech_logits.transpose(1, 2), masked_speech, ignore_index=IGNORE_ID)
            loss = loss_text + loss_speech

            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """Override prediction_step to handle T3's custom forward pass."""
            model.eval()
            with torch.no_grad():
                loss = self.compute_loss(model, inputs, return_outputs=False)
            model.train()
            return (loss, None, None)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        bf16=True,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,  # Custom collator handles columns
        report_to="wandb" if wandb_api_key else "none",
        run_name=f"chatterbox-lora-r{lora_rank}-lr{learning_rate}-ep{epochs}",
    )

    trainer = ChatterboxTrainer(
        model=t3_lora,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=ChatterboxCollator(),
    )

    print("\nStarting LoRA training...")
    trainer.train()

    # ----------------------------------------------------------------
    # Merge LoRA weights and save
    # ----------------------------------------------------------------
    print("\nMerging LoRA weights...")
    merged_t3 = t3_lora.merge_and_unload()

    model_save_dir = output_dir / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged_t3.state_dict(), model_save_dir / "t3_finetuned.pt")
    print(f"Saved fine-tuned T3 to {model_save_dir}")

    # Replace T3 in the full model for inference
    cb_model.t3 = merged_t3

    volume.commit()

    # ----------------------------------------------------------------
    # Generate audio samples
    # ----------------------------------------------------------------
    print("\nGenerating audio samples...")
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    num_samples = min(5, len(val_data))
    model_sr = getattr(cb_model, 'sr', CB_SAMPLE_RATE)

    gen_kwargs = {"language_id": language} if language and language != "en" else {}

    print("\n--- Standard TTS samples ---")
    for i in range(num_samples):
        text = val_data[i]["text"]
        print(f"  Generating standard sample {i+1}: {text[:60]}...")
        try:
            wav = cb_model.generate(text, **gen_kwargs)
            if wav is not None:
                sample_path = samples_dir / f"standard_{i}.wav"
                torchaudio.save(str(sample_path), wav.cpu(), model_sr)
                print(f"  Saved: {sample_path}")
        except Exception as e:
            print(f"  Error generating standard sample {i}: {e}")

    print("\n--- Voice cloning samples ---")
    try:
        from datasets import load_dataset
        import numpy as np

        ds = load_dataset(dataset_name, split="train", token=hf_token)
        ref_idx = train_size  # First validation sample in original dataset
        ref_audio = ds[ref_idx]["audio"]

        if isinstance(ref_audio, dict) and "array" in ref_audio:
            ref_audio_path = str(output_dir / "reference_audio.wav")
            sf.write(
                ref_audio_path,
                np.array(ref_audio["array"], dtype=np.float32),
                ref_audio.get("sampling_rate", CB_SAMPLE_RATE),
            )
            print(f"  Reference audio saved to {ref_audio_path}")

            for i in range(num_samples):
                text = val_data[i]["text"]
                print(f"  Generating cloned sample {i+1}: {text[:60]}...")
                try:
                    wav = cb_model.generate(text, audio_prompt_path=ref_audio_path, **gen_kwargs)
                    if wav is not None:
                        sample_path = samples_dir / f"cloned_{i}.wav"
                        torchaudio.save(str(sample_path), wav.cpu(), model_sr)
                        print(f"  Saved: {sample_path}")
                except Exception as e:
                    print(f"  Error generating cloned sample {i}: {e}")
        else:
            print("  Could not extract reference audio from dataset")
    except Exception as e:
        print(f"  Could not generate cloned samples: {e}")

    volume.commit()

    # ----------------------------------------------------------------
    # Push to HuggingFace Hub
    # ----------------------------------------------------------------
    if hub_repo_id and hf_token:
        print(f"\nPushing model to HuggingFace Hub: {hub_repo_id}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(hub_repo_id, private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(model_save_dir / "t3_finetuned.pt"),
                path_in_repo="t3_finetuned.pt",
                repo_id=hub_repo_id,
            )
            print(f"Model pushed to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {model_save_dir}")
    print(f"Samples saved to: {samples_dir}")
    if hub_repo_id:
        print(f"Hub repo: https://huggingface.co/{hub_repo_id}")

    return {
        "output_dir": str(output_dir),
        "model_dir": str(model_save_dir),
        "samples_dir": str(samples_dir),
        "hub_repo": hub_repo_id,
    }


# ============================================================================
# Download
# ============================================================================

@app.function(image=image, volumes={"/data": volume})
def download_wav_files(subdir: str = "samples") -> list[tuple[str, bytes]]:
    """Download WAV files from the Modal volume."""
    samples_dir = Path(f"/data/output/chatterbox-ft/{subdir}")
    if not samples_dir.exists():
        return []
    files = [
        (f.name, f.read_bytes())
        for f in sorted(samples_dir.iterdir())
        if f.is_file() and f.suffix == ".wav"
    ]
    for name, _ in files:
        print(f"Prepared: {name}")
    return files


# ============================================================================
# Local Entrypoint
# ============================================================================

@app.local_entrypoint()
def main(
    dataset: str = "Trelis/ronan_tts_medium_clean",
    model_name: str = "ResembleAI/chatterbox-turbo",
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    hub_repo: str = "Trelis/chatterbox-ft",
    language: str = "en",
    max_samples: int = None,
    download: bool = True,
):
    """Run Chatterbox TTS training from local machine.

    Args:
        dataset: HuggingFace dataset with text + audio columns
        model_name: Base model (ResembleAI/chatterbox-turbo or ResembleAI/chatterbox)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for LoRA fine-tuning
        lora_rank: LoRA rank (default: 32)
        lora_alpha: LoRA alpha (default: 64)
        hub_repo: HuggingFace Hub repo to push model
        language: Language ID for multilingual model (e.g., fr, de, es)
        max_samples: Limit number of training samples (for testing)
        download: Download audio samples after training
    """
    from dotenv import load_dotenv
    import os

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if hf_token:
        print("HF token found (set)")
    else:
        print("Warning: No HF_TOKEN found in .env - private datasets/repos may not work")

    if wandb_api_key:
        print("WandB API key found")
    else:
        print("WandB logging disabled (no WANDB_API_KEY in .env)")

    print("\n" + "=" * 60)
    print("CHATTERBOX TTS MODAL TRAINING")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"WandB: {'enabled' if wandb_api_key else 'disabled'}")
    if hub_repo:
        print(f"Hub repo: {hub_repo}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("=" * 60 + "\n")

    print("Starting training on Modal...")
    result = train_chatterbox.remote(
        dataset_name=dataset,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        hub_repo_id=hub_repo,
        hf_token=hf_token,
        wandb_api_key=wandb_api_key,
        max_samples=max_samples,
        language=language,
    )
    print(f"\nTraining result: {result}")

    if download:
        print("\nDownloading audio samples...")
        samples_dir = Path("./output/chatterbox-ft/samples")
        samples_dir.mkdir(parents=True, exist_ok=True)

        files = download_wav_files.remote("samples")
        for filename, content in files:
            (samples_dir / filename).write_bytes(content)
            print(f"Downloaded: {filename}")

    print("\nDone!")
