"""All-layer SSL feature extraction for Phase 2 (CHConv).

Extracts hidden states from all transformer layers of frozen SSL models,
averages over phone durations to get phone-level representations, and saves
as numpy arrays for HConv/CHConv training.

Input:  SpeechOcean762 audio WAVs + phone duration features (from Phase 1 data)
Output: [N_utt, 50, L, D] per model per split (float16 to save disk)

Uses the same phone-level averaging as the original ConPCO feature extraction
pipeline — "SSL embeddings are obtained by chunking based on phone durations
and computing the mean of the resulting segments."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# SSL models used in MuFFIN (3 × Large, 1024-dim, 25 layers including CNN output)
SSL_MODELS = {
    "w2v_300m": "facebook/wav2vec2-large",
    "hubert": "facebook/hubert-large-ll60k",
    "wavlm": "microsoft/wavlm-large",
}

# Frame rate for all three models: 50 Hz (20ms per frame)
FRAME_RATE_HZ = 50


def _load_audio(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio as 16kHz mono numpy array."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        # Simple resample via librosa (already a dependency)
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def _phone_average(
    hidden_states: torch.Tensor,
    durations: np.ndarray,
    max_phones: int = 50,
) -> np.ndarray:
    """Average frame-level features over phone durations.

    Args:
        hidden_states: [L, T_frames, D] — all-layer features for one utterance.
        durations:     [N_phones] — phone durations in seconds.
        max_phones:    Pad/truncate to this many phones.

    Returns:
        [max_phones, L, D] — phone-averaged all-layer features (float16).
    """
    L, T_frames, D = hidden_states.shape
    n_phones = len(durations)

    # Convert durations (seconds) to frame counts
    frame_counts = np.round(durations * FRAME_RATE_HZ).astype(int)
    # Adjust last phone to consume remaining frames
    total_assigned = frame_counts.sum()
    if total_assigned < T_frames and n_phones > 0:
        frame_counts[-1] += T_frames - total_assigned
    elif total_assigned > T_frames and n_phones > 0:
        frame_counts[-1] = max(0, frame_counts[-1] - (total_assigned - T_frames))

    # Average per phone per layer
    result = np.zeros((max_phones, L, D), dtype=np.float16)
    start = 0
    for i in range(min(n_phones, max_phones)):
        end = min(start + frame_counts[i], T_frames)
        if end > start:
            chunk = hidden_states[:, start:end, :].float().mean(dim=1)  # [L, D]
            result[i] = chunk.half().cpu().numpy()
        start = end

    return result


def extract_all_layers(
    model_key: str,
    audio_dir: Path,
    utterance_ids: list[str],
    speaker_map: dict[str, str],
    durations: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """Extract all-layer features for one SSL model.

    Args:
        model_key:     Key in SSL_MODELS (e.g., "hubert").
        audio_dir:     Path to WAVE/ directory with speaker subdirs.
        utterance_ids: Sorted list of utterance IDs (matching npy row order).
        speaker_map:   utt_id → speaker_dir mapping.
        durations:     [N_utt, 50, 1] phone durations from dur_feat.npy.
        device:        "cuda" or "cpu".

    Returns:
        [N_utt, 50, L, D] all-layer phone-averaged features (float16).
    """
    model_name = SSL_MODELS[model_key]
    print(f"Loading {model_name}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    N = len(utterance_ids)
    all_features: np.ndarray | None = None  # lazy init after we know L

    with torch.no_grad():
        for idx, utt_id in enumerate(tqdm(utterance_ids, desc=model_key)):
            # Find audio file
            spk_dir = speaker_map[utt_id]
            wav_path = audio_dir / spk_dir / f"{utt_id}.WAV"
            if not wav_path.exists():
                wav_path = audio_dir / spk_dir / f"{utt_id}.wav"

            audio = _load_audio(wav_path)

            # Run model with all hidden states
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)

            # Stack all hidden states: list of [1, T, D] → [L, T, D]
            hs = torch.stack(outputs.hidden_states, dim=0).squeeze(1)  # [L, T, D]

            # Lazy init output array
            if all_features is None:
                L, _, D = hs.shape
                print(f"  {model_key}: L={L}, D={D}")
                all_features = np.zeros((N, 50, L, D), dtype=np.float16)

            # Phone-average using durations
            dur_vec = durations[idx, :, 0]
            valid_mask = dur_vec > 0
            n_phones = int(valid_mask.sum())
            phone_durs = dur_vec[:n_phones]

            all_features[idx] = _phone_average(hs, phone_durs)

    assert all_features is not None, "No utterances processed"
    return all_features


def extract_split(
    split: str,
    features_dir: Path,
    audio_dir: Path,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> None:
    """Extract all-layer features for one split (train or test).

    Args:
        split:        "train" or "test".
        features_dir: Path to existing feature dir (has dur_feat.npy, etc.).
        audio_dir:    Path to SpeechOcean762 WAVE/ directory.
        output_dir:   Where to save. Defaults to features_dir.
        device:       "cuda" or "cpu".
    """
    output_dir = output_dir or features_dir
    prefix = "tr" if split == "train" else "te"
    data_subdir = features_dir / "seq_data_librispeech_v4"

    # Load durations
    dur_path = data_subdir / f"{prefix}_dur_feat.npy"
    durations = np.load(dur_path)
    print(f"Loaded durations: {dur_path} shape={durations.shape}")

    # Get sorted utterance IDs from SpeechOcean762
    so762_root = audio_dir.parent  # WAVE/ is inside the SpeechOcean762 root
    utt2spk_path = so762_root / split / "utt2spk"
    utterance_ids = []
    speaker_map = {}
    with open(utt2spk_path) as f:
        for line in f:
            utt_id, spk_id = line.strip().split()
            utterance_ids.append(utt_id)
            # Find speaker directory name
            speaker_map[utt_id] = f"SPEAKER{spk_id.zfill(4)}"
    utterance_ids = sorted(utterance_ids)
    assert len(utterance_ids) == durations.shape[0], (
        f"Utterance count mismatch: {len(utterance_ids)} vs {durations.shape[0]}"
    )

    # Extract for each SSL model
    for model_key in SSL_MODELS:
        out_path = data_subdir / f"{prefix}_{model_key}_all_layers.npy"
        if out_path.exists():
            print(f"Skipping {out_path} (already exists)")
            continue

        features = extract_all_layers(
            model_key, audio_dir, utterance_ids, speaker_map, durations, device
        )
        np.save(out_path, features)
        print(f"Saved {out_path} shape={features.shape} ({features.nbytes / 1e9:.1f} GB)")
