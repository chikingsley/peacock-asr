from datasets import Dataset, DatasetDict, Audio
import webvtt
from datetime import datetime
import librosa
import soundfile as sf
import os
from huggingface_hub import login

# Setup
hf_username = "Trelis"  # Your Hugging Face username
repo_name = "llm-lingo"  # Name of the repository on the Hub
train_audio_file = "data/train.mp3"  # Path to the training audio file
train_vtt_file = "data/train_corrected.vtt"  # Path to the training VTT file
validation_audio_file = "data/validation.mp3"  # Path to the validation audio file
validation_vtt_file = "data/validation_corrected.vtt"  # Path to the validation VTT file
save_path = f"data/{repo_name}-dataset"  # Local save path

def parse_time(time_str):
    return datetime.strptime(time_str, '%H:%M:%S.%f')

def milliseconds(time_obj):
    return (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + time_obj.microsecond // 1000

def time_to_samples(time_ms, sr):
    return int((time_ms / 1000.0) * sr)

def transform_data(data):
    transformed = {"audio": [], "text": [], "start_time": [], "end_time": []}
    for item in data:
        for key in transformed:
            transformed[key].append(item[key])
    return transformed

def process_audio_file(audio_path, vtt_path, output_dir, max_duration=30):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the full audio file with librosa
    full_audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # Parse VTT file
    captions = webvtt.read(vtt_path)

    # Prepare data for 🤗 Datasets
    data = []
    current_text = []
    current_start = None
    current_end = None
    accumulated_duration = 0
    segment_counter = 0

    for caption in captions:
        start_time = parse_time(caption.start)
        end_time = parse_time(caption.end)
        duration = (end_time - start_time).total_seconds()

        if current_start is None:
            current_start = start_time

        if accumulated_duration + duration <= max_duration:
            current_text.append(caption.text)
            current_end = end_time
            accumulated_duration += duration
        else:
            # Process and save the audio segment in MP3 format
            segment_filename = f"{output_dir}/segment_{segment_counter}.mp3"
            start_sample = time_to_samples(milliseconds(current_start.time()), sr)
            end_sample = time_to_samples(milliseconds(current_end.time()), sr)
            audio_segment = full_audio[start_sample:end_sample]
            sf.write(segment_filename, audio_segment, sr, format='mp3')

            # Add the segment info to the dataset
            data.append({
                "audio": segment_filename,
                "text": ' '.join(current_text),
                "start_time": current_start.strftime('%H:%M:%S.%f')[:-3],
                "end_time": current_end.strftime('%H:%M:%S.%f')[:-3]
            })

            # Prepare for the next segment
            current_text = [caption.text]
            current_start = start_time
            current_end = end_time
            accumulated_duration = duration
            segment_counter += 1

    # Process and save any remaining audio segment
    if current_text:
        segment_filename = f"{output_dir}/segment_{segment_counter}.mp3"
        start_sample = time_to_samples(milliseconds(current_start.time()), sr)
        end_sample = time_to_samples(milliseconds(current_end.time()), sr)
        audio_segment = full_audio[start_sample:end_sample]
        sf.write(segment_filename, audio_segment, sr, format='mp3')

        data.append({
            "audio": segment_filename,
            "text": ' '.join(current_text),
            "start_time": current_start.strftime('%H:%M:%S.%f')[:-3],
            "end_time": current_end.strftime('%H:%M:%S.%f')[:-3]
        })

    return data

def create_dataset(train_audio_file, train_vtt_file, validation_audio_file, validation_vtt_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Process training audio file
    train_data = process_audio_file(train_audio_file, train_vtt_file, f"{output_dir}/train")

    # Process validation audio file
    validation_data = process_audio_file(validation_audio_file, validation_vtt_file, f"{output_dir}/validation")

    # Transform data into the correct format for Dataset.from_dict
    train_dataset = Dataset.from_dict(transform_data(train_data))
    valid_dataset = Dataset.from_dict(transform_data(validation_data))

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset
    })

    return dataset_dict

# Create dataset
dataset = create_dataset(train_audio_file, train_vtt_file, validation_audio_file, validation_vtt_file, save_path)

# Save dataset locally
dataset.save_to_disk(save_path)

# Cast the audio column to the Audio feature
dataset = dataset.cast_column("audio", Audio())

login()

# Push the dataset to the Hub
dataset.push_to_hub(repo_id=f"{hf_username}/{repo_name}")

print(f"Dataset pushed to {hf_username}/{repo_name}")