from generator import Segment, load_csm_1b
import torchaudio
import torch
import os
# Initialize the generator first
generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")

speakers = [0]
transcripts = [ 
    "Greetings! My name is Ronan and it's nice to meet you"
]
audio_paths = [
    "data/Greetings! My name is Ronan and it's nice to meet you.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]

text_to_generate = "How are you doing today?"

audio = generator.generate(
    text=text_to_generate,
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

# Create output directory if it doesn't exist
output_dir = "data/output"
os.makedirs(output_dir, exist_ok=True)

torchaudio.save(f"{output_dir}/baseline.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

audio = generator.generate(
    text=text_to_generate,
    speaker=0,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("data/output/cloned.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
