import soundfile as sf
import argparse
import os
import tempfile
import subprocess

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def truncate_video(input_path, max_seconds=30):
    """Truncate video to specified length in seconds."""
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "truncated_video.mp4")
    
    try:
        cmd = f"ffmpeg -i {input_path} -t {max_seconds} -c:v copy -c:a copy {output_path} -y"
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error truncating video: {e}")
        return input_path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process video with Qwen2.5-Omni model')
parser.add_argument('--input', type=str, default="input.mp4", 
                    help='Path to input video file (default: input.mp4)')
parser.add_argument('--truncate', action='store_true', 
                    help='Truncate video to 30 seconds')
parser.add_argument('--query', type=str, default="Describe what's happening in this video.",
                    help='Text query to ask about the video')
args = parser.parse_args()

# Process video if truncate option is specified
video_path = args.input
if args.truncate:
    print(f"Truncating video to 30 seconds: {video_path}")
    video_path = truncate_video(args.input)
    print(f"Truncated video saved to: {video_path}")

# Load the model on the available device(s)
model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", 
                                         torch_dtype="auto", 
                                         device_map="auto",
                                         attn_implementation="flash_attention_2")

# Print model architecture to a file
with open('architecture.txt', 'w') as f:
    print(model, file=f)
    
print("Model architecture saved to architecture.txt")

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# Create conversation with video input
conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": args.query},
        ],
    },
]

# Set use audio in video (set to True to include audio track from video)
USE_AUDIO_IN_VIDEO = True

# Prepare for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audios=audios, images=images, videos=videos, 
                  return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Generate both text and audio outputs
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# Decode text output
text_output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Model response:", text_output[0])

# Save audio output
sf.write(
    "video_output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)

print("Audio response saved to video_output.wav")

# Clean up temporary files
if args.truncate and os.path.dirname(video_path) != os.path.dirname(args.input):
    try:
        os.remove(video_path)
        os.rmdir(os.path.dirname(video_path))
        print("Cleaned up temporary files")
    except Exception as e:
        print(f"Error cleaning up: {e}")