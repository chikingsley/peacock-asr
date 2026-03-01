# Qwen2.5-Omni Audio Demo

In this folder:
- Inferencing with audio input and audio output
- Inferencing with video input and audio output
- Link to scripts for fine-tuning (not tested by Trelis)

## Setup

1. Install required packages:

>![TIP]
>If running locally, create a venv with `uv venv` and consider swapping the device to `mps` instead of `cuda` if using mac.

Run:
```bash
cd speech-to-speech/qwen2.5-omni
pip install uv
uv pip install soundfile qwen-omni-utils[decord] hf_transfer --system
```

Then:
```bash
uv pip uninstall transformers torch torchvision torchaudio --system
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --system
uv pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8 --system
uv pip install accelerate --system
uv pip install flash-attn --no-build-isolation --system
```

2. Install ffmpeg:

On Ubuntu/Debian:
```bash
apt update
apt install ffmpeg -y
```

On other systems, see [ffmpeg installation guide](https://ffmpeg.org/download.html).

3. Set environment variable for faster downloads:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="/workspace"
```

## Audio Input Inferencing

1. Place your input audio file (e.g., `input.wav`) in the same directory as the script.

2. Run the script:
```bash
uv run run_audio.py --system
```

The script will:
- Process your input audio
- Generate a text response
- Generate and save an audio response to `output.wav`

Optionally, you can swap the speaker from female to male by commenting in the following line in run_audio.py:
```bash
text_ids, audio = model.generate(**inputs, return_audio=True, spk="Ethan") # male voice
```

**## **Notes**
- The script uses the default Chelsie voice for audio output
- Minimum GPU memory requirements:
  - FP32: ~94GB
  - BF16: ~31GB (recommended)
- Flash Attention 2 is enabled by default for better performance
- Make sure your hardware is compatible with FlashAttention 2 (see [flash attention repository](https://github.com/Dao-AILab/flash-attention) for details) 

## Video Input Inferencing

1. Place your video file (e.g., `input.mp4`) in the same directory as the script.

2. Use the following script to process a video and generate both text and audio responses:

Run:
```bash
uv run run_video.py
```

You can also use these command line options:
```bash
# Process a different video file
uv run run_video.py --input path/to/your/video.mp4

# Truncate the video to 30 seconds to reduce memory usage
uv run run_video.py --truncate

# Specify a custom query about the video
uv run run_video.py --query "What's happening in the video? Who and what do you see and what is it about?" --truncate

# Combine multiple options
uv run run_video.py --input my_long_video.mp4 --truncate --query "Describe the people in this video"
```

3. You can modify the script for different use cases:
   - Change the prompt text to ask specific questions about the video
   - Set `USE_AUDIO_IN_VIDEO = False` to ignore the audio track in the video
   - Switch to male voice by adding `spk="Ethan"` to the generate call
   - Get text-only output by setting `return_audio=False` in the generate call

> **Note:** Processing videos requires more memory than audio. For a 15-second video, expect to use:
> - FP32: ~94GB 
> - BF16 with Flash Attention 2: ~31GB 

## Fine-tuning

>![IMPORTANT]
>Fine-tuning scripts have not been reviewed by Trelis.

- [Llama Factory](https://github.com/hiyouga/LLaMA-Factory/pull/7537) - allows tuning on one of audio/video/image inputs, along with text outputs. No support for tuning on audio outputs.
- [Align Anything](https://github.com/PKU-Alignment/align-anything/pull/169) - only supports text to text fine-tuning as of April 3rd 2025.

## SGLANG and vLLM notes
No support as of April 3rd 2025.