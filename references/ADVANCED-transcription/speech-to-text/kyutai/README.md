# Kyutai

Speech to text models. Learn more on the Kyutai Repo [here](https://github.com/kyutai-labs/delayed-streams-modeling/).

In this folder:
- `llm_lingo_1.wav` is a Trelis audio file.
- `Trelis_kyutai_inference_stt_pytorch.ipynb` is a Colab notebook that shows how to run inference on the model.
- `Trelis_Kyutai_STT_Fine_Tuning.ipynb` is a jupyter notebook for fine-tuning. Run it in Colab or, preferably on a remote GPU (A40, A100, H100 recommended) allowing for jupyter to be run in a web browser. For example, perhaps run with this affiliate [one-click-template - do review the readme for the template](https://console.runpod.io/deploy?template=kfrosdmse5&ref=jmfkcdio). Notebook is a work in progress!
- `fine-tuning` folder contains the fine-tuning scripts.

What you can do in this folder:
- One-command TTS inference on Mac or cpu.
- Notebook inference, with pytorch or transformers.
- Word corrections (by passing in text and sound samples).
- One-click Rust server setup.
- Fine-tuning scripts [NEW]: see the `fine-tuning` folder.

## About Kyutai SST Models:
- The key benefit over whisper is that Kyutai models are streaming! This means that you get a faster response time, and higher accuracy (because the model is not conditioned on hearing the whole speech before being able to respond).

There are two models:
- `kyutai/stt-1b-en_fr`, an English and French model with ~1B parameters, a 0.5 second delay, and a semantic VAD.
- `kyutai/stt-2.6b-en`, an English-only model with ~2.6B parameters and a 2.5 second delay.

**Kyutai Model License**
Kyutai's license for these models is CC-BY 4.0. This is permissive but, when redistributing (original or modified) weights, you must credit the authors, indicate if you made changes, and link to the origian license and work.

## Inference

### Grabbing an Audio File

You can add your own audio file, or, to use a Trelis file, run:
```bash
cd speech-to-text/kyutai
uv run --with datasets --with soundfile python -c "
from datasets import load_dataset
import soundfile as sf
ds = load_dataset('Trelis/llm-lingo', split='validation')
audio = ds[3]['audio']
sf.write('llm_lingo_1.wav', audio['array'], audio['sampling_rate'])
print('Downloaded llm_lingo_1.wav')
"
```

### MLX Inference
On a Mac with an M1/M2/M3/M4 chip, you can run - without needing to git clone the repo:

```bash
## if you haven't done this already
# cd speech-to-text/kyutai 

uvx --with "moshi-mlx>=0.3.0" \
  python -m moshi_mlx.run_inference \
  --hf-repo kyutai/stt-2.6b-en-mlx \
  llm_lingo_1.wav --temp 0
```

### Running inference in a Notebook (Colab or local on Mac)

Set up a Virtual Environment for Local Notebooks:
```bash
# Create virtual environment with Python 3.10
uv venv kyutai-env --python 3.10

# Activate environment
source kyutai-env/bin/activate  # On Windows: kyutai-env\Scripts\activate

# Install only ipykernel (other dependencies are installed in the notebook)
uv pip install ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name kyutai-env --display-name "Kyutai STT"
```

Then select "Kyutai STT" as your kernel when running the notebook. The notebook will handle installing the required dependencies.

OR just upload to Colab and run the notebook there OR run on an A40, A100, H100 on Runpod, perhaps with this affiliate [one-click-template](https://console.runpod.io/deploy?template=ifyqsvjlzj).

### Running inference via cloning of the git repo

To run the following, first clone the repo:
```bash
git clone https://github.com/kyutai-labs/delayed-streams-modeling/
cd delayed-streams-modeling
# to create a virtual environment
uv venv
```

#### Run on a Mac with a Microphone

```bash
uv run scripts/stt_from_mic_mlx.py
```

#### Word Level Timestamps
Run on MAC (or with cuda or with cpu):
```bash
uv run --with hf_transfer \
  scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr \
  ../llm_lingo_1.wav \
  --device cpu
```
Moshi provides word level timestamps.

#### Text and/or voice assisted transcription
The idea here is that you can pass in unfamiliar words to assist with transcription. You can just pass in the text via `--prompt_text` or you can pass in a file via `--prompt_file` that has a recording of that tricky word (or words).

**Option 1: Text-only prompting (simpler)**
Unfortunately, this not very robust...
```bash
uv run --with julius --with moshi scripts/stt_from_file_with_prompt_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr \
  --file ../llm_lingo_1.wav \
  --device mps \
  --prompt_text "Tricksy, Phi-2"
```
You can try with the bigger model, but only passing words won't work all that well:
```bash
uv run --with julius --with moshi scripts/stt_from_file_with_prompt_pytorch.py \
  --hf-repo kyutai/stt-2.6b-en \
  --file ../llm_lingo_1.wav \
  --device mps \
  --prompt_text "Phi-2"
```

**Option 2: Audio + text prompting (if you have a WAV file of the word)**
It works a lot better if you record yourself saying those words.
```bash
# First convert M4A to WAV if needed:
# ffmpeg -i Tricksy.m4a Tricksy.wav

uv run --with julius --with moshi scripts/stt_from_file_with_prompt_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr \
  --file ../llm_lingo_1.wav \
  --device mps \
  --prompt_text "Tricksy" \
  --prompt_file ../Tricksy.wav \
  --cut-prompt-transcript
```

If you need to convert audio files to compatible formats:

**M4A to WAV (recommended for audio prompts):**
```bash
ffmpeg -i ../Tricksy.m4a ../Tricksy.wav
```

**Install ffmpeg if needed:**
```bash
# On macOS with Homebrew
brew install ffmpeg

# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

#### Running a Server

You can start with this [CUDA / Pytorch with TCP](https://console.runpod.io/deploy?template=v52vmwy837&ref=jmfkcdio)) and skip to step 7.

<details>
<summary>Manual Rust Server Setup</summary>
Once the pod is loaded, ssh in.

The Rust implementation provides a server that can process multiple streaming queries in parallel. Depending on the amount of memory on your GPU, you may have to adjust the batch size from the config file. For a L40S GPU, a batch size of 64 works well and requests can be processed at 3x real-time speed.

##### 1. Install System Dependencies & 2. Install Rust and Cargo & 3. Install the Server
```bash
# Install required system packages
apt update
apt install -y pkg-config libssl-dev build-essential cmake

# Verify required tools are available
pkg-config --version
cmake --version
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# then proceed with standard installation

# Reload your shell or run:
source ~/.cargo/env

# Verify installation
cargo --version

# Install the server
cargo install --features cuda moshi-server
```

##### 4. Clone the Repository and Get Config Files
```bash
# Clone the repository to get config files
git clone https://github.com/kyutai-labs/delayed-streams-modeling/
cd delayed-streams-modeling
```

##### 5. Start the Server
```bash
# Make sure you're in the delayed-streams-modeling directory
# For English + French model (1B parameters)
moshi-server worker --config configs/config-stt-en_fr-hf.toml

# # For English-only model (2.6B parameters)
# moshi-server worker --config configs/config-stt-en-hf.toml
```
>[!TIP]
>If you have a fine-tuned model, you'll need to update the config file on the server

##### 6. RunPod Configuration
- **Port**: Make sure to expose TCPport **8080** in your RunPod interface (this will already be done if you used the one click template above)
- **GPU Memory**: Adjust `batch_size` in the config file based on your GPU (probably not needed).
- **Authentication**: Default API key is `public_token`
</details>

##### 7. Get Your RunPod TCP Endpoint

After starting your server on RunPod, you need to get the public TCP endpoint:

Go to Connect and then copy the TCP port that should look a bit like this "63.141.33.65:22177".

##### 8. Client Usage (from your laptop)

**IMPORTANT**: Replace `63.141.33.65:22041` with your actual RunPod TCP endpoint from step 6.

**From microphone (real-time):**
```bash
# List available microphones
cd delayed-streams-modeling
uv run scripts/stt_from_mic_rust_server.py --list-devices

# Use specific microphone device
uv run scripts/stt_from_mic_rust_server.py \
  --url ws://69.30.85.81:22144 \
  --device 4

# Show voice activity detection
uv run scripts/stt_from_mic_rust_server.py \
  --url ws://63.141.33.65:22041 \
  --show-vad \
  --device 4
```

**From audio file:**
```bash
# Connect to RunPod server
uv run scripts/stt_from_file_rust_server.py ../llm_lingo_1.wav \
  --url ws://69.30.85.81:22026

# Process as fast as possible (no real-time simulation)
uv run scripts/stt_from_file_rust_server.py ../llm_lingo_1.wav \
  --url ws://69.30.85.81:22026 \
  --rtf 1000

# With custom API key
uv run scripts/stt_from_file_rust_server.py audio/bria.mp3 \
  --url ws://69.30.85.81:22026 \
  --api-key your_api_key
```

**For localhost testing (if running server locally):**
```bash
# Microphone
uv run scripts/stt_from_mic_rust_server.py

# File
uv run scripts/stt_from_file_rust_server.py audio/bria.mp3
```

**Connection Details:**
- **Default URL**: `ws://127.0.0.1:8080` (for local development)
- **RunPod URL**: `ws://your-ip:port` (non-secure WebSocket for direct TCP)
- **API Endpoint**: `/api/asr-streaming` (automatically appended)
- **Authentication**: Via `kyutai-api-key` header or `?auth_id={api_key}` URL parameter

The server supports real-time streaming transcription with word-level timestamps and can handle multiple concurrent connections.

## Converting Fine-tuned Models for Rust Server

After fine-tuning a Kyutai STT model using the HuggingFace Transformers library (see `fine-tuning/` folder), you need to convert the weights to Candle format before using them with the Rust server.

**Why conversion is needed**: The Transformers and Candle formats use different weight structures:
- Transformers combines all embeddings into one tensor; Candle splits them
- Transformers has separate Q/K/V projections; Candle uses combined `in_proj_weight`
- Layer naming conventions differ

**Scripts in `fine-tuning/`**:
- `convert_transformers_to_candle.py` - Converts fine-tuned models to Candle format
- `tests/test_conversion.py` - Validates the conversion is correct

```bash
# Convert a fine-tuned model
cd fine-tuning
uv run python convert_transformers_to_candle.py \
    --input ./kyutai-finetuned \
    --output ./kyutai-finetuned-candle/model.safetensors

# Validate the conversion
uv run python tests/test_conversion.py \
    --original kyutai/stt-1b-en_fr-candle \
    --converted ./kyutai-finetuned-candle/model.safetensors
```

Then update your Rust server config to point to the converted weights. See `fine-tuning/KYUTAI_FINETUNING_EXPLAINED.md` Section 9 for details.

---

## Changelog

**Jan 5, 2025**: Added `convert_transformers_to_candle.py` and `test_conversion.py` scripts for converting fine-tuned models to Candle/Rust server format.