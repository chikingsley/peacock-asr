# HuggingFace Speech-to-Speech Guide

This is a guide on using the [HuggingFace raw implementation](https://github.com/huggingface/speech-to-speech) for speech-to-speech.

Two methods are covered here:
1. Running on a Mac (will be slow with 8GB RAM or even 16 GB RAM)
2. Running on a GPU

## Running on a Mac

Run installations with:
```
pip install uv
git clone https://github.com/huggingface/speech-to-speech.git
cd speech-to-speech
git fetch --all
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install git+https://github.com/nltk/nltk.git@3.8.2
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
uv pip install -e .
uv run --with unidic -- python -m unidic download
cd ..
uv pip install -r requirements.txt
uv pip install lightning-whisper-mlx mlx-lm
uv run -- python s2s_pipeline.py --device mps --mode local --stt whisper-mlx --llm mlx-lm --tts melo --lm_model_name mlx-community/SmolLM-360M-Instruct-8bit
```
or with Gemma 2 B, a slightly larger TTS model:
```
uv run -- python s2s_pipeline.py --device mps --mode local --stt whisper-mlx --llm mlx-lm --tts melo --lm_model_name mlx-community/quantized-gemma-2b-it
```

Note that if you have a more powerful Mac (32+ GB RAM), you can try:
```
python s2s_pipeline.py --local_mac_optimal_settings
```
which will run using Lightning Whisper, Phi-3-Mini language model, and Melo TTS.

## GPU installation on a remote rental service
>[!TIP]
>If running on a remote GPU, pick one that is close to your location (e.g. if in Ireland, pick one in Ireland, the UK or mainland Europe).

There are two pieces here, server-setup and client-setup. You run the models on the server, but you need scripts on your client (local setup) to run the client.

### Server Setup (via TCP)

Two options:
- Run remotely on Runpod using the following Runpod template [here](https://runpod.io/console/deploy?template=xen5lu2cuf&ref=jmfkcdio).
- Run remotely on Vast.ai using this template [here](https://cloud.vast.ai/?ref_id=98762&creator_id=98762&name=HuggingFace%20Speech-to-Speech%20Server%20by%20Trelis).
Note that ports 12345 and 12346 are exposed for TCP

The full run command in the TCP template is:
```
python s2s_pipeline.py --recv_host 0.0.0.0 --send_host 0.0.0.0 --play_steps_s 1.5 --lm_model_name HuggingFaceTB/SmolLM-360M-Instruct
```

The Runpod and VastAI templates will run using Distil-Whisper-Large v3, Phi-3-Mini language model, and ParlerTTS.

To use a larger language model, adjust the run command in template. Melo is not supported at the time of writing for Cuda, so the server will run using ParlerTTS (a larger autoregressive TTS model).

--- Manual instructions ---

Run a ` pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` image and expose the 12345 and 12346 TCP ports with `-p 12345:12345` and `-p 12346:12346`.

Manually install and run the script:
```
apt-get update && apt-get install -y git build-essential gcc portaudio19-dev && rm -rf /var/lib/apt/lists/*
git clone https://github.com/TrelisResearch/speech-to-speech-cuda.git
cd speech-to-speech-cuda
pip install uv
uv venv
source .venv/bin/activate
uv pip install --no-cache-dir -r requirements.txt
uv pip install hf_transfer numpy scipy pyaudio psutil
uv pip install flash-attn --no-build-isolation
python s2s_pipeline.py --recv_host 0.0.0.0 --send_host 0.0.0.0 --play_steps_s 1.5
```
--- End of manual instructions ---

### Client Setup

Run this on your laptop.

Install with:
```
pip install uv
apt-get update -y
apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev -y
git clone https://github.com/Trelis/speech-to-speech-cuda.git
cd speech-to-speech-cuda
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
pip install sounddevice
```

Then, for Runpod / VastAI, run:
```
python listen_and_play.py --host 149.36.1.168 --send_port 42768 --recv_port 47941 --send_rate 16000 --recv_rate 16000
```
Note: The host IP and port numbers are based on the TCP Port Mappings:
- For Runpod, check them under connection settings. The ports should correspond to internal ports 12345 and 12346 for Runpod/TCP.
- For VastAI, find the wide blue button on the top of a Vast AI instance:

## BONUS - Local Approach (i.e. you have an Nvidia GPU)
> [!WARNING]
> This is a work in progress and has not been tested.

Download and run the docker container from [here](https://github.com/TrelisResearch/speech-to-speech-cuda/).

Expose the 12345 port with `-p 12345:12345` and the 12346 port with `-p 12346:12346`.

Once running, run the following command:
```
python s2s_pipeline.py --recv_host 0.0.0.0 --send_host 0.0.0.0 --play_steps_s 1.5
```