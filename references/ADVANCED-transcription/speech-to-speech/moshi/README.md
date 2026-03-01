# Moshi
Todo:
[ ] Understand how the parallel channels are fed in together to the model.
[ ] Better understand how the loss is calculated, by further inspecting the fine-tuning code.
[ ] Come up with some data generation methods for fine-tuning (to make it sound like me). Probably requires some kind of TTS recording pipeline. Put this into thinking...

## Running on Mac

Install
```bash
uv venv
uv pip install -U moshi_mlx
```
Then, to run in your terminal use:
```bash
uv run python -m moshi_mlx.local -q 8
```
Or, to run in a web client, run:
```bash
uv run python -m moshi_mlx.local_web -q 8
```

## Run on a rented GPU (e.g. Runpod)

Start up a rented GPU (Ampere or Hopper, recommended) on a service like Runpod ([one-click-template here, affiliate link](https://runpod.io/gsc?template=ifyqsvjlzj&ref=jmfkcdio)).

Connect to the instance via ssh in your Cursor/VSCode/Windsurf IDE.

Then install
```bash
pip install uv
uv pip install -U moshi hf_transfer --system
export HF_HUB_ENABLE_HF_TRANSFER=1 # for fast weight downloads
export HF_HOME="/workspace" # to direct the model weights to the volume instead of the container
```

To run in a web client, run:
```bash
uv run python -m moshi.server --hf-repo kyutai/moshika-pytorch-bf16 # for better quality ~15 GB weights (btw, quality is not noticably better).
uv run python -m moshi.server --hf-repo kyutai/moshika-pytorch-q8 # for smaller weights ~9 GB
```

Or to run a barebones client (no echo cancellation or lag handling):
```bash
uv run python -m moshi.client
```

## Fine-tune Moshi

If you have stereo audio (two channels, one for each speaker) then you can fine-tune the assistant's voice to sound like the assistant (SPEAKER_MAIN) in your dataset. In principle, this fine-tuning can improve performance on edge cases like noise or silence or tough accents, or having Moshi respond a bit better with the right cadence.