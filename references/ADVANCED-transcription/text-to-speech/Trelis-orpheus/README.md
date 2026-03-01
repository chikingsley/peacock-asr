# Trelis Orpheus

>![IMPORTANT]
>These scripts are built upon those available in the [Orpheus repo](https://github.com/canopyai/Orpheus-TTS)

You can:
- Inference
- Voice clone
- Fine-tune (incl. dataset generation)
- Use a fine-tuned model *with* voice cloning!

You can run this model on CPU or on a remote service such as Runpod (one-click template [here, affiliate link](https://www.runpod.io/console/deploy?template=ifyqsvjlzj&ref=jmfkcdio)).

## Running Inference

### Run locally
For running locally (will work on Mac but will be slow):
```bash
cd text-to-speech/Trelis-orpheus
python -m venv oEnv
source oEnv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=oEnv --display-name "Python (oEnv)"
pip install jupyterlab
```
Then run the `Trelis_Orpheus_Inference.ipynb` notebook with `jupyter lab` or in your IDE (Cursor/Windsurf).

### Run on GPU

Open the `Trelis_Orpheus_Inference.ipynb` notebook in collab OR on a remote server like Runpod (one-click template [here (affiliate link)](https://runpod.io/gsc?template=ifyqsvjlzj&ref=jmfkcdio)).

This will additionally allow you to run with vllm, although this is not running correctly as of March 20th 2025.

## Voice Cloning
Open the `Trelis_Orpheus_clone.ipynb` notebook. Note that using >30 seconds, perhaps even a few minutes of voice, is best for cloning.

My quick testing suggests that CSM-1B clones better than Orpheus.

## Fine-tuning

### Dataset Preparation

1. Install dependencies:
```bash
cd text-to-speech/Trelis-orpheus/fine-tune
pip install uv
uv venv
uv pip install -r requirements.txt
```
if running on gpu, also run:
```bash
uv pip install flash_attn --no-build-isolation
```

2. Record your voice dataset:
```bash
uv run record_voice_dataset.py --speaker Ronan --dataset_name Trelis/orpheus-ft --test
```

The script will:
- Guide you through recording phrases from various categories
- Play back each recording for verification
- Allow you to re-record if you're not satisfied
- Track progress and allow pausing between recordings
- Push the raw dataset to Hugging Face Hub when complete

**Dataset Format**

The dataset format used for Orpheus fine-tuning has a specific structure:
- Each item contains a `text` field with the transcript, prefixed with the speaker name (e.g., "Ronan: This is a test")
- Each item contains a `source` field with the speaker name
- Each item contains an `audio` field with:
  - `path`: Path to the audio file
  - `array`: The actual audio data
  - `sampling_rate`: Sample rate of the audio (24000Hz)

This format is critical for the tokenization script to correctly process the data for multi-speaker fine-tuning.

**Push Existing Recordings**

If you already have voice recordings or if the push to Hub failed, you can use the `push_raw_dataset.py` script:
```bash
uv run push_raw_dataset.py --dir voice_dataset/Ronan --speaker Ronan --dataset_name Trelis/orpheus-ft
```

This is useful if:
- You already have recordings but need to push them to Hub
- The original push to Hub failed
- You want to reuse recordings with better metadata

3. Tokenize the dataset:

Use the Python script to tokenize your dataset rather than the notebook (which can have locale issues):

```bash
# Install required packages
uv pip install torchaudio datasets snac huggingface_hub -qU

# Login to Hugging Face first
huggingface-cli login
```

If you want to create a private dataset, add the `--private` flag:
```bash
uv run tokenise_speech_dataset.py --input_dataset Trelis/orpheus-ft --output_dataset Trelis/orpheus-formatted --device cuda --private # or cuda for NVIDIA GPUs
```

The script:
- Creates a repository on Hugging Face Hub for the output dataset
- Downloads your raw dataset from HuggingFace
- Loads the SNAC model for audio tokenization
- Processes each audio file to extract codes
- Creates a formatted dataset with proper metadata suitable for fine-tuning
- Pushes the formatted dataset back to HuggingFace

**Verifying your dataset**

If you need to check if your dataset was properly pushed to Hugging Face or examine its structure:
```bash
uv run check_dataset.py --dataset Trelis/orpheus-formatted
```

This will:
- Verify the dataset exists on the Hub
- Show when it was last modified
- Display the number of examples
- Show the structure of the first example
- List all available features/fields

### Train the Model
- Modify `fine-tune/config.yaml` with your dataset and training parameters
  - `TTS_dataset`: Your HuggingFace dataset name
  - `model_name`: Base model to fine-tune
  - `epochs`: Number of training epochs
  - `batch_size`: Batch size for training
  - `learning_rate`: Learning rate for training
  - `scheduler_type`: Learning rate scheduler ("constant", "linear", "cosine", "cosine_with_restarts")
  - `save_steps`: How often to save checkpoints
  - `run_name`: Name for the training run

- Run training:
```bash
uv run huggingface-cli login

uv pip install transformers datasets tensorboard trl torch hf_transfer accelerate
uv pip install flash_attn --no-build-isolation # if on gpu
```
Then configure the training with:
```bash
uv run accelerate config
```
Then run:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run accelerate launch train.py # for full fine-tuning
```

or for LoRA:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv pip install peft
uv run accelerate launch train-lora.py
```

Note: Test with just 4 samples. Then try 50 rows. If needed, go as far as 250 rows.

**LoRA Training Notes**

As an alternative to full fine-tuning, you can use LoRA (Low-Rank Adaptation) which is much more efficient in terms of memory usage and training time. We've provided a script that:

- Uses a rank of 32 and alpha of 64
- Targets the projection layers in the model
- Uses the learning rate scheduler specified in config.yaml (defaults to "cosine")
- Merges the LoRA weights with the base model at the end

### Using Tensorboard

The training script now automatically sets up Tensorboard logging with an evaluation dataset. The evaluation dataset is created by taking 10% of your training data (minimum 1 example, maximum 25 examples).

To view training and evaluation metrics in real-time:

```bash
# Launch tensorboard with the logs directory
# Do this in a new terminal
cd text-to-speech/Trelis-orpheus/fine-tune/

# Install tensorboard if not already installed
uv pip install tensorboard
uv run tensorboard --logdir=./checkpoints/logs
```

This will start a Tensorboard server that you can access by opening a web browser and navigating to `http://localhost:6006`. 

Tensorboard provides visualizations of:
- Training and evaluation loss
- Learning rate progression
- Model gradients
- Other training metrics

You can track your model's performance over time and compare different training runs by giving them unique `run_name` values in the config.yaml file.

### Testing Your Fine-Tuned Model

After fine-tuning, you can test your model using the provided inference script. This script works with both local fine-tuned models and models hosted on Hugging Face.

#### Run Inference

Basic usage:
```bash
# Test the locally saved model
uv run python inference.py --model_path ./output/orpheus-tts-0.1-pretrained-ft --device cuda --prompt "This is a test of my fine-tuned voice model." --output "fft.wav" --speaker Ronan
uv run python inference.py --model_path ./output/orpheus-tts-0.1-pretrained-lora-ft --device cuda --prompt "This is a test of my fine-tuned voice model." --output "lora.wav" --speaker Ronan
```

>![TIP]
>For fine-tuned models, it's important to use `--speaker Ronan` to append a prefix of "Ronan: " to all prompts.

The script will:
1. Load your fine-tuned model
2. Generate speech based on your prompt
3. Save the audio to a WAV file (default: output.wav)
4. Print the path to the generated audio file

This is a simpler alternative to the notebook for quickly testing your model's performance after fine-tuning. It works well on Mac with MPS (Apple Silicon), as well as NVIDIA GPUs and CPUs.

### Voice Cloning Inference

For voice cloning using a reference audio sample, you can use the `clone_inference.py` script, which is a command-line version of the cloning notebook:

```bash
uv pip install librosa torchaudio

# Using a local fully fine-tuned fine-tuned model
uv run python clone_inference.py \
  --model_path "./output/orpheus-tts-0.1-pretrained-ft" \
  --device cuda \
  --reference_audio "X.wav" \
  --reference_text "This is how you count to three: one, two, three" \
  --prompt "four, five, six, seven, eight, nine ten." \
  --speaker "Ronan" \
  --output "cloned_fft_model.wav"

  # Using a local fine-tuned model (lora)
uv run python clone_inference.py \
  --model_path "./output/orpheus-tts-0.1-pretrained-lora-ft" \
  --device cuda \
  --reference_audio "X.wav" \
  --reference_text "This is how you count to three: one, two, three" \
  --prompt "four, five, six, seven, eight, nine ten." \
  --speaker "Ronan" \
  --output "cloned_lora_model.wav"

# Disable automatic audio playback
uv run python clone_inference.py --no_play
```

This script allows you to:
1. Use a reference audio file (default: X.wav in the parent directory)
2. Provide the transcript of what's spoken in the reference audio
3. Generate new speech in the voice from the reference audio
4. Customize generation parameters like temperature and top_p
5. Use either a Hugging Face model or a local fine-tuned model
6. Automatically play back both the reference and cloned audio for comparison
7. Add a speaker prefix to both reference and prompt text for multi-speaker models

>![TIP]
>For best results, use a high-quality reference audio file that's at least 5-10 seconds long, with clear speech and minimal background noise.

>![TIP]
>When using fine-tuned models, especially multi-speaker ones, be sure to use the `--speaker` parameter with the correct speaker name (e.g., `--speaker "Ronan"`). This ensures the model knows which voice to use for generation, even with voice cloning.

>![NOTE]
>The script will automatically play the reference audio and then the generated audio when the process completes. This feature works on macOS, Windows, and Linux systems with compatible audio players. Audio playback is automatically disabled in SSH sessions. You can also disable playback manually using the `--no_play` flag.

### Pushing Your Fine-Tuned Model to Hugging Face

After fine-tuning, you can push your model to Hugging Face Hub for easier sharing and reuse:
```bash
# Install required packages if needed
uv pip install transformers huggingface_hub colorama

# Login to Hugging Face (if not already logged in)
huggingface-cli login

```bash
uv run push_model_to_hub.py --model_path ./output/orpheus-tts-0.1-pretrained-ft --repo_id Trelis/orpheus-tts-0.1-pretrained-ft --private
```

The script will:
1. Validate that your model directory contains the necessary files
2. Create a HuggingFace repository with the specified visibility
3. Upload your model files with HF Transfer for faster uploads
4. Generate a model card with basic information and usage examples
5. Provide links to access your model on Hugging Face Hub