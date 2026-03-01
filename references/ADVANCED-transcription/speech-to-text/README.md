# Speech to Text
- `modal-whisper-server`: Deploy faster-whisper on Modal with OpenAI-compatible API. Supports Whisper Large v3 and v3 Turbo.
- `word-confidence-comparison`: Compare the performance of open source and private transcription APIs around speed and word confidence.
- `kyutai`: A fast streaming speech to text model. Now includes fine-tuning!
- `Trelis_whisper_fine-tuning_v2.0.0.ipynb`: Whisper Data Preparation and Fine-tuning Notebook, now using Unsloth.
- `Trelis-Whisper-Transcription.ipynb` ([Colab Link](https://colab.research.google.com/drive/1OkT0CLE219qbwQoXV94wNk_4Un7Du2sH?usp=sharing)): Audio file and YouTube transcription notebook.
- `voxtral`: Fine-tuning and serving for Voxtral Speech to Text model (LoRA on Modal + vLLM serving). See this folder for more.
- `whisper/modal-training`: Fine-tune Whisper-large-v3-turbo with LoRA on Modal. Before/after WER comparison.

- `archive/`:
    - `prepare-data.py`: Prepares a HuggingFace dataset, given mp3 audio files and vtt transcripts. (Now included in the fine-tuning notebook)
    - `prepare-data-multi-input.py`: Is a variant that will read in all mp3 files with the string 'train' and the string 'validation' in the file names, to be used as training and validation content, respectively.
    - `whisper-fine-tuning.ipynb`: A HuggingFace jupyter/colab notebook for transcription and fine-tuning (old script from HuggingFace with minor edits by Trelis).
    - `fine-tuning-whisper-v1.1.0.ipynb`: Deprecated (as of November 2025) Whisper fine-tuning script witih transformers.

What can you use this for?
- Fine-tuning transcription models for unfamilar terms/words/acronyms 
- Fine-tuning for specific accents
- Fine-tuning for new/uncommon languages

## Data Preparation
To fine-tune a model, you'll first need to generate a dataset with 30 second snippets of audio and corresponding captions (in vtt format).

**Audio Creation**

Quite simply, you can record yourself reading out some text:
- Once to generate train.mp3, and
- another time to generate validation.mp3 .
If your recording device saves audio files in another format, you can convert them with free online tools.

Before recording, it can be helpful to create `train_keywords.txt` and `validation_keywords.txt` files for you to read from:

- `train_keywords.txt` simply holds keywords on which I wish to train the model. I use this to create recordings for test and validation datasets, simply by reading out the words and recording myself.
- `validation_keywords.txt` is just a shuffled version of `train_keywords.txt`. I read out these words (in sentences) in order to create a validation set.

**Transcript Creation**

The fastest way to create a clean transcript is to first use a transcription tool.

`Trelis_whisper_fine-tuning_v2.0.0.ipynb` allows you do do that if you upload train.mp3 and validation.mp3 .

You can run the whisper-fine-tuning notebook on your laptop (slower) or on a free Colab notebook (slower).

Once you have created `train.vtt` and `validation.vtt`, you need to proof read and correct those files! See the ipynb notebook for more tips (including an automation to correct the files for you).

You'll then be able to segment your audio and transcription files in the notebook. (See below for the deprecated scripts method).

## Fine-Tuning
Once your data is up on HuggingFace Hub, you can go continue in the fine-tuning ipynb notebook and run through the steps for fine tuning.

After fine-tuning you can push your model to HuggingFace, including in OpenAI format or CTranslate2 format (for faster whisper).

Further guidance is provided on [Trelis' YouTube channel](https://YouTube.com/@TrelisResearch) via the two videos covering Whisper.

## Inference Server Setup

The fastest way to run inference is using Faster Whisper, which uses the CTranslate2 library to accelerate inference.

You can quickly spin up a Faster Whisper server using a Runpod template.

New to Runpod, you can use this affiliate link to support the Trelis YouTube channel:
- [Runpod](https://runpod.io/?ref=jmfkcdio)

### About the Faster Whisper Templates
>![TIP]
> Copied from the RunPod one-click template README.

Run an OpenAI-style endpoint for Whisper models, including Whisper Turbo or Whisper fine-tunes

Prepared by Trelis Research from [YouTube](https://youtube.com/@TrelisResearch), [HuggingFace](https://huggingface.co/Trelis), [GitHub](https://github.com/TrelisResearch).

Credit to https://github.com/fedirz/faster-whisper-server for the server library.

### Configuration
- Optionally change pre-loaded model to a custom model on HuggingFace.
- All models must be in Ctranslate2 format, see [here](https://github.com/SYSTRAN/faster-whisper) for conversion scripts.
- A correctly formatted Whisper Turbo model is set as the default.

Once up and running, you can make queries using OpenAI style requests.

For cURL requests, try:
```bash
curl https://<POD-ID>-8000.proxy.runpod.net/v1/audio/transcriptions \
     -F "file=@data/validation.mp3" \
     -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
     -F "language=en" \
     -F "response_format=vtt"
```

For python requests, try:
```python
from openai import OpenAI

# Set up the OpenAI client
client = OpenAI(api_key="EMPTY", base_url="https://<POD_ID>-8000.proxy.runpod.net/v1/")

# Open the audio file
audio_file = open("data/validation.mp3", "rb")

# Send the transcription request
transcript = client.audio.transcriptions.create(
    model="deepdml/faster-whisper-large-v3-turbo-ct2",  # Model to use
    file=audio_file,                                    # The audio file
    language="en",                                      # Language for transcription
    response_format="vtt"                               # Response format
)

# Print the transcription result
print(transcript.text)
```

If you specify a model that is not preloaded via env variables, it will be loaded on the fly. Note that specifying language speeds up transcription (because language is then autofilled in each line of the generated response).

### Simple and Batch Inference of One-click Templates
With a one-click template up and running, you can run inference using `simple-inference.py` or `concurrent-inference.py`. (Note that current requests are just handled in series, not yet in parallel.)

First, set up a virtual environment and install the inference requirements:
```
pip install --upgrade pip
python -m venv inferenceEnv
source inferenceEnv/bin/activate
pip install -r inference-requirements.txt
```

and optionally run:
```
export POD_ID="your_pod_id"
```
if you want to avoid being asked recurrently for your pod_id.

## Audio Segmentation via Scripts (Deprecated, and now supported in the fine-tuning ipynb)
Upload your cleaned train_corrected.vtt, cleaned validation_corrected.vtt and mp3 files to the data folder in this repo.

Next, set up a virtual environment (for Windows, get the tweaked commands from ChatGPT):
```
python -m venv asrEnv
source asrEnv/bin/activate
pip install -r requirements.txt
```
Now you are ready to run `python prepare-data.py`, which will automatically slice your audio and transcripts into 30 second segments and push them to HuggingFace as a dataset. If you haven't already logged into HuggingFace from command line, you can find commands online to do so using HuggingFace cli.

**Handling multiple mp3 and vtt files**
If you place multiple mp3 and vtt files in the `data` folder, you can run `prepare-data-multi-input.py` in order to automatically read in those files as inputs. Make sure that you have `train` and `validation` in the file names of all mp3 and vtt files you want to include. There must be one vtt file for every mp3 file, and they should have matching file-names (e.g. 'train1.mp3', 'train1.vtt').

## Changelog
25Feb2026:
- Add Voxtral fine-tuning (LoRA on Modal) and vLLM serving
- Add Whisper Modal fine-tuning script with before/after WER measurement
- Both models: baseline → fine-tuned WER on Trelis/llm-lingo (Voxtral 30.6%→14.6%, Whisper 37.0%→15.1%)

7jan2025:
- Add modal whisper server support
- Add word confidence comparison of transcription APIs

3jan2025:
- Support kyutai stt fine-tuning

19Nov2025:
- Support Whisper fine-tuning with Unsloth.

15Oct2024:
- Allow for Whisper Turbo model.
- Add Faster Whisper server instructions.

05Jul2024:
- add script for preparing data from multiple .mp3 files.
