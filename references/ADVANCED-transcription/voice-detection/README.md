# Voice Detection (incl. Turn Detection and Diarisation)

This folder is includes:
- This README.md
  - Voice detection with Silero VAD
  - Turn detection with Pipecat Smart Turn
  - Speaker diarization with Pyannote
  - Speaker detection with Pyannote
  - Speaker recognition with Pyannote
  - Speaker clustering with Pyannote
  - Speaker diarisation with Nemo (with and without a neural diarizer)
- Diarisation with Nvidia NeMo: See [nemo/README.md](nemo/README.md)

## Silero VAD

A lightweight (2MB) voice activity detector that can process audio in real-time.

### Technical Details

The model uses a sophisticated neural architecture optimized for real-time speech detection:

1. **Dual Sample Rate Support**: Contains two identical model paths for 16kHz and 8kHz audio
2. **Architecture**:
   - STFT (Short-Time Fourier Transform) frontend for frequency analysis
   - 4-layer encoder with specialized VAD blocks (Conv1D + ReLU)
   - LSTM-based decoder for temporal modeling
   - Binary classifier (speech/non-speech) with sigmoid activation

**Model Size**: 
- Total parameters: 462,594
- Trainable parameters: 264,450
- Compressed size: ~2MB

**Processing**:
- Works on 32ms chunks (512 samples at 16kHz)
- Merges segments with less than 100ms silence between them
- Filters out segments shorter than 250ms
- Configurable sensitivity via threshold parameter (0.0-1.0)

<details>
<summary>Comparison with Diarization Approaches</summary>

**Comparison with Pyannote Segmentation**:
- Silero VAD uses STFT to analyze frequencies, while Pyannote uses SincNet to learn directly from waveforms
- Silero's simpler architecture (4 Conv1D layers + single LSTM) focuses on speed and basic speech detection
- Pyannote's more complex architecture (learnable filters + 4 bidirectional LSTMs) enables speaker identification
- Neither model captures linguistic meaning, but Pyannote better captures speaker characteristics and speech patterns

**Understanding Model Capabilities**:
- **Silero VAD**: Uses fixed frequency analysis (STFT) to detect speech presence
- **Pyannote**: Uses learnable bandpass filters (SincNet) that can adapt to important frequency ranges
  - Can learn prosodic features (rhythm, stress, intonation)
  - Captures temporal patterns through bidirectional LSTMs
  - Better at understanding "how" speech is delivered
  - Segments based on acoustic patterns, not linguistic content
  - Limited with incomplete turns because it's trained on speaker changes, not turn completeness
- **Whisper**: Optimized for semantic understanding ("what" was said)
  - May capture some prosody but as a byproduct
  - Different focus than VAD/segmentation models

**How Segmentation Works**:
1. **Acoustic-Based Segmentation**:
   - Uses SincNet to analyze raw audio patterns
   - Learns to detect changes in speaker characteristics
   - Identifies transitions based on voice properties, not meaning
   
2. **Limitations for Turn Detection**:
   - Training data focuses on speaker transitions
   - Can detect when speakers change but not if a turn is complete
   - May indirectly capture some turn-taking cues through prosody:
     - Rising/falling intonation patterns
     - Rhythm changes in filler words ("um", "uh")
     - Speech rate variations
   - But lacks explicit training on these patterns for turn completion
   - Cannot reliably distinguish between:
     - Natural pauses vs turn endings
     - Filler words vs content words
     - Complete vs incomplete phrases

**Key Challenge for Real-time Systems**:
The fundamental limitation isn't that segmentation fails - it successfully detects speaker changes and segments. The real challenge is prediction: when a speaker pauses, there's no way to know if they:
1. Are done speaking (complete turn)
2. Are pausing to think (incomplete turn)
3. Are using a rhetorical pause
4. Have been interrupted

This is why turn detection (Smart Turn) needs linguistic understanding - it must predict future intent, not just detect current state.

Each model specializes in different aspects of speech:
- VAD → Speech presence
- Segmentation → Speaking style and speaker identity
- Whisper → Semantic content

The model will output timestamps for each detected speech segment.

</details>

### Quick Test

1. Install dependencies:
```bash
uv venv
uv pip install torch torchaudio sounddevice soundfile
```

2. Record test audio:
```bash
uv run python -c '
import sounddevice as sd
import soundfile as sf

# Record audio
duration = 10  # seconds
fs = 16000    # sample rate
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
print("Recording... speak for 10 seconds")
sd.wait()

# Save as WAV file
sf.write("conversation.wav", recording, fs)
print("Saved recording to conversation.wav")
'
```

3. Run VAD:
```bash
uv run python -c '
import torch
import torchaudio
import numpy as np

# Load the model
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                            model="silero_vad",
                            force_reload=True,
                            trust_repo=True)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = model.to(device)

# Get VAD function
get_speech_timestamps = utils[0]

# Load audio
wav, sr = torchaudio.load("conversation.wav")
wav = wav.to(device)

# Get speech timestamps
speech_timestamps = get_speech_timestamps(
    wav[0],                    # Audio samples
    model,                     # VAD model
    sampling_rate=sr,          # Sample rate (16000)
    threshold=0.3,             # Speech threshold (0.0-1.0, lower = more sensitive)
    min_speech_duration_ms=250,# Minimum speech segment length
    min_silence_duration_ms=100,# Minimum silence between segments
    window_size_samples=512    # Size of each chunk to process
)

# Print results
print("\nSpeech segments detected:")
for i, segment in enumerate(speech_timestamps):
    start_time = segment["start"] / sr
    end_time = segment["end"] / sr
    print(f"Segment {i+1}: {start_time:.1f}s -> {end_time:.1f}s")
'
```

The model will output timestamps for each detected speech segment. Note that Silero VAD:
- Only detects presence/absence of speech
- Works on 30ms chunks in real-time
- Uses small buffers (200ms before, 1000ms after) to avoid clipping
- Does not identify speakers or detect turn completion

We'll see later how to use MarbleNet VAD - from Nvidia, which is smaller.

## Turn Detection: Pipecat Smart Turn

Trained on [this](https://huggingface.co/datasets/pipecat-ai/human_5_all/viewer/default/train?sort%5Bcolumn%5D=endpoint_bool&sort%5Bdirection%5D=asc&row=1932) labelled data and synth data that does not improve performance much.

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

### macOS (using Homebrew)
```bash
brew install portaudio
```

### Usage

Run a command-line utility that streams audio from the system microphone, detects segment start/stop using VAD, and sends each segment to the model for a phrase endpoint prediction.

```bash
git clone https://github.com/pipecat-ai/smart-turn.git
cd smart-turn
```

> **Note**: It will take about 30 seconds to start up the first time.

```bash
uv venv --python 3.9
uv pip install -r requirements.txt
uv run record_and_predict.py
```

### Example Phrases
The "vocabulary" is limited. Try phrases like:
- "I can't seem to, um ..."
- "I can't seem to, um, find the return label."

## Pyannote Segmentation

The segmentation model (5.9MB) is a more sophisticated version of VAD that can:
- Process 10-second chunks of 16kHz mono audio
- Detect multiple speakers and their overlaps
- Output 7 classes: non-speech, 3 individual speakers, and their pairwise overlaps (1&2, 1&3, 2&3)

The model has some key limitations:
- Maximum of 3 speakers per chunk due to the powerset encoding
- Overlap detection works by identifying when two speakers are active simultaneously
- Cannot detect three-way overlaps (all speakers talking at once)

### Inspect Model

To see the model's architecture and parameters:
```bash
uv run python -c '
from pyannote.audio import Model
import torch

# Load model
model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token="YOUR_HUGGINGFACE_TOKEN"
    )

# Save model info to file
with open("segmentation_model_info.txt", "w") as f:
    print(model, file=f)
    print("\nModel state keys:", file=f)
    print(model.state_dict().keys(), file=f)
    print("Model information saved to segmentation_model_info.txt")
'
```

The model architecture (PyanNet) consists of several key components:

1. **SincNet Frontend**:
   - Processes raw audio using learnable bandpass filters
   - Uses instance normalization and three convolutional layers
   - Reduces dimensionality through max pooling
   - Output: 60 frequency channels

2. **LSTM Sequence Processor**:
   - 4-layer bidirectional LSTM
   - 128 units per direction (256 total)
   - 50% dropout for regularization
   - Processes temporal relationships in the audio

3. **Classification Head**:
   - Two linear layers (256→128→128)
   - Final classifier outputs 7 classes:
     - Non-speech
     - 3 individual speakers
     - 3 pairwise overlaps
   - LogSoftmax activation for class probabilities

This architecture combines:
- Efficient raw audio processing (SincNet)
- Long-range temporal modeling (LSTM)
- Multi-class classification for speaker states

The model is relatively small (5.9MB) but effective for real-time segmentation tasks.

### Quick Test

1. Install and setup:
```bash
uv venv
uv pip install pyannote.audio sounddevice soundfile torch

# Accept model terms at huggingface.co and get token
```

2. Record test audio:
```bash
# Record 10 seconds of conversation between two speakers
uv run python -c '
import sounddevice as sd
import soundfile as sf

# Record audio
duration = 10  # seconds
fs = 16000    # sample rate
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
print("Recording... speak for 10 seconds")
sd.wait()

# Save as WAV file
sf.write("conversation.wav", recording, fs)
print("Saved recording to conversation.wav")
'
```

3. Test segmentation and VAD:
```bash
uv run python -c '
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import torch

# Load segmentation model
model = Model.from_pretrained(
    "pyannote/segmentation-3.0", 
    use_auth_token="YOUR_HUGGINGFACE_TOKEN"
)

# Setup VAD pipeline
pipeline = VoiceActivityDetection(segmentation=model)
pipeline.instantiate({
    "min_duration_on": 0.0,   # minimum speech duration
    "min_duration_off": 0.0   # minimum silence duration
})

# Use GPU if available
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "mps"))

# Process the recording
vad = pipeline("conversation.wav")

# Print VAD results
for speech in vad.get_timeline():
    print(f"Speech detected: {speech.start:.1f}s -> {speech.end:.1f}s")
'
```

4. Test overlap detection:
```bash
uv run python -c '
from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection
import torch

# Use same model as above

model = Model.from_pretrained(
"pyannote/segmentation-3.0",
use_auth_token="YOUR_HUGGINGFACE_TOKEN"
)

pipeline = OverlappedSpeechDetection(segmentation=model)
pipeline.instantiate({
    "min_duration_on": 0.0,   # minimum overlap duration
    "min_duration_off": 0.0   # minimum non-overlap duration
})

# Use GPU if available
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "mps"))

# Process the recording
osd = pipeline("conversation.wav")

# Print overlap results
for overlap in osd.get_timeline():
    print(f"Speaker overlap: {overlap.start:.1f}s -> {overlap.end:.1f}s")
'
```

Note: This segmentation model is just one part of the full diarization pipeline. For full speaker identification, use the diarization pipeline described in the next section.

## Pyannote Diarization

To inspect the full pyannote pipeline, run:
```bash
uv pip install pyannote.audio
uv run python inspect_pyannote.py
```

Pyannote provides speaker diarization - identifying who spoke when in an audio recording. Note that it provides timestamps and speaker labels, but not transcripts.

### Architecture

Pyannote's diarization system consists of three main components working together:

1. **Segmentation Model (PyanNet)**
   - Uses SincNet frontend for raw audio processing:
     - Learnable bandpass filters
     - Instance normalization and three convolutional layers
     - Max pooling for dimensionality reduction
   - Deep LSTM for temporal processing:
     - 4 bidirectional layers
     - 128 units per direction (256 total)
     - 50% dropout for regularization
   - Classification head outputs 7 classes:
     - Non-speech
     - 3 individual speakers
     - 3 pairwise speaker overlaps

2. **Speaker Embedding Model (ECAPA-TDNN)**
   - Based on SpeechBrain's ECAPA-TDNN architecture
   - State-of-the-art speaker recognition model
   - Trained on the VoxCeleb dataset
   - Converts speech segments into speaker embeddings
   - Enables speaker identification across segments

3. **Diarization Pipeline**
   - Orchestrates the entire process:
     1. Segments audio using PyanNet
     2. Extracts speaker embeddings using ECAPA-TDNN
     3. Clusters similar embeddings to identify unique speakers
     4. Generates timeline of who spoke when
   - Handles end-to-end processing from raw audio to speaker labels

This architecture combines sophisticated audio processing (SincNet), temporal understanding (LSTM), and speaker recognition (ECAPA-TDNN) to achieve robust diarization performance.

### Setup

1. Install pyannote:
```bash
uv venv
uv pip install -r requirements-pyannote.txt
```

2. Get HuggingFace access token:
- Go to https://hf.co/settings/tokens
- Create a new token
- Accept the user conditions for models:
  - pyannote/segmentation-3.0
  - pyannote/speaker-diarization-3.1

3. Create a test recording:
```bash
# Record 10 seconds of conversation between two speakers
uv run python -c '
import sounddevice as sd
import soundfile as sf

# Record audio
duration = 10  # seconds
fs = 16000    # sample rate
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
print("Recording... speak for 10 seconds")
sd.wait()

# Save as WAV file
sf.write("conversation.wav", recording, fs)
print("Saved recording to conversation.wav")
'
```

4. Run diarization (put in your huggingface token):
```bash
uv run python -c '
from pyannote.audio import Pipeline
import torch

# Initialize pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HUGGINGFACE_TOKEN"
)

# Use GPU if available
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "mps"))

# Process the audio file
diarization = pipeline("noisy_overlapping.wav")

# Print results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s -> {turn.end:.1f}s] {speaker}")
'
```

The output will look like:
```
[0.2s -> 1.5s] speaker_0
[1.8s -> 3.9s] speaker_1
[4.2s -> 5.7s] speaker_0
```
Each line shows:
- Start and end timestamps in seconds
- Speaker label (speaker_0, speaker_1, etc.)
- Note: These are arbitrary labels - the model doesn't identify specific people, it just groups speech segments by speaker

### Tips
- Ensure clear audio with minimal background noise
- Try to avoid speaker overlap for best results
- The model works best with recordings between 30 seconds and 30 minutes
- For real-time applications, consider using Smart Turn instead

...





## Nemo Diarisation

See [nemo/README.md](nemo/README.md)

## FAQs

**How does Voice Activation Detection (VAD) work?**
VAD uses a lightweight neural network (Silero VAD) to classify short audio chunks (30ms) as speech or non-speech. It processes audio in real-time, maintaining small buffers before (200ms) and after (1000ms) detected speech to avoid clipping. The model is trained on a massive dataset covering 6000+ languages, making it robust across different speakers and conditions.

**Why is VAD not good enough for turn detection?**
VAD only detects the presence/absence of speech, without understanding linguistic content or context. It can't differentiate between natural pauses and true turn completions, or understand linguistic cues like grammar structure, intonation, or filler words that humans use to signal turn completion.

**How does Smart Turn solve this?**
Smart Turn adds a second layer of intelligence using a Wav2Vec2-BERT model that understands both acoustic and linguistic patterns. While VAD handles basic speech detection, Smart Turn analyzes the full context of the speech segment to determine if it's truly complete or if the speaker is likely to continue. It can recognize patterns like incomplete phrases or filler words that suggest an incomplete turn.

**Why not just use a Silero model fine-tuned on triads (complete, incomplete, noise)?**
While combining these classifications into one model seems logical, the two-stage approach is more practical because:
1. It separates simple, fast speech detection (VAD) from complex linguistic analysis
2. The lightweight VAD model (2MB) can run efficiently on short chunks while the larger model (580M parameters) only processes actual speech
3. Each model can be trained and improved independently with different types of training data
4. The two-stage approach enables real-time processing while maintaining sophisticated linguistic understanding

Really, the issue is likely that silero doesn't capture enough semantic information for this to work well.

**How is turn detection different than diarisation?**
Turn detection and diarization solve different problems in conversation analysis:
- Turn Detection (Smart Turn): Makes real-time decisions about WHEN a speaker has finished their turn, useful for conversational AI that needs to know when to respond
- Diarization (Pyannote): Analyzes WHO spoke WHEN in a recording, typically working retrospectively on complete recordings to identify and label different speakers

While they can be complementary:
- Turn detection helps with real-time interaction
- Diarization helps with post-processing and transcription
- Combining both could give you both speaker identification and turn completion detection