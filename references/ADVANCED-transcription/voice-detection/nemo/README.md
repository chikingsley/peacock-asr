# Nemo Diarisation

## Installation
First run, for Ubuntu/Debian:

```
apt-get install sox libsndfile1 ffmpeg
```
and for MacOS (using Homebrew):
```
brew install sox libsndfile ffmpeg
```
Then set up a virtual environment:
```
pip install uv
uv venv --python 3.10
uv pip install ipykernel
uv run python -m ipykernel install --user --name=.venv --display-name="Python (.venv)" 
```

And then start the notebook (either in vs code or with jupyter or jupyter lab).
>![TIP]
>You may need to restart Cursor/Windsurf/VSCode after installing the kernel.

Note, when running clustering, you can add the following to unset multiprocessing:
```
config.num_workers = 0  # Set to 0 to disable multiprocessing and avoid pickling errors
```
To use multiprocessing you may look into why there is a pickling issue.

## How NeMo Diarization Works

NeMo's diarization system consists of a pipeline with several components working in sequence:

### Pipeline Components

1. **Voice Activity Detection (VAD)**
   - Identifies speech vs. non-speech regions
   - Can use either:
     - Oracle VAD (ground truth from RTTM files)
     - NeMo's VAD model (MarbleNet-based, ~1MB)
   - Smaller and simpler than Silero VAD

2. **Segmentation**
   - Unlike Pyannote, NeMo uses deterministic segmentation (not a neural model)
   - Process:
     1. VAD first identifies speech/non-speech at the frame level (typically 10-20ms frames)
     2. Contiguous speech frames are grouped into speech regions
     3. These speech regions are then divided into overlapping windows of different sizes
     4. Each scale uses a different window length (e.g., 1.5s, 1.25s, 1.0s, 0.75s, 0.5s)
     5. Windows slide across speech regions with corresponding shift lengths
   - The multi-scale approach processes the same speech content at different temporal resolutions
   - Each scale's segmentation is processed independently through the embedding and clustering stages

3. **Speaker Embedding Extraction (TitaNet, ~102MB)**
   - Extracts speaker characteristics from audio segments
   - Architecture:
     - Input: Mel-spectrogram features (80 dimensions)
     - Multiple JasperBlocks with depthwise separable convolutions
     - SqueezeExcite attention mechanisms for better feature extraction
     - AttentivePoolLayer for aggregating frame-level features
     - Output: 192-dimensional speaker embeddings
   - Processes each segment to create a vector representing speaker characteristics

4. **Clustering**
   - Uses spectral clustering to group similar embeddings
   - Determines number of speakers and creates speaker profiles
   - Similar to Pyannote's approach but with multi-scale fusion

5. **Neural Diarization (MSDD, ~5MB)**
   - Multi-Scale Diarization Decoder (MSDD)
   - Refines clustering results and handles overlapping speech
   - Uses a pairwise approach to process speaker combinations

The MSDD neural diarizer uses a clever pairwise approach:

1. For N speakers detected by clustering, it processes all N(N-1)/2 possible pairs
2. For each pair (A,B), it outputs:
   - Is speaker A active at this frame? (0-1 probability)
   - Is speaker B active at this frame? (0-1 probability)
3. If a speaker appears in multiple pairs, results are averaged
4. This approach allows detecting overlapping speech and handling any number of speakers

### Neural Diarizer (MSDD) Architecture in Detail

The MSDD component is a sophisticated neural architecture designed to refine speaker assignments and detect overlapping speech. Despite being only ~5MB in size (compared to TitaNet's ~102MB), it plays a crucial role in the diarization pipeline.

#### MSDD Layer Structure

1. **Input Processing**
   - Takes speaker embeddings from TitaNet as input
   - Processes speaker pairs to determine who is speaking when

2. **Convolutional Feature Extraction**
   - Two parallel convolutional layers with different kernel sizes:
     - First ConvLayer: 16 filters with 15×1 kernel size
     - Second ConvLayer: 16 filters with 16×1 kernel size
   - Each followed by ReLU activation and BatchNorm
   - Captures patterns at different temporal scales

3. **Temporal Modeling**
   - Bidirectional LSTM with:
     - 2 layers
     - 256 hidden units per direction
     - 0.5 dropout rate
   - Models long-range dependencies between speaker turns
   - Captures the temporal dynamics of conversations

4. **Decision Layers**
   - `conv_to_linear`: Transforms convolutional features (3072 dimensions) to 256 dimensions
   - `linear_to_weights`: Produces 5 weight values for decision making
   - `hidden_to_spks`: Final layer that outputs 2 values per speaker pair
   - `dist_to_emb`: Processes cosine similarity features (10 dimensions) to 256 dimensions

5. **Output Generation**
   - For each pair of speakers (A,B), outputs:
     - Probability that speaker A is active at each frame
     - Probability that speaker B is active at each frame
   - When a speaker appears in multiple pairs, results are averaged
   - Final output: Frame-level speaker activity probabilities

#### Pairwise Processing Approach

For N speakers detected by clustering:
1. The system processes all N(N-1)/2 possible speaker pairs
2. For each frame, it determines if either or both speakers in a pair are active
3. This approach naturally handles overlapping speech
4. Results are combined to produce the final diarization output

### Comparing Different Approaches to Speaker Detection

There are several ways to approach speaker detection/diarization:

1. **NeMo's Approach**:
   - VAD: MarbleNet-based VAD (~1MB) or Oracle VAD
   - Segmentation: Deterministic windowing
   - Speaker Identification: TitaNet embeddings (~102MB) + spectral clustering
   - Overlap Detection: MSDD neural model (~5MB)
   - Strengths: Multi-scale processing, high-quality embeddings, overlap detection
   - Weaknesses: Large models, not real-time capable

2. **Pyannote's Approach**:
   - VAD: Derived from PyanNet segmentation model
   - Segmentation: Neural PyanNet model
   - Speaker Identification: ECAPA-TDNN embeddings + clustering
   - Overlap Detection: Built into segmentation model
   - Strengths: Integrated pipeline, good overlap detection
   - Weaknesses: Limited to detecting 3 speakers in segmentation

3. **Hybrid Approach (Silero + Embedding)**:
   - VAD: Lightweight Silero VAD
   - Segmentation: Simple windowing of VAD regions
   - Speaker Identification: Any embedding model + clustering
   - Strengths: Faster VAD, more flexible
   - Weaknesses: May miss nuanced speaker transitions

4. **Speaker Identification (not Diarization)**:
   - For known speakers with pre-enrolled voice profiles
   - Can use any VAD (Silero, NeMo, Pyannote) to detect speech
   - Extract embeddings from speech segments
   - Compare to known speaker profiles (no clustering needed)
   - Better suited for real-time applications

### Model Weights Location

When you run NeMo diarization, models are automatically downloaded to:
```
~/.cache/torch/NeMo/NeMo_[version]/
```

For example:
```
/Users/[username]/.cache/torch/NeMo/NeMo_2.3.0rc0/titanet-l/[hash]/titanet-l.nemo
/Users/[username]/.cache/torch/NeMo/NeMo_2.3.0rc0/diar_msdd_telephonic/[hash]/diar_msdd_telephonic.nemo
```

The complete neural diarizer package (~107MB) includes:
- TitaNet speaker embedding model (~102MB)
- MSDD neural diarization component (~5MB)
- Other small supporting components

### Limitations for Real-Time Applications

NeMo diarization (like Pyannote) has significant limitations for real-time use:

1. **Global Processing Requirement**: Clustering needs the entire audio to determine speakers
2. **Look-Ahead Dependencies**: Neural models use bidirectional processing requiring future context
3. **Batch Processing Design**: Designed for complete recordings, not streaming audio
4. **Cannot Handle Incomplete Turns**: Not designed to predict if a speaker will continue

For real-time applications, a pipeline using Silero VAD + pre-enrolled speaker profiles would likely be more effective than trying to adapt full diarization systems like NeMo or Pyannote.
