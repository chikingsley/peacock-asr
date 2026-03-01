# CSM-1B Fine-tuning Guide

## Prerequisites

- A CUDA-capable GPU with at least 16GB VRAM (recommended)
- Python 3.10
- Approximately 10-15 minutes of high-quality audio recordings of your voice

## Setup

Before starting the fine-tuning process, make sure you have set up the environment:

```bash
cd text-to-speech/Trelis-csm
uv venv --python 3.10.0
uv pip install -r requirements.txt
```

## Data Preparation
Note: This is working well.

### Option 1: Use the Data Collection Script

The easiest way to create a proper dataset is to use our data collection script:

```bash
uv run python fine-tune/collect_data.py
```

This interactive script will:
1. Display a series of sentences for you to read aloud
2. Guide you through recording each sentence (press Enter to start and again to stop)
3. Let you review each recording and re-record if needed
4. Automatically process and organize the recordings
5. Create a structured dataset ready for fine-tuning

The recording process is simple:
- Press Enter to START recording when prompted
- Read the displayed phrase clearly and at a natural pace
- Press Enter again to STOP the recording when you're finished
- Listen to the playback and choose whether to keep or discard the recording

### Option 2: Manual Data Preparation

If you prefer to prepare your own data:

1. Create a directory structure as follows:
   ```
   fine-tune/dataset/
   ├── metadata.csv
   └── wavs/
       ├── sample1.wav
       ├── sample2.wav
       └── ...
   ```

2. Record high-quality WAV files with the following specifications:
   - 24 kHz sample rate
   - Mono channel
   - Clear recordings in a quiet environment
   - Each recording should be 1-10 seconds long

3. Create a `metadata.csv` file with the following format:
   ```
   filename|text|speaker_id
   sample1.wav|This is the transcription of sample 1.|0
   sample2.wav|This is the transcription of sample 2.|0
   ```

   Notes:
   - The `filename` should be the relative path from the `wavs` directory
   - The `speaker_id` should be 0 for your voice (which will be used as the assistant voice)
   - The `text` should be the exact transcription of what was spoken

## Fine-tuning Process

### Current Issue: Attention Mask Shape Mismatch

We are currently encountering an attention mask shape mismatch error during training. The error message indicates:
```
output with shape [3, 32, 76, 76] doesn't match the broadcast shape [3, 3, 32, 76, 76]
```

#### Model Architecture Details
From `models.py`:
```python
# Backbone (1B)
num_heads=32,
num_kv_heads=8,
# Decoder (100M)
num_heads=8,
num_kv_heads=2,
```

#### Attempted Solutions

1. **Initial Approach**: Using simple causal masks
   ```python
   backbone_mask = torch.tril(torch.ones(b, 32, s, s))  # [batch_size, num_heads, seq_len, seq_len]
   decoder_mask = torch.tril(torch.ones(b, 8, s, s))
   ```
   Result: Shape mismatch error

2. **KV Heads Approach**: Tried to account for KV heads
   ```python
   backbone_mask = torch.tril(torch.ones(b, 8, 4, s, s))  # [batch_size, num_kv_heads, q_per_kv, seq_len, seq_len]
   decoder_mask = torch.tril(torch.ones(b, 2, 4, s, s))
   ```
   Result: Shape mismatch error

3. **Query Heads Approach**: Tried using query head dimensions
   ```python
   backbone_mask = torch.tril(torch.ones(b, 32, s, s))  # [batch_size, num_heads, seq_len, seq_len]
   decoder_mask = torch.tril(torch.ones(b, 8, s, s))
   ```
   Result: Shape mismatch error

4. **Current Attempt**: Added extra dimension
   ```python
   backbone_mask = torch.tril(torch.ones(b, 3, 32, s, s))  # [batch_size, 3, num_heads, seq_len, seq_len]
   decoder_mask = torch.tril(torch.ones(b, 3, 8, s, s))
   ```
   Result: Shape mismatch error

#### Key Observations
1. The error consistently shows a mismatch between a 4D tensor `[3, 32, 76, 76]` and a 5D tensor `[3, 3, 32, 76, 76]`
2. The model uses grouped-query attention (GQA) with different numbers of KV heads
3. The attention module's forward pass shows complex reshaping operations for key/value tensors

#### Next Steps Needed
1. Understanding the exact shape transformations in the attention module's forward pass
2. Determining how the mask should be shaped to match the expanded key/value tensors
3. Investigating if the mask needs to be transformed differently for GQA vs standard attention

### Historical Issues and Solutions

1. **Dtype Mismatch Issues**
   - Initially faced dtype mismatches between float32 and bfloat16
   - Fixed by explicitly converting tensors to the model's dtype (bfloat16) at each step
   - Added dtype conversions after backbone processing, decoder processing, and embedding operations

2. **In-place Operation Errors**
   - Encountered "modified by an inplace operation" errors during gradient computation
   - Fixed by creating new versions of model files (`models_train.py` and `generator_train.py`)
   - Removed all in-place operations and KV caches
   - Ensured new tensors are created instead of modifying existing ones

3. **Sequence Length Mismatch**
   - Initially faced issues with batching sequences of different lengths
   - Implemented custom collate function to pad sequences to the same length within each batch
   - Added proper padding for tokens, masks, and position indices

4. **Model Loading and Initialization Issues**
   - Initially faced recursion errors during model loading
   - Fixed by properly implementing the `from_pretrained` method
   - Ensured correct initialization of model components and attention configurations

### Current Implementation

The fine-tuning process now uses:
- Training-friendly versions of the model and generator (`models_train.py` and `generator_train.py`)
- Custom collate function to handle variable-length sequences
- TensorBoard for logging training metrics
- Proper handling of data types and tensor operations
- Focus on training only the projection layers (w1, w2, w3) while keeping other parameters frozen:
  - w1: projection from backbone to decoder
  - w2: projection to first codebook
  - w3: projections for remaining codebooks

### Running the Fine-tuning

To start the fine-tuning process:

```bash
uv run python fine-tune/fine-tune.py
```

The script will:
1. Load the CSM-1B model
2. Process your dataset
3. Train the attention modules while keeping other parameters frozen
4. Save checkpoints after each epoch
5. Log training metrics to TensorBoard

You can monitor the training progress using TensorBoard:
```bash
tensorboard --logdir runs/csm-fine-tuning
```

### Notes
- The training process focuses on fine-tuning only the attention modules to preserve the model's core capabilities while adapting it to your voice
- Checkpoints are saved after each epoch in the `checkpoints` directory
- Training metrics are logged to TensorBoard for visualization
- The model is trained in bfloat16 precision for efficiency