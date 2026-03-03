# Execution: From Research to Production

Step-by-step implementation plan, training infrastructure, deployment, and costs.

**Hardware**: RunPod cloud GPUs (A100 80GB / H100 available) | RTX 5070 12GB (gmk-server for inference/prototyping) | macOS dev machine

---

## 1. Executive Summary

The pronunciation assessment field is at an inflection point. Four key advances in 2024-2025 make it possible to build a system that matches or beats commercial offerings:

1. **Segmentation-free GOP** (2507.16838) eliminates the forced alignment bottleneck
2. **LoRA-tuned multimodal LLMs** achieve PCC > 0.7 on a single RTX 4090
3. **NeMo's 17MB pronunciation scorer** beats human expert correlation (0.580 vs 0.555)
4. **Qwen3-ASR** provides SOTA open-source ASR with a dedicated forced aligner (Apache 2.0)

No existing system combines all of these advances. That is the gap.

---

## 2. The Gap We Fill

### Open-Source SOTA (What Exists Today)

| System | Approach | Performance | Limitation |
|--------|----------|-------------|------------|
| **GOPT** (this repo) | GOP features + Transformer | PCC 0.612 phone, 0.742 sentence | Requires Kaldi for GOP extraction |
| **GOP-SF** (2507.16838) | Segmentation-free GOP via CTC | AUC 0.914 (best on CMU Kids) | Research paper only, no production code |
| **NeMo Parakeet scorer** | CTC + tiny MLP head | PCC 0.580 (beats human 0.555) | Only 17MB model, limited to NeMo ecosystem |
| **Phi-4 + LoRA** (2509.02915) | Multimodal LLM fine-tuned | PCC > 0.7, WER/PER < 0.15 | 5.6B params, high inference cost |

### The Gap We Fill

No open-source system currently provides:
- Modern base model (not Kaldi) for GOP extraction
- Segmentation-free scoring (alignment-agnostic)
- Multi-aspect assessment (phoneme + word + sentence + prosody)
- Mispronunciation detection AND diagnosis in one pipeline
- Production-ready API with <200ms latency
- ONNX-optimized deployment

---

## 3. Training Code Repo Triage

This section provides a repo/model view focused on one criterion: **can we actually train or fine-tune this in a reproducible way for pronunciation work?**

### Priority Stack (Training-Code First)

| Priority | Repo / Model Family | Training Code Status | Best Use in Roadmap | Main Caveat |
|---|---|---|---|---|
| P0 | **Kaldi GOP recipe** (`kaldi/egs/gop_speechocean762`) | Full recipe + scripts | Baseline reproduction and sanity checks for phone-level GOP | Older TDNN/GMM-HMM stack, not modern multilingual |
| P0 | **Meta Omnilingual-ASR** | Explicit data prep + train/eval recipes for CTC/LLM | Multilingual backbone with CTC variants that fit GOP-style feature extraction | Need careful unit mapping for phone-level scoring |
| P0 | **fairseq wav2vec2/XLSR** | End-to-end pretrain + CTC fine-tune commands | Strong controllable CTC backbones for GOP experiments | More engineering overhead than turnkey APIs |
| P1 | **Qwen3-ASR + ForcedAligner** | Fine-tuning script exists (`finetuning/qwen3_asr_sft.py`) | High-quality transcription + alignment-assisted diagnostics | Main ASR output is text-focused; not a direct phone-posterior stack |
| P1 | **NVIDIA Parakeet CTC (NeMo)** | NeMo training scripts/configs referenced directly | Strong English CTC baseline for fast MVP | English-focused model cards (still useful for EN-first) |
| P2 | **Allosaurus** | Built-in adaptation workflow (`adapt_model`) | Fast universal phone-recognition baseline and low-cost adaptation | Fine-tuning scope is single-language and timestamp is CTC-approximate |

### Repo-Level Detail

**Kaldi**
- Keep it as the baseline truth anchor.
- `gop_speechocean762/s5/run.sh` is still the clean reference for GOP feature extraction pipeline stages.

**fairseq / original wav2vec direction**
- Still a great research backbone if you want full control:
  - `examples/wav2vec/README.md` includes wav2vec2 pretraining and CTC fine-tuning commands.
  - If you want explicit sequence-to-sequence workflows, fairseq has `examples/speech_to_text/README.md` (S2T framework), separate from wav2vec-CTC.

**Allosaurus**
- Practical for phone recognition experiments.
- Includes a concrete fine-tune path (`prep_feat`, `prep_token`, `adapt_model`) and can run CPU-only if needed.

**Omnilingual-ASR (Meta)**
- Ships both CTC and LLM model families, and exposes recipe-driven training/eval workflows.
- For pronunciation assessment, start with **CTC 300M/1B** before jumping to LLM variants.

**Parakeet CTC (NVIDIA)**
- Strong EN CTC baseline and has explicit NeMo training script/config references.
- Still include even if English-focused.

### Track A: English-First MVP (fastest to useful APA/MDD signal)

1. Reproduce Kaldi GOP baseline (`gop_speechocean762`) to lock reference metrics.
2. Add one modern English CTC backend:
   - Option 1: Parakeet CTC 0.6B (fastest practical path).
   - Option 2: fairseq wav2vec2 CTC fine-tune (more control).
3. Keep GOPT head unchanged and compare feature quality against Kaldi GOP features.

### Track B: Multilingual-first backbone

1. Start Omnilingual **CTC_300M_v2** inference + small fine-tune recipe run.
2. Validate whether token units can be mapped cleanly into pronunciation scoring features.
3. Scale to CTC_1B_v2 only after feature compatibility is confirmed.

### Track C: High-quality transcription/alignment sidecar

1. Use Qwen3-ASR (0.6B or 1.7B) for robust transcript + language ID.
2. Use Qwen3-ForcedAligner for timestamped feedback layers.
3. Keep this as complementary to CTC-GOP stack until phone-level posterior integration is clean.

### Where We Need To Dig Deeper

1. **Omnilingual unit-to-phoneme mapping** -- Confirm whether tokenizer outputs can be transformed into stable phone-level features compatible with GOP/GOPT.
2. **Qwen3 for phone-level scoring** -- Great for ASR + timestamps, but we still need a clear bridge from text-oriented outputs to phone-confidence style features.
3. **Parakeet licensing/data constraints** -- Model cards note mixed private/public training data and CC-BY-4.0 terms. Confirm product/legal fit.
4. **Fairseq vs fairseq2 implementation choice** -- Fairseq is rich for wav2vec experiments; Omnilingual is on fairseq2. Decide early which stack is your long-term training infra.
5. **Allosaurus limits** -- Useful and fast, but adaptation constraints (single-language fine-tune) mean it should be baseline/ablation, not final backbone.

### Suggested Immediate Execution Order

1. **This week** -- Kaldi baseline re-run + Parakeet CTC feature-prototype + GOPT retrain.
2. **Next week** -- Omnilingual CTC_300M_v2 recipe trial on a narrow dataset subset.
3. **Then** -- Add Qwen3-ASR/Aligner as inference sidecar for richer feedback output.

---

## 4. Architecture Decision

### Recommended: Two-Track Approach (Unconstrained GPU via RunPod)

With access to A100 80GB / H100 GPUs on RunPod, we can go aggressive on both tracks -- using the largest, best-performing models without compromise.

**Track A: Production (Ship Fast, Serve Cheap)** -- WavLM-large CTC + GOP-SF + GOPT
- Uses the **large** (317M) model instead of base -- significantly better phoneme representations
- Full fine-tuning on A100 (no need for LoRA workarounds)
- After training, deploy as ONNX int8 (~150MB) -- serves on CPU or cheap GPU
- <200ms inference latency

**Track B: Maximum Quality** -- Phi-4-multimodal + LoRA (Primary Recommendation)
- Full bf16 LoRA on A100 -- exactly how the paper did it, no QLoRA compromises
- Single model does everything: APA + MDD + transcription simultaneously
- Can also experiment with larger LoRA ranks (16-32) and full audio encoder unfreezing
- Achieves highest accuracy ceiling (PCC > 0.7 demonstrated)

**Track C: Experimental** -- Ensemble / Model Soup
- Combine Track A scores + Track B scores for best-of-both
- Or try data2vec2-large / XLS-R-1B as even stronger CTC backbones
- Whisper-large-v3 fine-tuned for phoneme output as an alternative backbone

### Architecture Diagrams

```markdown
Track A: Production Pipeline (train on A100, deploy anywhere)
=============================================================
Audio (16kHz) --> [WavLM-large CTC (317M)] --> Frame-level posteriors
                                                     |
                                                     v
Canonical Text --> [GOP-SF (alignment-free)] --> GOP Feature Vectors [50 x feat_dim]
                                                        |
                                                        v
                                              [GOPT Transformer] --> Scores
                                                                     - Phoneme accuracy
                                                                     - Word accuracy/stress
                                                                     - Utterance fluency/prosody/total

Train: A100 80GB | Deploy: ONNX int8 ~150MB, <200ms on CPU | Serve cost: minimal

Track B: Maximum Quality Pipeline (train & serve on GPU)
========================================================
Audio (16kHz) + Canonical Text --> [Phi-4-multimodal + LoRA (bf16)]
                                          |
                                          v
                                   Structured Output:
                                   - Phoneme transcription
                                   - Per-phoneme scores
                                   - Mispronunciation labels
                                   - Word/sentence scores
                                   - Diagnostic feedback

Train: A100 80GB, bf16, ~2-4h | Deploy: ~11GB bf16 | Serve: GPU required

Track C: Experimental (maximum accuracy ceiling)
================================================
Audio --> [data2vec2-large / XLS-R-1B CTC] --> posteriors
       +  [Phi-4-multimodal + LoRA] --> LLM scores
       --> Ensemble scoring head --> Final scores

Train: A100/H100 | Deploy: flexible (ensemble or distill to single model)
```

---

## 5. Base Model Selection

### For Track A (Production -- train large, deploy small)

With A100 80GB, use the **large** variants for training -- they produce substantially better representations. The model is only large at training time; at deployment you export to ONNX int8.

| Model | Params | Why | VRAM (Train) | VRAM (Infer ONNX int8) |
|-------|--------|-----|-------------|----------------------|
| **WavLM-large** | 317M | Best SSL model for phoneme tasks; 94K hours pre-training; large models significantly outperform base | ~20-25GB FP16 full fine-tune | ~300MB (~80MB int8) |
| **data2vec2-large** | 317M | Research shows it outperforms HuBERT and Whisper for pronunciation assessment specifically | ~20-25GB FP16 | ~300MB (~80MB int8) |
| **XLS-R-1B** | 1B | 436K hours, 128 languages -- best for multilingual pronunciation | ~40-50GB FP16 | ~1GB (~250MB int8) |
| WavLM-base+ | 95M | Fallback if you want faster iteration | ~6-8GB FP16 | ~100MB (~25MB int8) |

**Recommendation**: Start with **WavLM-large** (full fine-tune on A100). If targeting multilingual, go **XLS-R-1B**. data2vec2-large is the dark horse -- specifically proven best for pronunciation tasks but less community support.

### For Track B (Maximum Quality)

No need for QLoRA compromises -- run full bf16 LoRA on A100.

| Model | Params | Audio Support | License | VRAM (bf16 LoRA) |
|-------|--------|---------------|---------|-----------------|
| **Phi-4-multimodal-instruct** | 5.6B | Native audio encoder + projector | MIT | ~24-30GB (bf16 LoRA) |
| Qwen-Audio-Chat | 7B | Native audio understanding | Apache 2.0 | ~30-40GB (bf16 LoRA) |
| Qwen3-ASR | ~600M | Dedicated ASR + forced aligner | Apache 2.0 | ~8-12GB (full fine-tune) |

**Recommendation**: **Phi-4-multimodal** on A100 80GB with bf16 LoRA -- reproduces the paper setup exactly. Can also experiment with **higher LoRA rank (16-32)** and **unfreezing the audio encoder** for potentially even better results (the paper only tuned LoRA adapters).

### For Track C (Experimental)

| Model | Params | Why | VRAM |
|-------|--------|-----|------|
| **Whisper-large-v3-turbo** | 809M | Largest fine-tuning ecosystem; can be adapted for phoneme output | ~35-40GB full fine-tune |
| **XLS-R-1B** | 1B | Massive multilingual CTC backbone | ~50GB full fine-tune |
| **Canary-Qwen-2.5B** | 2.5B | #1 on Open ASR Leaderboard | ~40GB LoRA |

**Recommendation**: These are for experimentation after Tracks A and B are established. XLS-R-1B as CTC backbone could be the strongest GOP feature extractor possible.

---

## 6. Training Strategy

### Phase 1: CTC Phoneme Recognition (Base for GOP)

```python
# Fine-tune WavLM-large with CTC on LibriSpeech (on RunPod A100 80GB)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model = Wav2Vec2ForCTC.from_pretrained(
    "microsoft/wavlm-large",  # 317M params -- use large, we have the VRAM
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    attn_implementation="flash_attention_2",
)

# Training config for A100 80GB -- go big on batch size
training_args = TrainingArguments(
    bf16=True,  # bf16 preferred on A100 (better numerical stability than fp16)
    per_device_train_batch_size=32,  # large batch -- A100 has the memory
    gradient_accumulation_steps=2,   # effective batch 64
    learning_rate=3e-5,
    warmup_ratio=0.1,
    num_train_epochs=30,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=100,
    dataloader_num_workers=8,  # A100 machines usually have many CPU cores
)
```

**Time**: ~6-12h on A100 80GB for LibriSpeech 960h (faster than RTX 5070 due to larger batches + bf16)
**Cost**: ~$8-17 on RunPod ($1.39/hr A100 PCIe or $2.17/hr A100 SXM)
**Alternative**: Use pre-trained `facebook/wav2vec2-lv-60-espeak-cv-ft` (already CTC-trained on phonemes)

### Phase 2: GOP Feature Extraction

Two approaches:

**A. Self-Alignment GOP (GOP-SA)** -- Simpler, good performance
```text
1. Run audio through CTC model -> posteriors P(phone | frame)
2. Viterbi decode the canonical phone sequence -> alignment
3. For each aligned segment, compute:
   GOP-SA(p) = log P(target_phone | segment) - max_q log P(q | segment)
```

**B. Segmentation-Free GOP (GOP-SF)** -- Best performance, no alignment needed
```text
1. Run CTC forward-backward over full utterance
2. Compute p(target_phone | full_utterance, left_context, right_context)
3. Marginalize over all possible segmentations
4. Normalize by expected duration from forward algorithm
```

### Phase 3: Score Prediction (GOPT or Enhanced)

Train GOPT Transformer on extracted GOP features:
- Input: `[batch, 50, feat_dim]` GOP features + canonical phone embeddings
- Output: Multi-aspect scores at phoneme/word/utterance level
- Training: Minutes on any GPU, trivially small model

### Phase 4: Multimodal LLM (Primary High-Quality Track)

With A100 80GB, we can reproduce the paper setup exactly -- no quantization compromises.

```python
# Full bf16 LoRA fine-tuning Phi-4-multimodal on A100 80GB
from peft import LoraConfig, get_peft_model

# NO quantization needed -- A100 80GB fits the full model in bf16
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16,  # higher rank than paper (they used 8) -- we have the VRAM budget
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Can also experiment with unfreezing audio encoder layers
# (paper kept them frozen, but with A100 we can try unfreezing top layers)

# Training: AdamW, cosine schedule, lr=1e-4
# Batch size 2-4 (paper used 1), gradient accumulation 4-8
# 3 epochs on SpeechOcean762
```

**Time**: ~2-4h on A100 80GB with bf16 LoRA
**Cost**: ~$3-9 on RunPod
**Experiments to try** (that the paper couldn't on 24GB):
- LoRA rank 16 or 32 (vs paper's rank 8)
- Unfreeze top audio encoder layers
- Batch size 4 (vs paper's 1) for better gradient estimates
- Train on SpeechOcean762 + L2-ARCTIC combined
- Try Qwen-Audio-Chat as alternative base (Apache 2.0 license)

---

## 7. Data Strategy

### Primary Datasets

| Dataset | Use | Size | How to Get |
|---------|-----|------|-----------|
| **SpeechOcean762** | Pronunciation scoring training/eval | 5,000 utts, ~5h | [OpenSLR 101](https://www.openslr.org/101/) or [HuggingFace](https://huggingface.co/datasets/mispeech/speechocean762) |
| **L2-ARCTIC** | MDD training (phoneme-level error annotations) | 26,867 utts, ~24h | [TAMU PSI Lab](https://psi.engr.tamu.edu/l2-arctic-corpus/) |
| **LibriSpeech** | CTC phoneme recognizer pre-training | 960h | [OpenSLR 12](https://www.openslr.org/12/) |
| **CMU Kids** | Child speech evaluation | 9.1h | [CMU FestVox](http://www.festvox.org/cmu_arctic/) |

### Data Augmentation

1. **TTS-based mispronunciation generation**: Use neural TTS to synthesize audio with injected phoneme errors (substitution/deletion/insertion based on L1-specific confusion matrices)
2. **Pseudo-labeling**: Train initial model on labeled data, generate pseudo-labels on unlabeled L2 speech, retrain -- shown to improve MDD F1 by 2.48%
3. **SpeechBlender**: Blend native and non-native speech segments to create controlled mispronunciation examples

### How Much Data Is Enough?

- **CTC fine-tuning**: 5h (TIMIT) to 960h (LibriSpeech) depending on quality needs
- **Pronunciation scoring**: SpeechOcean762's ~2,500 train utterances is sufficient for GOPT-level results
- **MDD**: L2-ARCTIC's ~3,600 annotated utterances + augmentation is minimum viable
- **Multimodal LLM LoRA**: SpeechOcean762 alone (3 epochs) achieves PCC > 0.7

---

## 8. Implementation Phases

### Phase 1: Baseline (Days 1-3)

**Goal**: Get GOPT running with pre-extracted features, establish baseline metrics.

```bash
# Already have the repo cloned
cd /Users/chiejimofor/Documents/Github/gopt

# Download pre-extracted GOP features (skip Kaldi entirely)
# Dropbox link from README: https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1

# Install dependencies
uv venv && uv pip install -r requirements.txt

# Run training
cd src && ./run.sh
```

**Expected results**: PCC 0.612 phone, 0.549 word, 0.742 sentence on SpeechOcean762.

### Phase 2: Modern Base Model (Week 1-2)

**Goal**: Replace Kaldi with WavLM-large CTC for GOP extraction.

1. Fine-tune WavLM-large with CTC on LibriSpeech for phoneme recognition
   - Use HuggingFace `Wav2Vec2ForCTC` with WavLM-large weights
   - Train on RunPod A100 80GB, ~6-12h (~$8-17)
   - Full fine-tuning (no LoRA needed -- A100 has plenty of VRAM)
   - Also run data2vec2-large as comparison (shown to outperform for pronunciation)

2. Build GOP extraction pipeline
   - Replace Kaldi GOP scripts with PyTorch-based extraction
   - Implement Self-Alignment GOP (GOP-SA) from paper 2507.16838
   - Extract features in same format as GOPT expects: `[utterances, 50, feat_dim]`

3. Retrain GOPT on new features, compare with Kaldi baseline

### Phase 3: Segmentation-Free Enhancement (Week 3-4)

**Goal**: Implement GOP-SF for alignment-agnostic scoring.

1. Implement CTC forward-backward algorithm for GOP-SF
   - Compute numerator: CTC loss with canonical transcription
   - Compute denominator: CTC loss with SDI (substitution/deletion/insertion) graph
   - Extract expected duration for normalization

2. Generate GOP-SF feature vectors (LPP + LPR + expected duration)

3. Compare GOP-SA vs GOP-SF on SpeechOcean762 and CMU Kids

### Phase 4: MDD Integration (Week 5-6)

**Goal**: Add mispronunciation detection and diagnosis.

1. Fine-tune WavLM-large on L2-ARCTIC with MDD annotations (A100)
2. Implement Needleman-Wunsch alignment for comparing predicted vs. canonical phonemes
3. Add MDD head alongside pronunciation scoring
4. Generate diagnostic output: which phoneme was wrong, what was produced instead

### Phase 5: Multimodal LLM Track (Week 5-7, parallel with Phase 4)

**Goal**: Train the highest-quality single model.

1. LoRA fine-tune Phi-4-multimodal on SpeechOcean762 (A100 80GB, bf16, ~$6)
2. Experiment with higher LoRA ranks (16, 32) and unfreezing audio encoder
3. Train on combined SpeechOcean762 + L2-ARCTIC for joint APA + MDD
4. Compare with Track A pipeline -- if Track B wins, it becomes primary

### Phase 6: Production API (Week 8-9)

**Goal**: Deploy best model(s) as a production-ready API.

1. Export WavLM-large + GOPT pipeline to ONNX
2. Apply INT8 dynamic quantization (~3x speedup, ~4x size reduction)
3. Build FastAPI server with WebSocket support
4. Target <200ms end-to-end latency for Track A model
5. Serve Track B (Phi-4) on GPU endpoint for premium quality tier
6. Benchmark against Azure Speech API

### Phase 7: Push Beyond SOTA (Ongoing)

**Goal**: Exceed commercial API performance.

1. Train XLS-R-1B as CTC backbone for multilingual pronunciation (A100/H100)
2. Ensemble Track A + Track B scores for best-of-both
3. Distill ensemble into a single efficient model
4. Add prosody analysis, fluency scoring, content assessment
5. Multi-language support via XLS-R-1B or Qwen3-ASR

---

## 9. Deployment & API Design

### ONNX Optimization Pipeline

```python
# Export wav2vec2/WavLM to ONNX
import torch

dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
torch.onnx.export(
    model, dummy_input, "wavlm_ctc.onnx",
    input_names=["audio"],
    output_names=["logits"],
    dynamic_axes={"audio": {1: "audio_length"}, "logits": {1: "seq_length"}},
    opset_version=14,
)

# Quantize to INT8
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("wavlm_ctc.onnx", "wavlm_ctc_int8.onnx", weight_type=QuantType.QInt8)
```

### Expected Performance (Track A: WavLM-large based)

| Configuration | Model Size | Inference (CPU) | Inference (GPU) |
|--------------|-----------|-----------------|-----------------|
| PyTorch FP32 (WavLM-large) | ~1.3GB | ~500-800ms | ~40-70ms |
| PyTorch FP16 (WavLM-large) | ~650MB | ~300-500ms | ~25-45ms |
| ONNX FP32 | ~1.3GB | ~250-400ms | ~25-40ms |
| ONNX INT8 | ~350MB | ~80-150ms | ~15-30ms |
| GOPT head | ~2MB | ~1-5ms | ~1ms |
| **Total pipeline (ONNX int8)** | **~352MB** | **~85-160ms** | **~20-35ms** |

Note: For CPU deployment where size matters, WavLM-base+ ONNX int8 (~100MB total) is still excellent.

### API Endpoint Design

```text
POST /v1/assess
Content-Type: multipart/form-data

Fields:
  audio: <binary wav/mp3/webm>
  text: "The quick brown fox jumps over the lazy dog"
  language: "en-US" (optional, default en-US)
  detail_level: "full" | "sentence" | "word" (optional, default full)

Response:
{
  "utterance": {
    "accuracy": 0.82,
    "fluency": 0.91,
    "prosody": 0.76,
    "completeness": 0.95,
    "total": 0.85
  },
  "words": [
    {
      "word": "The",
      "accuracy": 0.95,
      "stress": 0.88,
      "total": 0.92,
      "phonemes": [
        {"phone": "DH", "score": 0.93, "expected": "DH", "actual": "DH", "error": null},
        {"phone": "AH", "score": 0.97, "expected": "AH", "actual": "AH", "error": null}
      ]
    },
    {
      "word": "quick",
      "accuracy": 0.65,
      "stress": 0.70,
      "total": 0.67,
      "phonemes": [
        {"phone": "K", "score": 0.90, "expected": "K", "actual": "K", "error": null},
        {"phone": "W", "score": 0.40, "expected": "W", "actual": "V", "error": "substitution"},
        {"phone": "IH", "score": 0.85, "expected": "IH", "actual": "IH", "error": null},
        {"phone": "K", "score": 0.88, "expected": "K", "actual": "K", "error": null}
      ]
    }
  ],
  "mispronunciations": [
    {
      "word": "quick",
      "phone_index": 1,
      "expected": "W",
      "actual": "V",
      "type": "substitution",
      "suggestion": "Round your lips more and use less teeth contact"
    }
  ]
}
```

### Deployment Options

| Target | Stack | Latency | Use Case |
|--------|-------|---------|----------|
| **Self-hosted API** | FastAPI + ONNX Runtime + uvicorn | <100ms GPU, <200ms CPU | Primary deployment |
| **Browser** | ONNX Runtime Web (WASM + WebGPU) | ~200-500ms | Client-side, no server |
| **Mobile** | ONNX Runtime Mobile / CoreML | ~150-300ms | iOS/Android apps |
| **Edge** | TensorRT on Jetson / ONNX on ARM | ~100-200ms | Embedded devices |

---

## 10. Differentiation Strategy

### How to Beat Closed APIs

| Feature | Azure | ELSA | Our System |
|---------|-------|------|------------|
| Open source | No | No | **Yes** |
| Self-hostable | No | No | **Yes** |
| Per-phoneme scoring | Yes | Yes | **Yes** |
| Mispronunciation diagnosis | Limited | Yes | **Yes + explanations** |
| Segmentation-free | No (forced align) | Unknown | **Yes (GOP-SF)** |
| Works offline | No | Partial | **Yes (ONNX)** |
| Privacy (data stays local) | No | No | **Yes** |
| Model size | N/A (cloud) | N/A | **~100MB (int8)** |
| Custom model training | No | No | **Yes** |
| Latency | ~300-500ms (network) | ~200ms | **<100ms (local GPU)** |

### Key Differentiators

1. **Open-source & self-hostable**: Schools, universities, and privacy-conscious apps can run it locally
2. **Segmentation-free scoring**: More robust than Azure's forced-alignment approach
3. **Tiny deployment footprint**: ~100MB ONNX int8 vs cloud-only competitors
4. **Customizable**: Users can fine-tune on their own data, add languages, specialize for demographics
5. **Diagnostic feedback**: Not just scores but actionable pronunciation tips

---

## 11. Compute Cost Estimates

### RunPod GPU Options

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| **A100 PCIe** | 40GB | ~$1.39/hr | WavLM-large, data2vec2-large fine-tuning |
| **A100 SXM** | 80GB | ~$2.17/hr | Phi-4-multimodal LoRA, XLS-R-1B, any experiment |
| **H100 SXM** | 80GB | ~$3.49/hr | Fastest training, large batch sizes |
| **A6000** | 48GB | ~$0.79/hr | Budget option for medium models |

### Training Cost Breakdown

| Task | GPU | Time | Cost |
|------|-----|------|------|
| GOPT training (pre-extracted features) | Any (or local CPU) | ~5 min | ~$0 |
| WavLM-large CTC fine-tune (LibriSpeech 960h) | A100 80GB | ~6-12h | **~$13-26** |
| data2vec2-large CTC fine-tune (LibriSpeech) | A100 80GB | ~6-12h | ~$13-26 |
| XLS-R-1B CTC fine-tune (LibriSpeech) | A100 80GB | ~12-20h | ~$26-43 |
| GOP-SA/SF feature extraction (SpeechOcean762) | A100 40GB | ~30min-1h | ~$1-2 |
| Phi-4-multimodal bf16 LoRA (SpeechOcean762) | A100 80GB | ~2-4h | **~$4-9** |
| Phi-4-multimodal bf16 LoRA (SO762 + L2-ARCTIC) | A100 80GB | ~4-8h | ~$9-17 |
| Full pipeline (all tracks, all experiments) | A100 80GB | ~2-3 days | **~$100-150** |
| ONNX export + quantization | Local RTX 5070 | ~10 min | $0 |

### Local RTX 5070 12GB (for prototyping & serving)

| Task | Time | Feasible? |
|------|------|-----------|
| GOPT training (pre-extracted features) | ~5 minutes | Yes |
| WavLM-base+ quick experiments (TIMIT 5h) | ~1-2h | Yes |
| GOP feature extraction (small datasets) | ~30min-1h | Yes |
| ONNX inference serving (Track A) | Real-time | Yes -- primary deployment target |
| Phi-4 inference serving (Track B) | Real-time | Needs 4-bit quantization |

**Strategy**: Train on RunPod, deploy locally. Total estimated training budget: **~$50-150** for a complete system.

---

## 12. Success Metrics

### Target Performance (SpeechOcean762)

With unconstrained GPU, targets are more aggressive -- large models + full fine-tuning + multimodal LLM track.

| Metric | GOPT Baseline | Track A Target | Track B Target | Azure (est.) |
|--------|--------------|---------------|---------------|-------------|
| Phone PCC | 0.612 | **>0.70** | **>0.75** | ~0.65-0.70 |
| Word PCC | 0.549 | **>0.62** | **>0.68** | ~0.55-0.60 |
| Sentence PCC | 0.742 | **>0.80** | **>0.83** | ~0.75-0.80 |
| MDD AUC | N/A | **>0.92** | **>0.94** | ~0.85-0.90 |
| Inference latency | N/A | **<200ms** (ONNX) | **<1s** (GPU) | ~300-500ms |
| Model size (deploy) | ~380MB | **<200MB** (int8) | **~3GB** (4-bit) | Cloud-only |

### Milestones

- [ ] Phase 1: GOPT baseline running (Day 3)
- [ ] Phase 2: WavLM-large CTC + GOP-SA pipeline on RunPod (Week 2)
- [ ] Phase 3: GOP-SF implementation (Week 4)
- [ ] Phase 4: MDD integration with L2-ARCTIC (Week 6)
- [ ] Phase 5: Phi-4-multimodal bf16 LoRA trained (Week 7)
- [ ] Phase 6: ONNX API deployed, benchmarked vs Azure (Week 9)
- [ ] Phase 6: Beat GOPT baseline by >15% Phone PCC (Week 9)
- [ ] Phase 7: XLS-R-1B / ensemble experiments (Ongoing)
- [ ] Phase 7: Multi-language support (Ongoing)

---

## 13. Key Repos & Resources

### Code Repositories

| Repo | Purpose | Link |
|------|---------|------|
| GOPT | Transformer pronunciation scorer | [github.com/YuanGongND/gopt](https://github.com/YuanGongND/gopt) |
| wav2vec2mdd | End-to-end MDD | [github.com/vocaliodmiku/wav2vec2mdd](https://github.com/vocaliodmiku/wav2vec2mdd) |
| joint-apa-mdd-mtl | Joint APA+MDD | [github.com/rhss10/joint-apa-mdd-mtl](https://github.com/rhss10/joint-apa-mdd-mtl) |
| IPA-Wav2Vec2 | Phoneme recognition | [github.com/Srinath-N-R/IPA-Wav2Vec2-Phoneme-Recognition](https://github.com/Srinath-N-R/IPA-Wav2Vec2-Phoneme-Recognition/) |
| Multilingual-PR | wav2vec2 vs HuBERT vs WavLM | [github.com/ASR-project/Multilingual-PR](https://github.com/ASR-project/Multilingual-PR) |
| wav2vec2-service | ONNX export example | [github.com/ccoreilly/wav2vec2-service](https://github.com/ccoreilly/wav2vec2-service) |
| Kaldi GOP recipe | SpeechOcean762 baseline | [github.com/kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop_speechocean762) |
| fairseq | wav2vec2 pretraining + CTC fine-tuning | [github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) |
| Omnilingual-ASR | Multilingual CTC/LLM recipes | [github.com/facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) |
| Allosaurus | Universal phone recognition | [github.com/xinjli/allosaurus](https://github.com/xinjli/allosaurus) |
| Qwen3-ASR | ASR + forced aligner | [github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) |

### Training-Specific Guides

- Kaldi GOP run pipeline: <https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/gop_speechocean762/s5/run.sh>
- fairseq wav2vec2 training/fine-tuning: <https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md>
- fairseq S2T (seq-to-seq): <https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/README.md>
- Omnilingual data prep: <https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/dataprep/README.md>
- Omnilingual training recipes: <https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/README.md>
- Qwen3-ASR fine-tuning: <https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md>
- NeMo CTC training script: <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_ctc/speech_to_text_ctc.py>
- Montreal Forced Aligner user guide: <https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/index.html>

### Model Cards

- NVIDIA Parakeet CTC 0.6B: <https://huggingface.co/nvidia/parakeet-ctc-0.6b>
- NVIDIA Parakeet CTC 1.1B: <https://huggingface.co/nvidia/parakeet-ctc-1.1b>

### Key Papers (Priority Order)

1. **2507.16838** -- Segmentation-Free GOP
2. **2509.02915** -- LoRA Fine-tuned Speech Multimodal LLM
3. **2506.02080** -- Enhancing GOP in CTC-Based MDD
4. **2508.03937** -- LCS-CTC Soft Alignments
5. **2310.13974** -- Pronunciation Assessment Review
6. **2203.15937** -- Improving MDD with wav2vec2
7. **2507.14346** -- Phonetic Error Detection
8. **2206.07289** -- Text-Aware End-to-End MDD
9. **2509.03256** -- Comparison of E2E Speech Assessment Models
10. **2104.01378** -- SpeechOcean762 Dataset Paper
11. **2511.09690** -- Omnilingual-ASR
12. **2601.21337** -- Qwen3-ASR

### Tutorials

- [HuggingFace: Fine-Tune wav2vec2 for English ASR](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [HuggingFace: Fine-Tune XLSR-Wav2Vec2](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)
- [OpenVINO wav2vec2 Quantization](https://docs.openvino.ai/2024/notebooks/speech-recognition-quantization-wav2vec2-with-output.html)
- [ONNX Runtime Quantization Guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

### Datasets

- [SpeechOcean762 (OpenSLR)](https://www.openslr.org/101/) -- Primary benchmark
- [SpeechOcean762 (HuggingFace)](https://huggingface.co/datasets/mispeech/speechocean762)
- [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) -- MDD annotations
- [LibriSpeech](https://www.openslr.org/12/) -- CTC pre-training
- [Common Voice](https://commonvoice.mozilla.org/) -- Multilingual

---

*Last updated: 2026-02-22*
