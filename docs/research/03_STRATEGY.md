# Strategy: Building an Open-Source Pronunciation Assessment System

Comprehensive strategy covering competitive landscape, methods, models, and architecture decisions.

---

## 1. Competitive Landscape

SpeechAce's actual API documentation reveals **6 supported dialects** (en-us, en-gb, fr-fr, fr-ca, es-es, es-mx) -- far fewer than the "~30 languages" sometimes marketed. It offers 12 API endpoints spanning scripted and spontaneous speech assessment, with phoneme/syllable/word/sentence scoring on a 0-100 scale, stress and intonation analysis, fluency metrics (WCPM, pause analysis), and CEFR/IELTS/TOEFL/PTE score mapping. The claimed **0.9 Pearson correlation** with human judges is self-reported.

| Capability | SpeechAce | Open-source achievable | Gap |
|---|---|---|---|
| Phoneme scoring (0-100) | 0.9 PCC claimed | ~0.62 PCC (GOPT on speechocean762) | Moderate -- needs more training data |
| Word-level scoring | Yes | ~0.55 PCC | Similar approach, less data |
| Sentence-level scoring | Yes | ~0.74 PCC | Reasonable baseline |
| Stress detection | Per-phoneme/syllable | F0+intensity heuristics | Hard -- no production-ready model |
| "Sound most like" | Yes | GOP feature vectors reveal this | Directly from 41-dim GOP-SF features |
| Fluency (WCPM, pauses) | Yes | Parselmouth + VAD | Achievable with engineering |
| IELTS/CEFR mapping | Calibrated | Requires thousands of calibration samples | Hardest to replicate |
| Spontaneous speech | Grammar, vocab, coherence | Whisper/Qwen3 ASR + LLM scoring | Feasible but uncalibrated |
| Multi-language | 6 dialects | Potentially 60+ via wav2vec2-xlsr | More languages, less depth |

### Gap Analysis

The **hardest capabilities to replicate** are calibrated standardized test scores (IELTS/CEFR/TOEFL mapping requires years of validation against real exam data) and the claimed 0.9 PCC accuracy (the gap from 0.62 to 0.9 likely reflects SpeechAce's access to proprietary training data far larger than speechocean762's 5,000 utterances).

The core takeaway across all analysis: the bottleneck for this technology is no longer the model architectures or the algorithms -- it is the annotated non-native speech data. The path to building this is highly viable today.

---

## 2. Methods Breakdown

Historically, pronunciation assessment relied on rigid pipelines. Modern research has shifted towards end-to-end and segmentation-free approaches.

### Method A: ASR Transcription -> Text Comparison (e.g., Whisper)

*How it works*: Use a modern ASR model to transcribe the speech, run G2P (Grapheme-to-Phoneme) on both the output and the expected text, and compare the edit distance.

**Why Not**: This is fundamentally flawed. Models like Whisper are explicitly trained to be *robust to accents* and heavily rely on language models to "fix" phonetics. If a learner mispronounces a word but the context makes it obvious, Whisper transcribes it perfectly, meaning you cannot detect the mispronunciation.

### Method B: Traditional Forced Alignment (HMM-GMM or DNN) + GOP

*How it works*: Force-align the audio to the expected phoneme sequence to find boundaries (start/end times), then calculate the Goodness of Pronunciation (GOP) for each bounded segment.

**Why Not**: Forced alignment is extremely fragile on L2 (learner) speech. If the learner inserts a massive pause, stutters, or completely mangles a sound, the boundaries shift wildly. If the boundaries are wrong, the GOP score is meaningless.

### Method C: Segmentation-Free GOP (GOP-SF / arXiv:2507.16838)

*How it works*: Uses a CTC-trained acoustic model to extract frame-level phoneme probabilities. Instead of finding exact boundaries for a phoneme, it uses the CTC Forward Algorithm to marginalize (sum) the probabilities over *all possible segmentations*. It elegantly handles substitutions and deletions without committing to a single alignment.

**Why Yes (The Winner)**: This is the breakthrough. It bypasses the fragile alignment step completely. It works perfectly with modern end-to-end Self-Supervised Learning (SSL) models. It produces a rich 41-dimensional feature vector per phoneme that tells you exactly *which* alternative phoneme the learner sounded like.

### Method D: Multi-Granularity Assessment (GOPT Transformer)

*How it works*: Takes the GOP-SF feature vectors and feeds them into a Transformer (like GOPT) trained on annotated data (like `speechocean762`) to output scores at the phoneme, word, and sentence levels simultaneously.

**Why Yes**: Provides the 0-100 or 0-10 scores that users expect, effectively mapping abstract acoustic probabilities into human-readable proficiency metrics.

---

## 3. Architecture Approaches

### A. Classical Pipeline: GOP-SF + GOPT (Proven)

This is the gold standard for interpretable, classical deep learning pipelines. It combines the methods described above into a four-layer stack.

**The Four-Layer Architecture:**

**Layer 1: Phoneme Recognition**

Primary model: `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (300M params, Apache 2.0) for immediate deployment. This model provides frame-level IPA phoneme posteriors across 60+ languages, which is exactly what GOP scoring requires.

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft", do_phonemize=False)

input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
with torch.no_grad():
    logits = model(input_values).logits  # [1, T, num_phonemes]
log_posteriors = torch.nn.functional.log_softmax(logits, dim=-1)
```

For higher accuracy, consider upgrading to w2v-BERT 2.0 or ZIPA-CR -- see Section 4 for full model comparison.

**Layer 2: GOP Scoring with the Segmentation-Free Method**

Implement the GOP-SF-SD pipeline from `github.com/frank613/CTC-based-GOP`:

1. Fine-tune wav2vec2-xlsr-53 on native speech with CTC loss for phoneme recognition (freeze feature extractor, train CTC head on LibriSpeech train-clean-100 for English, Common Voice for other languages)
2. Compute GOP-SF-SD feature vectors (41 dimensions per phoneme) using the CTC forward algorithm -- this marginalizes over all possible segmentations and handles substitutions and deletions
3. Train GOPT Transformer (`github.com/YuanGongND/gopt`) on speechocean762 to map 41-dim features to multi-granularity scores:
   - Phoneme accuracy (0-2)
   - Word accuracy, stress, total (0-10)
   - Sentence accuracy, completeness, fluency, prosody, total (0-10)

Key formula for GOP-SF-SD:
```text
GOP-SF(l_i) = log(p_CTC(L_cano | O) / p_CTC(L_alt(l_i) | O))
```

where the denominator marginalizes over all possible substitutions and deletions at position i, computed efficiently via CTC forward algorithm.

For **logit-based GOP variants** (arXiv 2506.12067), which show stronger alignment with human perception:
```css
GOP_margin(p) = (1/d) * sum_t [logit_p(o_t) - max_{q!=p} logit_q(o_t)]
```

Recent work on **phonological-knowledge-enhanced GOP** (arXiv 2506.02080) restricts substitution candidates to phonologically similar phonemes, reducing computation and improving discrimination.

**Layer 3: Fluency and Prosody Assessment**

Build a parallel prosody analysis pipeline using **Parselmouth** (Praat-Python bridge) for acoustic feature extraction:

- **Speaking rate**: Count syllable nuclei via intensity peak detection, divide by phonation time. Native English: 3.3-5.9 syllables/sec; L2 speakers typically 2.5-4.5.
- **Pause analysis**: Use **Silero VAD** or **pyannote-audio** for voice activity detection. Compute pause count, mean duration, ratio of pause-to-speech time, count of long pauses (>400ms).
- **Stress detection**: Extract F0, intensity, and duration per syllable from forced-aligned segments. Compare against canonical stress patterns from CMU Pronouncing Dictionary (134K words with stress notation). Stressed syllables show higher F0, greater intensity, and longer duration.
- **Intonation**: Extract F0 contours via Parselmouth, compute slope (rising/falling), range, and variability. Compare to native patterns using DTW.
- **Rhythm metrics**: From phone-level alignment, compute nPVI-V (normalized Pairwise Variability Index for vowels), %V (proportion of vocalic intervals), VarcoV/VarcoC (rate-normalized duration variability). These distinguish stress-timed (English) from syllable-timed (French, Spanish) and mora-timed (Japanese) languages.

```python
import parselmouth
sound = parselmouth.Sound("learner_audio.wav")
pitch = sound.to_pitch()
intensity = sound.to_intensity()
f0_contour = pitch.selected_array['frequency']
```

**Layer 4: Feedback Generation with LLMs**

Implement a structured feedback pipeline:

1. **Error detection**: From GOP-SF scoring, identify phonemes with scores below threshold. The 41-dim feature vector reveals exactly which alternative phoneme the learner's pronunciation most resembles (the "sound most like" feature SpeechAce offers).
2. **L1-aware diagnosis**: Use PHOIBLE (`phoible.org`, 3,020 inventories for 2,186 languages) to identify which phonemes the learner's L1 lacks. Cross-reference with known L1 transfer patterns (e.g., Mandarin->English: /r/-/l/ confusion, no final clusters, /th/->s/).
3. **LLM feedback generation**: Pass structured error data to an LLM with a prompt template:

```yaml
Target word: "think" /thINk/
Detected pronunciation: /sINk/
Error: /th/ -> /s/ substitution (GOP score: 0.3/2.0)
Learner L1: Mandarin Chinese
Known L1 pattern: Mandarin lacks dental fricatives

Generate: (1) What went wrong, (2) Articulatory instruction for /th/,
(3) Minimal pair practice words, (4) Encouragement
```

Use **PanPhon** (`pip install panphon`) to map IPA symbols to articulatory feature vectors for generating precise tongue/lip placement instructions.

### B. Multimodal LLM Path (Bleeding Edge)

*(Based on arXiv:2509.02915 and current 2025/2026 SOTA)*

Instead of stringing together an ASR model, mathematical GOP formulas, and a separate scoring Transformer, you use a single Multimodal LLM that natively understands audio.

**How it works**: Take an open-source MLLM (like Microsoft's **Phi-4-multimodal-instruct** or Alibaba's **Qwen2-Audio**). Pass the raw audio and a text prompt: *"Rate the pronunciation of the audio compared to the text 'The city was...' and provide phoneme transcripts."*

**The Breakthrough**: By applying Low-Rank Adaptation (LoRA) fine-tuning to the speech adapter layers using a dataset like `speechocean762`, the model learns to output structured JSON containing:
1. The verbatim phoneme transcript of what the user *actually* said.
2. 0-100 scores for accuracy, fluency, and prosody.
3. Natural language feedback on articulatory errors.

**Pros**:
- Minimum engineering friction -- no complex forced aligners, CTC loss functions, or multi-stage inference pipelines
- Requires minimal data to fine-tune because the LLM already possesses vast reasoning and phonetic knowledge
- Assessment, detection, and feedback all in a single API call

**Cons**:
- Inference is heavier (requires an LLM-sized GPU like an A100/H100) compared to a tiny 300M parameter acoustic model
- Less interpretable -- the model can hallucinate scores
- Newer approach with less published validation

### C. Training-Free Retrieval (Experimental)

*(Based on arXiv:2511.20107)*

A fascinating approach that completely bypasses the need to fine-tune phoneme recognition models.

**How it works**: Take a massive, pre-trained SSL model (like **HuBERT Large**). Pass thousands of hours of perfectly pronounced speech through it and save the frame-level embeddings to a vector database, tagged by their phoneme.

When a learner speaks, extract the embeddings of their audio frame-by-frame. Do a simple K-Nearest Neighbors (KNN/Cosine Similarity) search against the "perfect" vector database. If the learner's embedding for the letter "T" matches closest to the vector for "S" in the database, flag a mispronunciation.

**Pros**:
- Zero training required -- just need a vector database (like Faiss or Milvus) and off-the-shelf HuBERT
- Simple conceptual architecture
- Easy to update by adding more reference embeddings

**Cons**:
- Slower at runtime due to dense vector lookups
- Slightly lower F1 score (69.60%) compared to heavily fine-tuned models
- No direct scoring output -- produces detection only, not graded scores

### D. Recommended Decision

The choice of architecture depends on constraints and goals. Here is a unified decision framework.

**If you have GPU budget and want maximum quality fastest:**

Use the **MLLM Oracle** path (Phase 1 -> Phase 2 distillation):

1. **Phase 1**: Spin up an A100 on RunPod. Download Phi-4-multimodal-instruct (or Qwen2-Audio). LoRA fine-tune on speechocean762 and L2-ARCTIC. Result: a state-of-the-art system doing assessment, detection, and feedback in a single API call within ~2 weeks.
2. **Phase 2** (optional): Use the MLLM as an oracle to auto-score thousands of hours of cheap, unlabeled L2 speech (from YouTube, podcasts, etc.). Use this auto-labeled dataset to train a tiny WavLM + Conformer model using PA-AF GOP. Result: a lightweight, blazing-fast model that achieves the accuracy of the heavy MLLM.

**If you want interpretability and low inference cost:**

Choose one of three classical paths:

| Path | Stack | Tradeoff |
|------|-------|----------|
| **Easiest MVP** | wav2vec2-xlsr-espeak + GOP-SF + GOPT | Zero training for acoustic model. Working in weeks. ~0.62 PCC. |
| **Best Multilingual** | w2v-BERT 2.0 (fine-tuned) + GOP-SF + LLM Feedback | Requires ~4-8 hrs fine-tuning. Best acoustic representations (4.5M hrs pretraining). |
| **Data-Augmented English** | w2v-BERT 2.0 + SpeechBlender augmentation + GOPT | Closes the 0.62-vs-0.9 PCC gap by generating synthetic L2 data from native speech. |

**Summary of the shift**: Initially, wav2vec2-xlsr + GOP-SF seemed like the most pragmatic path. After exhaustive review of late 2025 papers (specifically September-November 2025), Multimodal LLMs with LoRA tuning have rapidly overtaken traditional cascaded architectures in both ease-of-implementation and holistic scoring capability. The traditional methods (GOP-SF) are now primarily focused on computational optimizations (like Phoneme Confusion Maps) for edge deployments, rather than being the absolute highest-accuracy ceiling. The recommended strategy is: **start with the MLLM path for the quality ceiling, then distill into a classical pipeline for production scale**.

---

## 4. Model Selection Guide

**Goal**: Pick a CTC model that outputs frame-level phoneme posteriors for GOP computation.
**Constraint**: GOP requires a CTC-trained model that outputs **phonemes** (not BPE), with good temporal resolution.

### Option 1: wav2vec2-xlsr-53-espeak-cv-ft (Just Works)

> Works out of the box -- 387 IPA phoneme posteriors, zero adaptation. See also Option 6 (ZIPA) for an alternative.

| Property | Value |
|----------|-------|
| HuggingFace | [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) |
| Architecture | wav2vec2 Transformer (24 layers, d=1024) |
| Parameters | ~300M |
| Pretraining | 56K hours, 53 languages (XLSR-53 self-supervised) |
| Fine-tuning | CTC on CommonVoice with eSpeak phoneme labels |
| Output vocabulary | **387 IPA phoneme labels** |
| Temporal resolution | **20ms per frame** (best of all options) |
| License | **Apache 2.0** |
| Framework | HuggingFace Transformers |
| Monthly downloads | 371K+ |

**Why yes**:
- Zero adaptation needed -- load model, get phoneme posteriors, compute GOP
- 387 IPA labels covers every language's phonemes
- 20ms temporal resolution is the best available (4x better than Parakeet's 80ms)
- Proven in the GOP literature -- this is what the SOTA papers use
- Apache 2.0 -- fully open, commercial use OK
- HuggingFace native -- trivial to integrate

**Why not**:
- Encoder is from 2021 -- only 56K hours of pretraining (77x less than omnilingual-asr)
- 300M params, Transformer-only (no convolution modules like Conformer)
- WER ~3-4% vs modern models at 1.8%
- But: **WER does not matter for GOP**. What matters is phoneme posterior quality, and this model was specifically fine-tuned for that

**Code (ready to run)**:
```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

# audio = your 16kHz waveform tensor
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits  # [batch, time, 387]

posteriors = torch.softmax(logits, dim=-1)
# Done. 387 IPA phoneme posteriors at 20ms resolution.
# Feed these directly into GOP-SF computation.
```

**Verdict: START HERE.** This is the fastest path to a working GOP system. Use this to validate your pipeline end-to-end, then upgrade the encoder later if needed.

---

### Option 2: w2v-BERT 2.0 + Phoneme CTC Head (Best Multilingual Encoder)

> 4.5M hours of pretraining, 143 languages. The strongest SSL encoder available, but needs a phoneme head.

| Property | Value |
|----------|-------|
| HuggingFace | [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) |
| Fine-tuning guide | [huggingface.co/blog/fine-tune-w2v2-bert](https://huggingface.co/blog/fine-tune-w2v2-bert) |
| Architecture | Conformer encoder (24 layers, d=1024) with masked prediction + contrastive SSL |
| Parameters | ~600M |
| Pretraining | **4,500,000 hours**, 143 languages (from SeamlessM4T v2) |
| Output vocabulary | **None** -- this is a pretrained encoder, no CTC head yet |
| Temporal resolution | 20ms per frame |
| License | **MIT** (most permissive) |
| Framework | HuggingFace Transformers (`Wav2Vec2BertForCTC`) |

**Why yes**:
- 80x more pretraining data than XLSR-53 (4.5M vs 56K hours)
- Conformer architecture (convolutions + attention) -- better than pure Transformer
- 143 languages -- works for any language you want
- MIT license -- maximally permissive
- HuggingFace native with `Wav2Vec2BertForCTC` -- well-documented fine-tuning
- The [fine-tuning blog post](https://huggingface.co/blog/fine-tune-w2v2-bert) walks through the exact process of adding a CTC head
- 20ms temporal resolution (same as XLSR-53)
- This is what Meta's SeamlessM4T uses internally as its speech encoder

**Why not**:
- Requires fine-tuning -- you must add a phoneme CTC head and train it
- Fine-tuning takes ~4-8 hours on A100 80GB (LibriSpeech 960h)
- Nobody has published GOP results with w2v-BERT 2.0 yet (you would be first)
- Slightly more complex setup than Option 1

**Adaptation required**:
1. Add a phoneme CTC head (randomly initialized linear layer)
2. Fine-tune on phoneme-labeled data (LibriSpeech + eSpeak/CMU Dict for labels)
3. ~4-8 hours on A100 80GB

**Code (from the HuggingFace blog)**:
```python
from transformers import (
    Wav2Vec2BertForCTC,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2BertProcessor,
    TrainingArguments,
    Trainer,
)

# 1. Create a phoneme tokenizer (IPA or ARPABET vocabulary)
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",  # {"<pad>": 0, "<s>": 1, "</s>": 2, "AA": 3, "AE": 4, ...}
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 2. Load w2v-BERT 2.0 with a new CTC head
model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/w2v-bert-2.0",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    add_adapter=True,          # adapter layers between encoder and CTC head
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# 3. Freeze feature extractor, train CTC head + adapter + top encoder layers
model.freeze_feature_encoder()

# 4. Train
training_args = TrainingArguments(
    output_dir="./w2v-bert-phoneme-ctc",
    per_device_train_batch_size=16,   # A100 80GB can handle this
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    warmup_steps=500,
    num_train_epochs=10,
    bf16=True,
    evaluation_strategy="steps",
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,   # LibriSpeech with phoneme labels
    eval_dataset=eval_dataset,
    tokenizer=processor,
)
trainer.train()
```

**Verdict: BEST UPGRADE PATH.** After validating your pipeline with Option 1, this is the strongest encoder to upgrade to. Same 20ms temporal resolution as XLSR, but 80x more pretraining data.

---

### Option 3: Parakeet-CTC-0.6B + Phoneme Head (Best English Encoder)

> Best English ASR model. 1.87% WER on LibriSpeech. But 80ms temporal resolution is a tradeoff.

| Property | Value |
|----------|-------|
| HuggingFace | [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) |
| NeMo docs | [NVIDIA NeMo ASR](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html) |
| Architecture | FastConformer (24 layers, d=1024, depthwise-separable convolutions) |
| Parameters | 600M |
| Pretraining/Training | **64,000 hours** (40K proprietary NeMo + 24K public) |
| Output vocabulary | **1,024 BPE tokens** (NOT phonemes) |
| Temporal resolution | **80ms per frame** (8x downsampling) |
| WER (LibriSpeech clean/other) | **1.87% / 3.76%** |
| License | **CC-BY-4.0** |
| Framework | HuggingFace Transformers + NVIDIA NeMo |

**Why yes**:
- Best English speech encoder -- 1.87% WER, state-of-the-art
- FastConformer architecture is newer and more efficient than wav2vec2's Transformer
- 600M params with very rich acoustic representations
- CC-BY-4.0 -- commercial use OK
- Available in both HuggingFace and NeMo (NeMo has `change_vocabulary()` for easy head replacement)
- Strong candidate for English-only pronunciation assessment

**Why not**:
- **80ms temporal resolution** -- 4x worse than wav2vec2's 20ms. This matters for GOP:
  - A typical phoneme lasts 50-120ms
  - At 80ms/frame, a short phoneme might only get 1-2 frames
  - Less precision for alignment and posterior estimation
  - GOP-SF mitigates this (considers all alignments), but still fewer data points
- Requires adaptation -- must replace 1024 BPE head with phoneme head and fine-tune
- English-only (the 64K hours are all English)
- The 40K proprietary training hours mean you cannot fully reproduce the training
- NeMo framework adds dependency complexity

**Temporal resolution comparison**:
```yaml
wav2vec2 / w2v-BERT:  20ms/frame -> 50 frames per second -> ~5 frames per phoneme
Parakeet:             80ms/frame -> 12.5 frames per second -> ~1-2 frames per phoneme
                                                              ^^^^^^^^^^^^^^^^^^^
                                                              This is the concern
```

**Adaptation required**:
1. Replace 1024-BPE CTC head with phoneme CTC head
2. Fine-tune on phoneme-labeled data
3. ~2-4 hours on A100 80GB

**Code (NeMo -- easiest for head replacement)**:
```python
import nemo.collections.asr as nemo_asr

# Load pretrained Parakeet (keeps the powerful encoder)
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")

# Replace CTC head with phoneme vocabulary
# You need a phoneme tokenizer directory (ARPABET labels)
model.change_vocabulary(
    new_tokenizer_dir="/path/to/phoneme_tokenizer",
    new_tokenizer_type="char"  # character-level for phonemes
)
# Encoder weights PRESERVED, only CTC head reinitialized

# Fine-tune on LibriSpeech with phoneme labels
# trainer.fit(model, train_dataloader, val_dataloader)
```

**Code (HuggingFace -- if you prefer)**:
```python
from transformers import AutoModelForCTC, AutoProcessor
import torch.nn as nn

model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-0.6b")

# Replace the CTC head
num_phonemes = 40  # 39 ARPABET + blank
model.lm_head = nn.Conv1d(
    in_channels=1024,       # FastConformer hidden size
    out_channels=num_phonemes,
    kernel_size=1
)
nn.init.xavier_uniform_(model.lm_head.weight)

# Fine-tune with CTC loss on phoneme-labeled data
```

**Verdict: BEST FOR ENGLISH.** If you are building an English-only system and can accept 80ms resolution, this encoder produces richer representations than wav2vec2. The 80ms tradeoff is real but GOP-SF helps mitigate it.

---

### Option 4: Omnilingual ASR CTC (Already Has a CTC Head)

> 4.3M hours, 1600+ languages, Apache 2.0. Already CTC-trained with a character head -- just swap it for phonemes. The catch: fairseq2 framework.

| Property | Value |
|----------|-------|
| GitHub | [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) |
| Paper | [Omnilingual ASR (arXiv:2511.09690)](https://arxiv.org/abs/2511.09690) |
| Architecture | wav2vec2 Transformer (CNN frontend + Transformer encoder + linear CTC head) |
| Pretraining | **4,300,000 hours**, 1,600+ languages |
| Output vocabulary | **9,812 characters** (union of all writing systems -- NOT BPE, NOT phonemes) |
| Temporal resolution | **20ms per frame** |
| License | **Apache 2.0** (model + code), CC-BY 4.0 (corpus) |
| Framework | **fairseq2** (NOT HuggingFace Transformers) |
| Install | `pip install omnilingual-asr` |

**Available sizes**:

| Model | Params | Size | Encoder dim | HuggingFace ID |
|-------|--------|------|-------------|----------------|
| CTC 300M | 325M | 1.3 GB | 1,024 | [facebook/omniASR-CTC-300M](https://huggingface.co/facebook/omniASR-CTC-300M) |
| **CTC 1B** | **975M** | **3.7 GB** | **1,280** | [**facebook/omniASR-CTC-1B**](https://huggingface.co/facebook/omniASR-CTC-1B) |
| CTC 3B | 3.1B | 12 GB | 2,048 | `facebook/omniASR-CTC-3B` |
| CTC 7B | 6.5B | 25 GB | 2,048 | [facebook/omniASR-CTC-7B](https://huggingface.co/facebook/omniASR-CTC-7B) |

v2 variants also exist (`omniASR_CTC_1B_v2` etc.) with slightly larger vocab (10,288 chars) and improved accuracy.

**Why yes**:
- Already CTC-trained -- unlike the W2V variants (Option 5), these have a working CTC head
- The head replacement is architecturally trivial -- it is one linear layer (`final_proj`):
  ```text
  (batch, time, 1280) -> Linear -> (batch, time, 9812)   <- current
  (batch, time, 1280) -> Linear -> (batch, time, 70)      <- after swap
  ```
- 4.3M hours of pretraining -- 77x more than XLSR-53, comparable to w2v-BERT 2.0
- 1,600+ languages -- best multilingual coverage of any option
- 20ms temporal resolution -- same as wav2vec2 and w2v-BERT 2.0
- Apache 2.0 -- fully permissive
- Range of sizes: 300M for quick experiments, 1B for quality, 3B/7B if you want to push limits
- The 300M model (325M params) is a potential drop-in XLSR-53 replacement with 77x more pretraining
- ONNX exports already exist via [sherpa-onnx](https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12) (300M and 1B)

**Why not**:
- **fairseq2 framework** -- this is the main friction:
  - Models are on HuggingFace Hub but in fairseq2 format, not Transformers format
  - Cannot do `Wav2Vec2ForCTC.from_pretrained("facebook/omniASR-CTC-1B")`
  - Need `pip install omnilingual-asr` which pulls in fairseq2 with its own config system, asset manager, CUDA extensions
  - No HuggingFace Trainer integration -- must use fairseq2's training loop or write a conversion
  - Nobody has written a fairseq2-to-HuggingFace weight conversion script for these models
- Less community tooling and documentation than HuggingFace ecosystem
- Nobody has tested these encoders for pronunciation/GOP yet

**How it compares to the Meta lineage**:
```yaml
2020: XLSR-53         ->   56K hrs,   53 langs, Apache 2.0, HuggingFace  <- Option 1
2023: MMS-1B          ->  491K hrs, 1406 langs, CC-BY-NC (!), HuggingFace
2025: omniASR CTC 1B  -> 4.3M hrs, 1600 langs, Apache 2.0, fairseq2    <- Option 4
```

MMS-1B (`facebook/mms-1b-all`) is the middle child -- more data than XLSR but CC-BY-NC 4.0 (non-commercial). Omnilingual-asr replaced it with 9x more data and Apache 2.0. MMS-1B is effectively obsolete.

**Three paths to use these**:

**Path A: Work directly in fairseq2** (fastest to start, stays in fairseq2)
```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch.nn as nn

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_1B")
model = pipeline.model  # Wav2Vec2AsrModel instance

# The CTC head is model.final_proj
# Replace it:
model.final_proj = nn.Linear(1280, num_phonemes)  # 1280 = encoder dim for 1B
nn.init.xavier_uniform_(model.final_proj.weight)

# Fine-tune with CTC loss on phoneme-labeled data
# (requires fairseq2's training setup)
```

**Path B: Export encoder to ONNX, attach head in PyTorch** (avoids fairseq2 at inference)
```python
# Use the sherpa-onnx export script as a starting point:
# https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/omnilingual-asr
# Modify to export encoder-only, then attach a PyTorch phoneme head
```

**Path C: Write a weight conversion script** (one-time effort, then use HuggingFace forever)
```python
# The architectures are structurally compatible:
# fairseq2 Wav2Vec2AsrModel ~ HuggingFace Wav2Vec2ForCTC
# Both: CNN frontend -> Transformer encoder -> Linear CTC head
#
# Steps:
# 1. Load fairseq2 checkpoint
# 2. Map parameter names (fairseq2 naming -> HF naming)
# 3. Create Wav2Vec2Config(hidden_size=1280, ...)
# 4. Load mapped weights into Wav2Vec2ForCTC
# 5. Save as HF checkpoint
#
# Non-trivial but doable -- sherpa-onnx proved the weights are standard tensors
```

**Verdict: HIGH POTENTIAL, FRAMEWORK TAX.** The best encoder-for-the-money if you are willing to deal with fairseq2. If someone writes the weight conversion (Path C), this becomes immediately competitive with w2v-BERT 2.0. The 300M variant could replace XLSR-53 entirely.

---

### Option 5: Omnilingual ASR W2V (SSL Encoder Only -- No CTC Head)

> Same 4.3M hours / 1600 languages as Option 4, but raw SSL encoder. Same concept as w2v-BERT 2.0 -- you add your own CTC head.

| Property | Value |
|----------|-------|
| GitHub | [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) |
| Paper | [Omnilingual ASR (arXiv:2511.09690)](https://arxiv.org/abs/2511.09690) |
| Architecture | wav2vec2 Transformer (SSL pretrained, no output head) |
| Pretraining | **4,300,000 hours**, 1,600+ languages |
| Output | **None** -- SSL encoder only |
| Temporal resolution | **20ms per frame** |
| License | **Apache 2.0** |
| Framework | **fairseq2** |

**Available sizes**:

| Model | Params | Size | HuggingFace ID |
|-------|--------|------|----------------|
| W2V 300M | 317M | 1.2 GB | [facebook/omniASR-W2V-300M](https://huggingface.co/facebook/omniASR-W2V-300M) |
| **W2V 1B** | **965M** | **3.6 GB** | [facebook/omniASR-W2V-1B](https://huggingface.co/facebook/omniASR-W2V-1B) |
| W2V 3B | 3B | 12 GB | [facebook/omniASR-W2V-3B](https://huggingface.co/facebook/omniASR-W2V-3B) |
| W2V 7B | 6.5B | 25 GB | [facebook/omniASR-W2V-7B](https://huggingface.co/facebook/omniASR-W2V-7B) |

**Why yes**:
- Same massive pretraining as Option 4 (4.3M hours, 1600+ languages)
- SSL-pretrained means the encoder learned general speech representations (not biased toward any specific vocabulary)
- The 7B model is the largest speech SSL encoder publicly available
- Apache 2.0

**Why not**:
- Same fairseq2 friction as Option 4
- No CTC head at all -- more adaptation than Option 4 (which already has a character CTC head you can swap)
- Option 4 is strictly easier -- the CTC variants already went through CTC training, so the encoder is already optimized for CTC alignment. Starting from the SSL encoder means you would need to CTC-train from scratch

**When to prefer Option 5 over Option 4**:
- If you want a clean slate -- the CTC variants were fine-tuned on character-level ASR, which may have biased the encoder slightly toward character recognition. The W2V versions are "purer" SSL representations
- If you want the 7B model -- the CTC 7B exists too, but the W2V 7B may give better features for novel tasks since it was not specialized
- In practice: just use Option 4 (CTC variants) -- the existing CTC head proves the encoder works for CTC, and swapping the head is cheaper than training one from scratch

**w2v-BERT 2.0 vs Omnilingual W2V -- head to head**:

Since these are both "SSL encoder, add your own CTC head":

| | **w2v-BERT 2.0** (Option 2) | **Omnilingual W2V 1B** (Option 5) |
|---|---|---|
| Pretraining hours | 4.5M | 4.3M |
| Languages | 143 | 1,600 |
| Architecture | Conformer | wav2vec2 Transformer |
| Framework | **HuggingFace** | **fairseq2** |
| Fine-tuning docs | [Complete blog post](https://huggingface.co/blog/fine-tune-w2v2-bert) | None |
| Sizes available | 600M only | 300M, 1B, 3B, 7B |
| License | MIT | Apache 2.0 |

Comparable pretraining scale. w2v-BERT wins on framework. Omnilingual wins on language coverage and size range.

**Verdict: USE OPTION 4 INSTEAD.** Same encoder, but Option 4 already has CTC training done. Less work for you.

---

### Option 6: ZIPA-CR (Best Phone Recognizer Available -- Also Just Works)

> 77% better phone recognition than wav2vec2-xlsr-espeak. 127 IPA tokens, 20ms resolution, MIT license. Plug and play like Option 1.

| Property | Value |
|----------|-------|
| GitHub | [lingjzhu/zipa](https://github.com/lingjzhu/zipa) |
| Paper | [ZIPA (arXiv:2505.23170)](https://arxiv.org/abs/2505.23170), ACL 2025 |
| HuggingFace | [anyspeech](https://huggingface.co/anyspeech) org (all variants) |
| Architecture | Zipformer (U-Net-style multi-scale Transformer) with CR-CTC |
| Parameters | **64M** (small), **300M** (large) |
| Training data | **17,132 hours** labeled (IPAPack++, 88 languages) + 11,851h pseudo-labeled |
| Pretraining | **None** -- trained from scratch, supervised only (no SSL) |
| Output vocabulary | **127 IPA tokens** (SentencePiece unigram) |
| Temporal resolution | **20ms per frame** (50Hz, CR-CTC variant) |
| License | **MIT** |
| Framework | **icefall / k2** (Next-gen Kaldi) -- NOT HuggingFace Transformers |
| ONNX available | Yes (fp32, fp16, int8) -- **bypasses framework dependency** |

**Available models**:

| Model | Params | Type | HuggingFace |
|-------|--------|------|-------------|
| Zipa-CR-Small 500k | 64M | CR-CTC | [anyspeech/zipa-small-crctc-500k](https://huggingface.co/anyspeech/zipa-small-crctc-500k) |
| **Zipa-CR-NS-Large 800k** | **300M** | CR-CTC + Noisy Student | [**anyspeech/zipa-large-crctc-ns-800k**](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k) |
| Zipa-CR-NS-Small 700k | 64M | CR-CTC + NS | [anyspeech/zipa-small-crctc-ns-700k](https://huggingface.co/anyspeech/zipa-small-crctc-ns-700k) |
| Zipa-CR-NS-Large (no diacritics) | 300M | CR-CTC + NS | [anyspeech/zipa-large-crctc-ns-no-diacritics-780k](https://huggingface.co/anyspeech/zipa-large-crctc-ns-no-diacritics-780k) |

Also available: Transducer variants (Zipa-T), but CR-CTC is what we want for frame-level posteriors.

**Phone recognition accuracy (PFER = Phonetic Feature Error Rate)**:

| Model | Seen Languages PFER | Unseen Languages PFER |
|-------|--------------------|-----------------------|
| Allosaurus (11M) | 22.33 | 5.24 |
| **wav2vec2-xlsr-53-espeak (Option 1)** | **11.88** | **3.65** |
| **ZIPA-CR-NS-Large (300M)** | **2.71** | **3.20** |

Per-language on seen languages:

| Language | wav2vec2-xlsr-espeak | ZIPA-CR-NS-Large | Improvement |
|----------|---------------------|------------------|-------------|
| English (CommonVoice) | 5.45 | 0.66 | **88%** |
| German | 11.61 | 3.07 | 74% |
| Mandarin | 6.20 | 0.38 | 94% |
| French | 14.36 | 2.76 | 81% |

**Why yes**:
- Zero adaptation needed -- same as Option 1, outputs IPA phonemes directly
- 77% better phone recognition than wav2vec2-xlsr-espeak on seen languages
- 20ms temporal resolution (same as Options 1 and 2)
- MIT license -- most permissive possible
- ONNX models bypass the icefall framework entirely -- just `onnxruntime + soundfile`
- The 64M small model is incredible for edge deployment (better than wav2vec2-xlsr at 300M!)
- 127 IPA tokens covers all major speech sounds
- Two sizes let you trade off accuracy vs compute

**Why not**:
- icefall/k2 framework for PyTorch path (not HuggingFace). ONNX path avoids this
- Trained on G2P labels (not human phoneticians) -- may over-normalize actual pronunciation variation. For GOP against canonical targets this is actually helpful, but for detecting subtle L2 errors it could mask real variation
- Supervised only (no SSL pretraining) -- the 17K training hours is much less than wav2vec2's 56K SSL or w2v-BERT's 4.5M SSL. The architecture and loss function compensate, but less raw data exposure
- Newer and less battle-tested than wav2vec2-xlsr (which has years of GOP literature behind it)
- Nobody has tested ZIPA for GOP yet -- you would be the first

**Extracting posteriors for GOP**:
```python
# ONNX path (RECOMMENDED -- no framework dependencies)
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("zipa-large-crctc-ns-800k.onnx")
outputs = session.run(None, {"audio": audio_features})
log_probs = outputs[0][0]  # [time, 127] -- frame-level log probabilities
posteriors = np.exp(log_probs)  # convert to probabilities for GOP

# PyTorch path (requires icefall + k2 + lhotse + kaldifeat)
encoder_out, encoder_out_lens = model.encoder.forward_encoder(feature, feature_lens)
ctc_output = model.encoder.ctc_output(encoder_out)  # [batch, time, 127]
posteriors = torch.softmax(ctc_output, dim=-1)
```

**How ZIPA compares to Option 1 (wav2vec2-xlsr-espeak)**:

| | **Option 1: wav2vec2-xlsr-espeak** | **Option 6: ZIPA-CR** |
|---|---|---|
| Phone recognition | 11.88 PFER (seen) | **2.71 PFER (seen)** -- 77% better |
| Unseen languages | 3.65 PFER | **3.20 PFER** -- 12% better |
| Vocab | 387 IPA labels | 127 IPA tokens |
| Temporal resolution | 20ms | 20ms |
| Params | 300M | 64M or 300M |
| Pretraining style | SSL (56K hrs) + CTC fine-tune | Supervised from scratch (17K hrs) |
| Framework | **HuggingFace** | icefall (or **ONNX**) |
| License | Apache 2.0 | MIT |
| GOP literature | Proven SOTA | Untested for GOP |
| Edge deployment | 300M only | **64M small model** |

**Verdict: HIGHEST POTENTIAL FOR PHONE RECOGNITION.** If the 77% PFER improvement translates to better GOP scores (which it should -- better phone recognition = better posteriors), ZIPA-CR could outperform wav2vec2-xlsr as the GOP backbone. The 64M model is also the best option for edge deployment. The ONNX path makes integration straightforward.

**Recommended approach**: Run ZIPA-CR alongside Option 1 on the same test set and compare GOP scores directly. If ZIPA wins, it becomes the new default.

---

### Considered but Not Recommended

#### OWSM-CTC v4 1B -- Leaning No

| Property | Value |
|----------|-------|
| Architecture | E-Branchformer (27 layers, d=1024) |
| Parameters | 1.01B |
| Training data | 320,000 hours, 151+ languages |
| Output vocabulary | ~50,000 BPE tokens |
| Temporal resolution | 80ms per frame |
| License | CC-BY-4.0 |
| Framework | **ESPnet** (not HuggingFace) |

**Why not**:
- ESPnet is a heavier, less well-documented framework than HuggingFace or NeMo
- Fine-tuning documentation for OWSM-CTC is incomplete (known issue [#5930](https://github.com/espnet/espnet/issues/5930))
- 50K BPE vocabulary is the farthest from phonemes of any option
- 80ms resolution (same tradeoff as Parakeet, but without Parakeet's English dominance)
- The 320K hours of training is impressive, but w2v-BERT 2.0 has 4.5M hours (14x more)
- If you want multilingual, w2v-BERT 2.0 (Option 2) is strictly better: more data, better framework, better docs

One interesting thing: OWSM-CTC has intermediate CTC outputs at layers 6, 12, 15, 21. Lower layers capture more acoustic/phonetic info. This could be exploited for pronunciation features, but nobody has tried it.

**Bottom line**: Not worth the framework complexity when w2v-BERT 2.0 exists.

#### Qwen3-ASR -- Wrong Architecture (Use as Sidecar)

Not CTC-trained (uses attention-based encoder-decoder). The encoder is powerful (40M hours of pretraining) but produces contextualized representations designed for an autoregressive LLM, not frame-level phoneme posteriors. Adding a CTC head would partially negate the pretraining benefit.

**Qwen3-ForcedAligner** is the best forced aligner available (3-4x more accurate than Montreal FA), but outputs timestamps, not posteriors. Could be useful as a preprocessing step in a hybrid pipeline.

---

### Decision Summary

| Option | Adaptation | Time to GOP | Temporal Res | Pretraining | License | Framework | Best For |
|--------|-----------|-------------|-------------|-------------|---------|-----------|----------|
| **1. XLSR-espeak** | None | **Minutes** | **20ms** | 56K hrs | Apache 2.0 | HuggingFace | Getting started (HF ecosystem) |
| **2. w2v-BERT 2.0** | Add phoneme CTC head + fine-tune | ~4-8 hrs | **20ms** | **4.5M hrs** | MIT | HuggingFace | Best HF encoder, multilingual |
| **3. Parakeet-0.6B** | Replace BPE head + fine-tune | ~2-4 hrs | 80ms | 64K hrs | CC-BY-4.0 | HF + NeMo | Best English encoder |
| **4. omniASR CTC** | Swap char head -> phoneme + fine-tune | ~2-4 hrs | **20ms** | **4.3M hrs** | Apache 2.0 | **fairseq2** | Best if you accept fairseq2 |
| **5. omniASR W2V** | Add CTC head from scratch + fine-tune | ~8+ hrs | **20ms** | **4.3M hrs** | Apache 2.0 | **fairseq2** | Use Option 4 instead |
| **6. ZIPA-CR** | **None** | **Minutes** | **20ms** | 17K hrs (supervised) | MIT | icefall / **ONNX** | **Best phone recognition, edge** |
| ~~OWSM-CTC v4~~ | Replace head + ESPnet | ~8+ hrs | 80ms | 320K hrs | CC-BY-4.0 | ESPnet | Skip -- use Option 2 |

### The Two "Just Works" Options (No Adaptation Needed)

| | Option 1 (XLSR-espeak) | Option 6 (ZIPA-CR) |
|---|---|---|
| Phone accuracy (PFER) | 11.88 | **2.71** (77% better) |
| Vocab | 387 IPA | 127 IPA |
| Framework | **HuggingFace** (easiest) | ONNX or icefall |
| GOP literature proof | **Proven SOTA** | Untested |
| Edge model | 300M only | **64M available** |

### Recommended Timeline

```yaml
Week 1:  Option 1 (XLSR-espeak) + Option 6 (ZIPA-CR via ONNX)
         | run BOTH on the same test set, compare GOP scores
         | validate end-to-end: audio -> posteriors -> GOP-SF -> scores
         | test on SpeechOcean762 and CMU Kids
         | whichever scores better becomes your baseline

Week 2+: Option 2 (w2v-BERT 2.0) -> fine-tune phoneme CTC head on HuggingFace
         | swap encoder, keep same GOP pipeline
         | compare scores to Week 1 baseline

  OR     Option 4 (omniASR CTC 1B) -> swap char head for phoneme head in fairseq2
         | comparable pretraining to Option 2, 10x more languages
         | worth it if you also want non-English pronunciation

Optional: Option 3 (Parakeet) -> if English-only and 80ms resolution is acceptable

Sidecar: Qwen3-ASR + ForcedAligner for word-level transcription + timestamps
         | runs alongside the phoneme model, not instead of it
         | needed for: word highlighting, fidelity checking, LLM feedback
```

---

## 5. Repository Assessments

### ai-pronunciation-trainer: a baseline to surpass, not build on

This repo (~430 stars, AGPL-3.0) uses **Whisper ASR -> Epitran G2P -> phoneme edit distance**, which is fundamentally the wrong approach for production pronunciation assessment. It transcribes speech with Whisper, converts both expected and recognized text to IPA via rule-based G2P, then computes a percentage-match score per word. This means **any pronunciation error that Whisper "corrects" during transcription scores as perfect** -- and Whisper is explicitly trained to be robust to accented speech. The repo provides no training code, no pretrained scoring models, no acoustic-level analysis, and only supports English and German. Its value is as a reference UI implementation (Flask + HTML/JS frontend) and a clear illustration of what *not* to do at the acoustic scoring layer. The architecture pattern of "ASR -> text comparison" should be replaced entirely with posterior-probability-based GOP scoring.

### Qwen3-ASR: excellent transcription, no phoneme granularity

Qwen3-ASR (Apache 2.0, released January 2026) is a **Large Audio-Language Model** achieving state-of-the-art WER on English (4.5% on TED-LIUM vs. Whisper's 6.8%) and Chinese (5.0% vs. Whisper's 9.9%) across **52 languages/dialects**. The 1.7B model uses an AuT encoder with a Qwen3 LLM backbone. The companion **Qwen3-ForcedAligner-0.6B** is the first LLM-based non-autoregressive forced aligner, supporting 11 languages with **67-77% lower timestamp error** than Montreal Forced Aligner on word boundaries.

However, Qwen3-ASR has a **fundamental limitation for pronunciation assessment**: it outputs word/character-level text only, with no phoneme sequences, no phoneme posteriors, and no phoneme-level alignment. The ForcedAligner operates at word/character granularity, not phoneme granularity. The Qwen3-ASR-Toolkit is merely an API client for DashScope cloud, not a training framework. **Recommended role**: Use Qwen3-ASR as a high-accuracy transcription layer for spontaneous speech assessment and word-level fidelity classification (detecting omissions, insertions, repetitions), then hand off to a CTC-based phoneme model for fine-grained scoring.

### Facebook Omnilingual ASR: massive scale, adaptable CTC backbone

Meta's Omnilingual ASR (Apache 2.0, December 2025) covers **1,600+ languages** with models from 300M to 7B parameters in both CTC and LLM-ASR variants. The CTC models (`omniASR_CTC_300M/1B/3B_v2`) project wav2vec2 encoder embeddings to character-level vocabulary logits -- exactly the architecture needed for pronunciation assessment if retrained with a phoneme vocabulary. Training recipes are provided at `workflows/recipes/wav2vec2/asr/` with YAML configs:

```bash
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
  --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml
```

Recommended fine-tuning hyperparameters for low-resource phoneme recognition: `lr: 1e-05`, `num_steps: 5000`, `grad_accumulation: 4 batches`. However, the compute requirements are steep: **32 GPUs for the 300M model, 96 GPUs for 3B**. The system outputs characters, not phonemes, so the CTC head must be replaced and fine-tuned.

The far more immediately useful model from Meta's ecosystem is **`facebook/wav2vec2-xlsr-53-espeak-cv-ft`** -- wav2vec2-large-xlsr-53 fine-tuned on CommonVoice for IPA phoneme recognition in **60+ languages**. This model directly outputs IPA phoneme sequences and frame-level posterior probabilities, making it the single most important building block for the entire pronunciation assessment pipeline. See Section 4 for full model comparison.

### Allosaurus: elegant concept, outdated accuracy

Allosaurus (ICASSP 2020, ~660 stars) is a **universal phone recognizer** using a BiLSTM + CTC architecture with a linguistically-motivated allophone layer that maps between language-independent phones (187 from training) and language-specific phonemes via PHOIBLE inventories (2,000+ languages). At only **11M parameters**, it is extremely lightweight and can be fine-tuned with as few as 100 utterances for 40-59% PER reductions.

The ZIPA benchmark (ACL 2025) reveals the accuracy gap: **Allosaurus achieves 22.3% PFER vs. 11.9% for wav2vec2-xlsr-53-ft and 2.7% for ZIPA-LARGE**. Allosaurus uses MFCC features rather than learned SSL representations, which fundamentally limits its discriminative power. Its value lies in (1) its conceptual architecture -- the allophone layer mapping phones to language-specific phonemes is exactly the right framework, (2) its PHOIBLE integration for language inventory coverage, and (3) edge deployment where 11M params matters. **Do not use Allosaurus as the primary recognizer; adopt its phone-phoneme mapping concept with a modern SSL backbone.**

### Segmentation-free GOP (arXiv 2507.16838): the scoring breakthrough

This paper by Cao et al. (NTNU/KTH, IEEE journal, code at `github.com/frank613/CTC-based-GOP`) is **the most directly applicable resource for building this system**. It solves the critical problem of computing GOP scores from CTC-based models like wav2vec2, which have "peaky" outputs that break traditional forced-alignment-based GOP.

Three key methods are proposed. **GOP-SA (Self-Alignment)** uses the CTC model's own activations for both alignment and scoring. **GOP-SF (Segmentation-Free)** marginalizes over all possible phoneme segmentations using the CTC forward algorithm, eliminating forced alignment entirely. **GOP-SF-SD** extends this to handle substitution and deletion errors. The method constructs **41-dimensional feature vectors** per phoneme (1 log posterior + 39 substitution ratios + 1 deletion ratio) that feed into downstream scorers.

Results on speechocean762 with wav2vec2-xlsr-53 + CTC fine-tuned on LibriSpeech: GOP-SF-SD features with GOPT Transformer achieve **0.618 phoneme PCC** -- the best published result for this architecture. On CMU Kids mispronunciation detection, GOP-SF-SD reaches **0.915 AUC-ROC** for real errors, dramatically outperforming traditional DNN-GOP (0.796).

---

## 6. Dataset Landscape

| Dataset | Size | L2 | L1s | Annotations | License | Use |
|---|---|---|---|---|---|---|
| **speechocean762** | 5K utterances, ~7.5h | English | Mandarin | Phone/word/sentence scores | CC BY 4.0 | Primary training + benchmark |
| **L2-ARCTIC** | 27h, 27K utterances | English | 6 L1s | Phone substitutions/deletions | CC BY-NC 4.0 | MDD training |
| **Common Voice** | 26K+ hours | 104 languages | Native | Transcripts only | CC0 | Native acoustic model training |
| **MLS** | 50.5K hours | 8 languages | Native | Transcripts | CC BY 4.0 | Native model training |
| **FLEURS** | ~12h/language, 102 languages | Parallel | Native | Transcripts | CC BY 4.0 | Evaluation |
| **OMPAL** | 1,768 utterances | Mandarin | French | Word+sentence scores | Open | Mandarin L2 pilot |
| **ISLE** | ~18h | English | German, Italian | Phone errors, stress | Academic free | Historical reference |

### SpeechBlender: Synthetic Mispronunciation Data

The **SpeechBlender** method (arXiv 2211.00923) is critical for scaling: it generates training data with realistic mispronunciation patterns from *only native speech* by interpolating raw signals between phonetically similar segments. This achieved a **2% PCC gain over SOTA** on speechocean762 and generalizes to Arabic. Combined with TTS-based reference generation (synthesizing ideal pronunciations with voice cloning), you can bootstrap pronunciation assessment for new languages without collecting L2 speech.

---

## 7. Multilingual Architecture Decisions

The best base model for multilingual phoneme recognition is **`facebook/wav2vec2-xlsr-53-espeak-cv-ft`** for immediate deployment (60+ languages, IPA output, CTC posteriors) and **ZIPA-LARGE** for highest accuracy (88 languages, 2.7% PFER). For extreme language coverage, MMS models (`facebook/mms-1b-all`) support 1,107 languages but require per-language CTC head adaptation from character to phoneme output.

### The Allophone Layer Concept

Allosaurus's universal phone inventory concept is **architecturally correct** -- separating language-independent phones from language-specific phonemes via a trainable allophone layer initialized from PHOIBLE mappings. The recommended approach is to **graft this concept onto a modern backbone**: take wav2vec2-xlsr or ZIPA features, add an allophone-style mapping layer using PHOIBLE inventories, and train end-to-end. This gives you the linguistic elegance of Allosaurus with the acoustic power of SSL representations.

### PHOIBLE and Language Inventory Coverage

PHOIBLE (`phoible.org`) provides 3,020 phonological inventories for 2,186 languages. By cross-referencing a learner's L1 inventory with the target language, you can:
- Predict which phonemes will be difficult (phonemes absent from L1)
- Identify likely substitution patterns (L1 phonemes acoustically closest to target)
- Generate L1-specific feedback and practice exercises

### Training Data Bootstrapping for New Languages

Training data for multilingual pronunciation assessment is the critical bottleneck. Only speechocean762 (English, Mandarin L1) and L2-ARCTIC (English, 6 L1s) provide phoneme-level pronunciation annotations. For new languages, the bootstrapping strategy is:

1. Fine-tune wav2vec2-xlsr on native speech from Common Voice (26K+ hours, 104 languages, CC0)
2. Use **SpeechBlender** augmentation to generate synthetic mispronunciations from native data only
3. Collect ~2,000-5,000 annotated L2 utterances (minimum viable)
4. Transfer scoring models from English using multi-task training

Research shows adding typologically related languages improves performance by ~30%.

---

*Last updated: 2026-02-22*
