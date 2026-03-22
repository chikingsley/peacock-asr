# Pronunciation Assessment: Model Options & Pipeline

## How it works

```text
Audio ──→ [Phoneme Model] ──→ [GOP-SF Scoring] ──→ [GOPT Transformer] ──→ Scores
          (Options below)      (the paper)          (this repo)           (0-10)
                                                    trained on SpeechOcean762
```

Phoneme Model: Listens to audio, outputs frame-level IPA phoneme posteriors at every timestep.
GOP-SF (arXiv:2507.16838, code: github.com/frank613/CTC-based-GOP): Turns posteriors into 41-dim feature vectors per phoneme, no forced alignment needed.
GOPT (this repo, github.com/YuanGongND/gopt): Transformer that maps features to human-like pronunciation scores. Trained on SpeechOcean762 (5K utterances with expert annotations).

For a full app, add a word-level ASR sidecar (Whisper, Qwen3-ASR) for transcription + word timestamps alongside the phoneme model.

---

## Plug-and-play options (no adaptation needed)

### Option 1: wav2vec2-xlsr-53-espeak-cv-ft

The proven default. Load it, get posteriors, compute GOP.

```text
Link        https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft
Params      300M
Output      387 IPA phonemes, 20ms/frame
Training    56K hrs SSL (53 langs) + CTC fine-tune on CommonVoice
License     Apache 2.0
Framework   HuggingFace Transformers
Pros        Zero setup, proven in GOP literature, HuggingFace ecosystem
Cons        Older encoder (2021), 11.88 PFER on seen languages
```

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    posteriors = torch.softmax(model(**inputs).logits, dim=-1)  # [1, T, 387]
```

### Option 2: ZIPA-CR (best phone recognition)

77% better phone recognition than Option 1. Untested for GOP but should translate.

```text
Link        https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k
Paper       https://arxiv.org/abs/2505.23170 (ACL 2025)
Code        https://github.com/lingjzhu/zipa
Params      64M (small) / 300M (large)
Output      127 IPA tokens, 20ms/frame
Training    17K hrs supervised (88 langs, IPAPack++) — no SSL
License     MIT
Framework   icefall/k2 or ONNX (recommended)
Pros        2.71 PFER (vs 11.88), 64M edge model, ONNX bypasses framework
Cons        icefall ecosystem (use ONNX), trained on G2P labels, untested for GOP
```

```text
PFER comparison          wav2vec2-xlsr (Opt 1)    ZIPA-CR (Opt 2)
─────────────────────────────────────────────────────────────────
Seen languages avg       11.88                    2.71
English                   5.45                    0.66
Mandarin                  6.20                    0.38
Unseen languages avg      3.65                    3.20
```

```python
# ONNX path (no framework dependencies)
import onnxruntime as ort
session = ort.InferenceSession("zipa-large-crctc-ns-800k.onnx")
outputs = session.run(None, {"audio": audio_features})
log_probs = outputs[0][0]  # [T, 127]
```

---

## Upgrade options (need adaptation, better encoders)

### Option 3: w2v-BERT 2.0 — best multilingual encoder

Add a phoneme CTC head and fine-tune. 80x more pretraining than Option 1.

```text
Link             https://huggingface.co/facebook/w2v-bert-2.0
Fine-tune guide  https://huggingface.co/blog/fine-tune-w2v2-bert
Params           600M
Output           None (SSL encoder only) → add phoneme CTC head
Training         4.5M hrs SSL, 143 languages
License          MIT
Framework        HuggingFace Transformers
Adaptation       Add CTC head + fine-tune (~4-8 hrs on A100)
Temporal res     20ms/frame
```

### Option 4: Parakeet-CTC-0.6B — best English encoder

Replace BPE head with phoneme head. Best English WER (1.87%).

```text
Link             https://huggingface.co/nvidia/parakeet-ctc-0.6b
Docs             https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
Params           600M
Output           1,024 BPE tokens → replace with phoneme head
Training         64K hrs (English only)
License          CC-BY-4.0
Framework        HuggingFace + NeMo
Adaptation       Replace CTC head + fine-tune (~2-4 hrs on A100)
Temporal res     80ms/frame (4x coarser than Options 1/2/3)
```

### Option 5: omniASR CTC — massive multilingual, already has CTC head

Swap the character head for phonemes. Same scale as w2v-BERT 2.0 but 10x more languages.

```text
Links            https://huggingface.co/facebook/omniASR-CTC-300M
                 https://huggingface.co/facebook/omniASR-CTC-1B
                 https://huggingface.co/facebook/omniASR-CTC-7B
Paper            https://arxiv.org/abs/2511.09690
Code             https://github.com/facebookresearch/omnilingual-asr
Params           325M / 975M / 3.1B / 6.5B
Output           9,812 characters → swap to phoneme head
Training         4.3M hrs SSL + CTC fine-tune, 1,600 languages
License          Apache 2.0
Framework        fairseq2 (not HuggingFace)
Adaptation       Swap final_proj linear layer + fine-tune (~2-4 hrs)
Temporal res     20ms/frame
ONNX             https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12
```

---

## Not recommended

```text
Model            Why not
─────────────────────────────────────────────────────────────────────────────────
OWSM-CTC v4     ESPnet framework, 50K BPE, 80ms res. w2v-BERT 2.0 is better
Qwen3-ASR        Not CTC (attention-based). Use as word-level ASR sidecar only
MMS-1B           CC-BY-NC (non-commercial). Replaced by omniASR with Apache 2.0
Allosaurus       22.33 PFER, BiLSTM backbone. Conceptually elegant but outdated
omniASR W2V      Same encoder as omniASR CTC but without the CTC head. Just use
                 the CTC variant (Option 5) — same thing with less work
```

---

## Comparison

```text
Option             Adaptation              Time to GOP   20ms?   Pretraining   License       Framework
──────────────────────────────────────────────────────────────────────────────────────────────────────────
1. XLSR-espeak     None                    Minutes       Yes     56K hrs       Apache 2.0    HuggingFace
2. ZIPA-CR         None                    Minutes       Yes     17K hrs       MIT           ONNX
3. w2v-BERT 2.0    Add head + fine-tune    ~4-8 hrs      Yes     4.5M hrs      MIT           HuggingFace
4. Parakeet        Replace head + tune     ~2-4 hrs      No      64K hrs       CC-BY-4.0     HF + NeMo
5. omniASR CTC     Swap head + fine-tune   ~2-4 hrs      Yes     4.3M hrs      Apache 2.0    fairseq2
```

---

## Recommended path

```text
Week 1:  Run Option 1 + Option 2 on same test data
         Compare GOP scores — whichever wins is baseline
         Validate: audio → posteriors → GOP-SF → GOPT → scores
         Test on SpeechOcean762 + CMU Kids

Week 2+: Upgrade encoder
         HuggingFace path → Option 3 (w2v-BERT 2.0)
         fairseq2 path   → Option 5 (omniASR CTC 1B)
         English-only     → Option 4 (Parakeet)

Sidecar: Qwen3-ASR + ForcedAligner for word transcription + timestamps
```

---

## Key resources

```text
Resource                Link
─────────────────────────────────────────────────────────────────────────────────
GOP-SF paper            https://arxiv.org/abs/2507.16838
GOP-SF code             https://github.com/frank613/CTC-based-GOP
GOPT (this repo)        https://github.com/YuanGongND/gopt
SpeechOcean762          https://www.openslr.org/101/
ZIPA paper              https://arxiv.org/abs/2505.23170
w2v-BERT fine-tuning    https://huggingface.co/blog/fine-tune-w2v2-bert
CTC fine-tuning         https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
Omnilingual ASR paper   https://arxiv.org/abs/2511.09690
SpeechBlender           https://arxiv.org/abs/2211.00923 (data augmentation, no code)
Detailed notes          summaries/MODERN_MODELS.md
```
