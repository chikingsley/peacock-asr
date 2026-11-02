---
license: cc-by-4.0
language:
  - tg
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt_ctc-110m
base_model_relation: finetune
datasets:
  - Peacockery/tajik-asr-corpus-v3
tags:
  - tajik
  - speech
  - fastconformer
  - tdt
  - ctc
metrics:
  - wer
  - cer
model-index:
  - name: tajik-parakeet-tdt-110m-v1.1
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: google/fleurs
          name: FLEURS Tajik test, all 600 recordings
          config: tg_tj
          split: test
        metrics:
          - type: wer
            name: Normalized WER
            value: 17.481015571066962
          - type: cer
            name: Normalized CER
            value: 6.079636878887581
---

# tajik-parakeet-tdt-110m-v1.1

Updated Tajik automatic speech recognition model fine-tuned from [NVIDIA/Suno Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m). It uses a FastConformer encoder, TDT and CTC heads, and a Tajik SentencePiece BPE tokenizer with 1,024 tokens.

| Property  | Value                                         |
| --------- | --------------------------------------------- |
| Language  | Tajik, Cyrillic script                        |
| Input     | Mono audio at 16 kHz                          |
| Output    | Lowercase text, generally without punctuation |
| Framework | NVIDIA NeMo                                   |
| Release   | v1.1                                          |
| License   | CC BY 4.0                                     |

## Usage

Tested with NeMo 3.0.0 and PyTorch 2.9.1.

```python
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

checkpoint = hf_hub_download(
    repo_id="Peacockery/tajik-parakeet-tdt-110m-v1.1",
    filename="model.nemo",
)
model = ASRModel.restore_from(checkpoint)
model.eval()

for result in model.transcribe(["audio.wav"]):
    print(result.text)
```

The checkpoint includes its trained tokenizer. Use speech segmentation for long recordings; training clips were 0.5 to 30 seconds long.

## Evaluation

Evaluation uses fp32 inference and the project's Tajik text normalizer on both references and predictions. WER and CER are corpus-level percentages; lower is better.

| Dataset             | Recordings |   WER |   CER |
| ------------------- | ---------: | ----: | ----: |
| FLEURS Tajik test   |        600 | 17.48 |  6.08 |
| Development         |        239 | 17.99 |  5.73 |
| Conversational test |      1,625 | 34.72 | 15.63 |

The final checkpoint was selected by development WER before test evaluation. Development WER was 17.988% for the final checkpoint, 18.085% for the best-validation-loss checkpoint, and 18.801% for the averaged checkpoint. Exact scores are in [evaluation.json](evaluation.json).

FLEURS uses human reference transcripts. The conversational test uses machine-generated references and measures agreement with those transcripts. The previous model's training included clips from reserved evaluation videos; v1.1 excludes the known evaluation-source identities. Native-speaker-reviewed conversational references would provide a stronger quality benchmark.

## Training

The model was trained on the filtered [Tajik corpus v3](https://huggingface.co/datasets/Peacockery/tajik-asr-corpus-v3/tree/3b05a4bb89104c21643081250729595347d1188e): 173,240 clips totaling 987.54 hours. The training set contains 2,250 FLEURS reference-labeled clips and 170,990 clips with Scribe hypotheses and LLM-fused labels. Preparation applies duration limits and known evaluation-source exclusions.

Training started from the original NVIDIA checkpoint with a fresh Tajik BPE tokenizer and ran for 100,000 optimizer steps. At step 14,000, continuation restored the model, optimizer, scheduler, and global step, then switched to unfused TDT loss, shuffled manifest shards, and 12% FLEURS clip sampling. The remaining 88% of sampling used pseudo-labeled clips.

The recipe used BF16 mixed precision, AdamW, a peak learning rate of `3e-4`, 2,000 warmup steps, cosine decay to `1e-5`, and a 120-second batch audio budget. Validation checkpoints were ranked by loss, then compared using standalone fp32 development WER.

## Comparison with the original model

| Model                                                                 | Training data              | FLEURS WER | Conversational WER |
| --------------------------------------------------------------------- | -------------------------- | ---------: | -----------------: |
| [Original](https://huggingface.co/Peacockery/tajik-parakeet-tdt-110m) | Original 1,197-hour export |      19.03 |              33.85 |
| v1.1                                                                  | Filtered v3, 987.54 hours  |      17.48 |              34.72 |

v1.1 reduces FLEURS WER by 1.55 percentage points, an 8.1% relative reduction. The conversational comparison has the source-overlap and reference-quality limitations described above. The original checkpoint and its evaluation remain available in the linked model repository.

## Files and reproducibility

- `model.nemo`: model weights and embedded tokenizer.
- `tokenizer/`: matching tokenizer files.
- `manifests/`: compressed training, development, and evaluation manifests with relative audio paths and available source identities.
- [release.json](release.json): model, tokenizer, dataset, and manifest fingerprints.
- [evaluation.json](evaluation.json): full evaluation scores.

The audio is available through the source dataset. Manifest paths refer to the preparation output under an `audio/` directory. Training code is in [Peacock ASR](https://github.com/chikingsley/peacock-asr), commit `55af600a`.

Model SHA-256: `9c4674e40c4c7d12087f2c5694cf6e9042cbc8afb7eab173935b87f5ee6cbdea`.

## License

Model weights and Peacockery's fine-tuning contributions are licensed under [CC BY 4.0](LICENSE). Attribution belongs to NVIDIA/Suno for the base model and Peacockery Studio for the Tajik fine-tune. See [NOTICE](NOTICE).

Training recordings retain their respective rights. FLEURS is CC BY 4.0. Individual YouTube recording permissions remain under provenance review, with unresolved entries recorded as `unknown`. The model license covers rights held by the model licensors; recording rights remain with their owners.
