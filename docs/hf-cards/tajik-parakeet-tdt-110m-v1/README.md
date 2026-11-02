---
license: cc-by-4.0
language:
  - tg
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt_ctc-110m
base_model_relation: finetune
tags:
  - tajik
  - speech
  - audio
  - fastconformer
  - tdt
  - ctc
  - automatic-speech-recognition
metrics:
  - wer
  - cer
---

# tajik-parakeet-tdt-110m-v1

The updated model is available at [tajik-parakeet-tdt-110m-v1.1](https://huggingface.co/Peacockery/tajik-parakeet-tdt-110m-v1.1), with 17.48% FLEURS WER. This repository contains the original June 2026 checkpoint, with 19.03% FLEURS WER.

A Tajik speech-recognition fine-tune of [NVIDIA/Suno Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m), released by Peacockery Studio. The model uses a FastConformer encoder and hybrid TDT/CTC heads with a Tajik SentencePiece BPE vocabulary of 1,024 tokens. Input audio is mono at 16 kHz; output is Tajik Cyrillic text.

## License and attribution

The model weights and Peacockery's fine-tuning contributions are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

CC BY 4.0 permits use, redistribution, conversion, and adaptation, including commercial use and further fine-tuning, subject to its terms. Preserve attribution to NVIDIA/Suno and Peacockery Studio, link to the license, and identify changes such as ONNX conversion, quantization, or further fine-tuning. An application credits screen can contain the attribution and links, with the license and notice accompanying the distributed model.

Suggested attribution:

> Tajik Parakeet TDT 110M by Peacockery Studio, fine-tuned from NVIDIA/Suno Parakeet TDT-CTC 110M. Licensed under CC BY 4.0. Model: https://huggingface.co/Peacockery/tajik-parakeet-tdt-110m-v1. Base model: https://huggingface.co/nvidia/parakeet-tdt_ctc-110m. License: https://creativecommons.org/licenses/by/4.0/.

Training recordings and third-party datasets retain their respective rights and terms. Source permissions for the YouTube recordings and the Muhtasham augmented dataset remain under provenance review. This weight release grants the rights held by the model licensors; rights in third-party recordings remain with their owners.

## Training

The published checkpoint is the final model from a 100,000-step full fine-tune completed in June 2026. The training manifest contains 232,824 clips totaling 1,197.36 hours.

| Source                                |   Clips |    Hours | Source information                                                                                                                 |
| ------------------------------------- | ------: | -------: | ---------------------------------------------------------------------------------------------------------------------------------- |
| Tajik YouTube channels                | 229,470 | 1,186.20 | Project-generated Scribe pseudo-labels; individual source permissions under review                                                 |
| Google FLEURS `tg_tj`                 |   2,265 |     8.47 | [FLEURS](https://huggingface.co/datasets/google/fleurs), CC BY 4.0                                                                 |
| Muhtasham augmented Tajik             |     853 |     2.35 | [Dataset](https://huggingface.co/datasets/muhtasham/tajik-asr-augmented-test), license and original-source provenance under review |
| Common Voice Scripted Speech 25 Tajik |     236 |     0.34 | Mozilla Common Voice, CC0                                                                                                          |

The recipe used fresh Tajik BPE-1024 tokenization, BF16 training, a learning rate of `3e-4`, 2,000 warmup steps, cosine decay to `1e-5`, and Lhotse duration-based batching with a 120-second audio budget. Training clips passed a Scribe-agreement WER threshold of 0.35 and duration limits of 0.5-30 seconds. The TDT loss class count was corrected after vocabulary replacement to exclude the duration outputs.

The YouTube labels are machine-generated. Agreement between Scribe passes is a consistency signal; native-speaker review remains necessary to establish transcription correctness.

## Recorded evaluation

These scores come from the saved July 9, 2026 evaluation of the published checkpoint with NeMo 2.7.3, fp32 inference, and the project's Tajik text normalizer applied to references and predictions. WER and CER are corpus-level percentages. Case, punctuation, and other text normalization choices materially affect comparisons.

| Evaluation manifest                  | Clips | Normalized WER | Normalized CER |
| ------------------------------------ | ----: | -------------: | -------------: |
| FLEURS Tajik test                    |   600 |          19.03 |           6.72 |
| Development, FLEURS and Common Voice |   357 |          17.77 |           5.89 |

The October 28, 2026 audit found training clips from 143 videos reserved for an earlier conversational evaluation. That evaluation's recorded 33.85% WER therefore describes a benchmark with overlapping source videos. Generalization assessment requires a separate conversational test set. The supplied FLEURS and development manifests have disjoint audio paths from the training manifest; cross-source augmentation provenance remains under review.

## Usage

Load the bundled checkpoint and tokenizer with NeMo:

```python
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

checkpoint = hf_hub_download(
    repo_id="Peacockery/tajik-parakeet-tdt-110m-v1",
    filename="tajik-parakeet-tdt-110m_final.nemo",
)
model = ASRModel.restore_from(checkpoint)
model.eval()
transcriptions = model.transcribe(["tajik-16khz-mono.wav"])
for transcription in transcriptions:
    print(transcription.text)
```

Use the tokenizer already bundled in the fine-tuned model. NeMo vocabulary replacement rebuilds decoder and joint layers, so it belongs in a deliberate new-language training workflow.

ONNX export, INT8 quantization, and Android deployment are permitted by the model license under its terms. Runtime compatibility and accuracy for each conversion should be measured on the target implementation.

## Artifact identity and reproducibility

- Weight file: `tajik-parakeet-tdt-110m_final.nemo`
- SHA-256: `92e672d08b730a1c50cad7c0369c2d19fc1b9ebc83b16aa0fb5ae69267f3879f`
- Training code and audit: [Peacock ASR](https://github.com/chikingsley/peacock-asr)
- The supplied JSONL evaluation manifests contain historical local audio paths. Obtain the source audio and map those paths to your environment before evaluation.
