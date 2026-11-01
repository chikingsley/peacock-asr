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
  - name: Parakeet TDT 110M Tajik v1.1.0
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

# Parakeet TDT 110M Tajik v1.1.0

A compact Tajik speech recognizer from Peacockery Studio, fine-tuned from [NVIDIA/Suno Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m). It uses a FastConformer encoder, hybrid TDT/CTC heads, and a Tajik SentencePiece BPE vocabulary of 1,024 tokens. Input is mono 16 kHz audio; output is lowercase Tajik Cyrillic text with punctuation generally omitted.

This incremental release improves normalized FLEURS test WER from **19.03% to 17.48%**, a **1.55 percentage-point absolute improvement** and approximately **8.1% relative error reduction**. Its compact size supports a different deployment budget from larger multilingual recognizers. Accuracy, memory, and latency should be measured for each target runtime.

## Identity and versioning

- Repository: `Peacockery/parakeet-tdt-110m-tajik`, following `<family>-<size>-<language>`.
- Release tag: `v1.1.0`, an incremental model release.
- Training-data version: `v3`, independent of the model release number.
- Exact artifact identity: `parakeet_tdt_110m_tajik_v3_step_100000`.
- Weight file: `model.nemo`, containing the trained tokenizer and both heads.
- SHA-256: `9c4674e40c4c7d12087f2c5694cf6e9042cbc8afb7eab173935b87f5ee6cbdea`.

The [previous release](https://huggingface.co/Peacockery/tajik-parakeet-tdt-110m/tree/1b213a53b8f8e39e553039f39200467dafe25a7f) retains its historical repository and weights. This repository adopts the consistent family-first name.

## Evaluation

The final checkpoint was selected using standalone **fp32 development WER** across three candidates. The selected checkpoint was then evaluated on the fixed test sets. References and predictions use `omni_curator.process.normalize` with `tgk_Cyrl`; WER and CER are corpus-level percentages. See [evaluation.json](evaluation.json) for exact scores.

| Evaluation          | Clips | Normalized WER | Normalized CER | Reference type                     |
| ------------------- | ----: | -------------: | -------------: | ---------------------------------- |
| FLEURS Tajik test   |   600 |          17.48 |           6.08 | FLEURS human reference transcripts |
| Development         |   239 |          17.99 |           5.73 | Curated development split          |
| Conversational test | 1,625 |          34.72 |          15.63 | Machine-generated transcripts      |

Development WER was 17.988% for the final checkpoint, 18.085% for the best-validation-loss checkpoint, and 18.801% for the averaged candidate. The final checkpoint won this comparison. Raw FLEURS WER was 18.09% before text normalization.

The previous model scored 19.03% normalized WER on the same 600 FLEURS recordings. Its recorded conversational WER was 33.85%, compared with 34.72% here. The previous training export included clips from 143 reserved evaluation videos; this retrain excludes the known evaluation-source identities. Conversational references remain machine-generated, so this benchmark measures agreement with those references. A frozen, native-speaker-reviewed conversational test would provide stronger evidence about real-world transcription quality. Scores from other models, languages, datasets, or normalization schemes have separate comparison scope.

## Training and data

Training used the pinned [Peacockery Tajik corpus v3](https://huggingface.co/datasets/Peacockery/tajik-asr-corpus-v3/tree/3b05a4bb89104c21643081250729595347d1188e), prepared with explicit source-identity and duration filters. The retained training set has **173,240 clips totaling 987.54 hours**: 2,250 FLEURS reference-labeled clips and 170,990 Scribe-labeled clips. The preparation excludes 243 evaluation-identity conflicts and 7,200 clips outside the 0.5–30 second training range. Dataset revision, source-manifest fingerprints, and portable manifests accompany this release.

The model began from the original NVIDIA checkpoint with a fresh Tajik tokenizer. The initial phase reached step 14,000; continuation restored model weights, optimizer, scheduler, and global step, then finished at 100,000 total optimizer steps. The continuation used shuffled 200-row manifest shards with 12% FLEURS and 88% pseudo-labeled clip sampling. These percentages are sampling weights; corpus hours and unique clip counts are separate quantities.

The recipe uses NeMo 3.0.0, BF16 mixed precision, unfused TDT loss during continuation, AdamW at `3e-4`, 2,000 warmup steps, cosine decay to `1e-5`, and duration-based batching with a 120-second audio budget. Validation checkpoints monitor loss, followed by standalone fp32 WER selection. The duration-output loss class count is corrected after vocabulary replacement. The combined recipe changes were evaluated together; the result establishes their combined outcome.

The labels are lowercased and punctuation-free. YouTube labels come from Scribe hypotheses and LLM fusion; agreement measures machine-transcript consistency. The source corpus supplies the training audio. Individual recording-license status remains recorded as `unknown` where provenance is unresolved.

## Usage

The artifact was evaluated with NeMo 3.0.0 and Torch 2.9.1. Load the trained tokenizer bundled in the model:

```python
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

checkpoint = hf_hub_download(
    repo_id="Peacockery/parakeet-tdt-110m-tajik",
    filename="model.nemo",
    revision="v1.1.0",
)
model = ASRModel.restore_from(checkpoint)
model.eval()
for transcription in model.transcribe(["tajik-16khz-mono.wav"]):
    print(transcription.text)
```

Use speech segmentation for long recordings. Training clips were capped at 30 seconds. The release exports the original NeMo artifact; ONNX, quantized, browser, and mobile variants require conversion and target-runtime accuracy validation.

## Reproducibility

[release.json](release.json) records the exact model, tokenizer, dataset revision, manifest hashes, training phases, and file digests. The compressed manifests retain text, duration, source IDs, and available audio hashes. Their audio paths are relative to an `audio/` directory. The source audio is supplied by the linked dataset; the manifests describe the preparation output and preserve the original manifest hashes separately.

The training implementation is in [Peacock ASR](https://github.com/chikingsley/peacock-asr), commit `55af600a`. Its project commands include `tajik-parakeet-prepare`, `tajik-parakeet-train-tokenizer`, `tajik-parakeet-train-tdt`, `tajik-parakeet-upgrade`, and `tajik-parakeet-eval`, invoked through `uv run --project projects/tajik-asr --locked`.

## License and attribution

The model weights and Peacockery's fine-tuning contributions are released under **CC BY 4.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Preserve attribution to NVIDIA/Suno and Peacockery Studio, link to the license, and identify changes such as conversion, quantization, or further fine-tuning.

Suggested attribution: Parakeet TDT 110M Tajik v1.1.0 by Peacockery Studio, fine-tuned from NVIDIA/Suno Parakeet TDT-CTC 110M, licensed under CC BY 4.0.

Training recordings and third-party datasets retain their respective rights and terms. FLEURS is CC BY 4.0. Source permissions for the YouTube recordings remain under provenance review; the dataset's metadata retains that distinction. This weight release grants the rights held by the model licensors; rights in third-party recordings remain with their owners.
