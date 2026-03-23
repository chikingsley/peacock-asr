---
arxiv: 2006.11477
title: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
authors:
  - "Alexei Baevski"
  - "Yuhao Zhou"
  - "Abdelrahman Mohamed"
  - "Michael Auli"
citation_author: "Baevski et al"
year: 2020
venue: "NeurIPS 2020"
doi: "10.48550/arXiv.2006.11477"
source_pdf: "paper.pdf"
extraction_method: "Manually summarized from the published PDF; no local LaTeX source was added in this pass."
extracted_at: "2026-03-22"
llm_friendly: true
---

# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

## Metadata

- Authors: Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli
- Venue: NeurIPS 2020
- arXiv: 2006.11477
- DOI: 10.48550/arXiv.2006.11477
- Task: self-supervised speech representation learning for ASR transfer

## TL;DR

This is the backbone paper that made large-scale speech SSL practically central. The model masks spans in latent speech features, predicts quantized targets with a contrastive objective, and then fine-tunes with CTC. The main result is that speech-only pretraining plus light supervised fine-tuning can beat prior semi-supervised ASR systems, especially in low-label regimes.

## Abstract

The paper proposes a self-supervised framework for learning speech representations directly from raw audio. A convolutional encoder produces latent speech features, masked spans are passed through a Transformer context network, and the model is trained to distinguish the true quantized latent target from distractors. The quantization module is learned jointly. After pretraining, the encoder and Transformer are fine-tuned on labeled speech recognition tasks with a CTC loss. The core claim is that this setup is simpler than earlier semi-supervised pipelines while performing better.

## Method

- Encode raw waveform with a convolutional feature encoder.
- Mask spans in the latent sequence rather than masking raw input.
- Use a Transformer to build contextualized representations over masked latents.
- Learn quantized latent targets jointly with the encoder.
- Train with a contrastive prediction objective during pretraining.
- Fine-tune on downstream ASR using CTC.

## Results

- On LibriSpeech with all labels, the paper reports `1.8 / 3.3` WER on `test-clean / test-other`.
- With only `1 hour` of labeled data, the method beats the prior state of the art on the `100 hour` subset while using far less labeled supervision.
- With only `10 minutes` of labels and `53k` hours of unlabeled pretraining data, it still reaches `4.8 / 8.2` WER on `test-clean / test-other`.
- The paper also reports state-of-the-art results on TIMIT phoneme recognition and the LibriSpeech `100h clean` setup.

## Relevance To Peacock

This is one of the canonical source papers the vault needed. It is the starting point for almost every later SSL layer-selection, fusion, and pronunciation-scoring paper in the repo. Any discussion of using last-layer SSL features, layer averaging, or swapping in new speech backbones is downstream of this model family.
