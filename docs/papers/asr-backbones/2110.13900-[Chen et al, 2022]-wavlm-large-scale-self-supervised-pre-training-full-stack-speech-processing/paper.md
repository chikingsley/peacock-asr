---
arxiv: 2110.13900
title: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"
authors:
  - "Sanyuan Chen"
  - "Chengyi Wang"
  - "Zhengyang Chen"
  - "Yu Wu"
  - "Shujie Liu"
  - "Zhuo Chen"
  - "Jinyu Li"
  - "Naoyuki Kanda"
  - "Takuya Yoshioka"
  - "Xiong Xiao"
  - "Jian Wu"
  - "Long Zhou"
  - "Shuo Ren"
  - "Yanmin Qian"
  - "Yao Qian"
  - "Michael Zeng"
  - "Xiangzhan Yu"
  - "Furu Wei"
citation_author: "Chen et al"
year: 2022
venue: "IEEE JSTSP 2022"
doi: "10.1109/JSTSP.2022.3188113"
pages: "1505-1518"
source_pdf: "paper.pdf"
extraction_method: "Manually summarized from the published PDF; no local LaTeX source was added in this pass."
extracted_at: "2026-03-22"
llm_friendly: true
---

# WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing

## Metadata

- Authors: Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Michael Zeng, Xiangzhan Yu, Furu Wei
- Venue: IEEE Journal of Selected Topics in Signal Processing 2022
- DOI: 10.1109/JSTSP.2022.3188113
- arXiv: 2110.13900
- Pages: 1505-1518
- Task: universal speech SSL backbone for full-stack speech tasks

## TL;DR

WavLM pushes speech SSL from mostly-ASR transfer toward a general-purpose speech backbone. It combines masked speech prediction with denoising, adds gated relative position bias, expands the pretraining data mix, and explicitly targets full-stack speech tasks instead of only recognition. This is why it keeps showing up in later pronunciation, speaker, and fused-SSL systems.

## Abstract

The paper argues that speech SSL needs to model more than linguistic content if it is going to transfer well across ASR, speaker, and multi-speaker tasks. WavLM therefore combines masked speech prediction with a denoising objective so that the model retains speech-content modeling while gaining robustness and speaker-aware information useful for non-ASR tasks. It also introduces gated relative position bias in the Transformer and scales the pretraining corpus from `60k` to `94k` hours. The resulting model achieves state-of-the-art results on the SUPERB benchmark and strong transfer across diverse speech tasks.

## Method

- Start from the HuBERT-style masked prediction setup.
- Add speech denoising during pretraining so masked prediction happens against the original clean target.
- Simulate noisy and overlapped speech during training to improve non-ASR transfer.
- Use gated relative position bias in the Transformer.
- Increase data scale and diversify the pretraining mixture.

## Results

- The paper reports state-of-the-art performance on the SUPERB benchmark at the time of publication.
- WavLM Large improves a range of representative speech tasks beyond ASR, including speaker and multi-speaker settings.
- The extracted PDF notes `1.8 / 3.2` WER on LibriSpeech `test-clean / test-other`, while the broader point of the paper is that gains are not limited to recognition.

## Relevance To Peacock

WavLM is the third canonical SSL backbone the vault needed. It matters directly for this repo because several pronunciation papers here treat WavLM as one of the three default frozen SSL feature sources, and the broader layer-fusion discussion only makes sense if the source paper for that backbone is actually in the vault.
