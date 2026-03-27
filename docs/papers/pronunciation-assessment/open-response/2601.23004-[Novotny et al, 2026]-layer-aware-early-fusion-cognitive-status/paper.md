---
arxiv: 2601.23004
title: "Layer-Aware Early Fusion of Acoustic and Linguistic Embeddings for Cognitive Status Classification"
authors:
  - "Krystof Novotny"
  - "Laureano Moro-Velázquez"
  - "Jiri Mekyska"
citation_author: "Novotny et al."
year: 2026
venue: "ICASSP / arXiv:2601.23004"
source_pdf: "paper.pdf"
extraction_method: "Manual section-by-section rewrite from local LaTeX source and pdf text on 2026-03-23."
extracted_at: "2026-03-23"
llm_friendly: true
---

# Layer-Aware Early Fusion of Acoustic and Linguistic Embeddings for Cognitive Status Classification

## Metadata

- Authors: Krystof Novotny, Laureano Moro-Velázquez, Jiri Mekyska
- arXiv: 2601.23004
- Task: clinical speech classification of cognitive status (CN / MCI / ADRD)
- Modalities: acoustic speech embeddings + text embeddings from transcription

## Abstract

The paper studies whether combining acoustic and linguistic embeddings for cognitive status classification is better with early fusion or late fusion, and how fusion performance changes with acoustic encoder layer depth. It evaluates several acoustic + language model combinations on a DementiaBank-derived English dataset (1,629 participants) and reports both F1 score and multiclass log loss across multiple seed runs.

## 1. Introduction

The motivation is that cognitive impairment appears in both acoustic and lexical cues, and neither modality alone captures the full signal. Clinical systems are moving from handcrafted features toward learned multimodal models, but there is still uncertainty about:

- whether gains are due to true multimodal synergy or just ensembling;
- whether final acoustic layers are best or intermediate layers are better;
- whether adding temporal grounding for text helps when pairing with acoustic features;
- how much linguistic information is already present in acoustic embeddings (“semantic leakage”).

The study explicitly compares early fusion (EF) and late fusion (LF) in this framework.

### 1.1 State of the Art

The related literature is grouped into:

- unimodal and simple fusion approaches from acoustic and text modalities,
- hybrid fusion methods with guided interactions such as attention,
- and layer-selection effects in speech encoders.

The authors note mixed findings for EF vs LF and increasing interest in “which layer” rather than only “which model.”

### 1.2 Research Questions

The paper’s four questions are:

- Is improvement from multimodal processing really synergy, not just generic ensembling?
- Is the final encoder layer best for classification or can intermediate layers be more informative?
- Can timing and pause structure from speech improve lexical-stream utility in fusion?
- Do acoustic models already carry semantic content, limiting gains from text fusion?

## 2. Methodology

### 2.1 Materials

Experiments are based on the PREPARE Challenge dataset for cognitive-status prediction (ENGLISH subset only), about 30-second clips from 1,629 unique participants.

Class distributions:

- CN: 929 (388 M / 541 F), mean age 74.9 ± 8.4
- MCI: 134 (66 M / 68 F), mean age 72.5 ± 7.3
- ADRD: 566 (239 M / 327 F), mean age 75.9 ± 8.3

Data split:

- train 64%
- validation 16%
- test 20%
- stratified by diagnosis, sex, and source corpus
- fixed seed for reproducibility

### 2.2 Audio and Text Representations

Four acoustic/text model combinations are evaluated:

- wav2vec 2.0 + DistilBERT
- wav2vec 2.0 + RoBERTa
- Whisper + DistilBERT
- Whisper + RoBERTa

Both acoustic encoders expose 12 layers and 768-dim hidden states. Audio is resampled to 16 kHz, hidden states are extracted frame-wise, and WhisperX word-level alignments provide frame spans.

Text pipeline:

- transcribe with WhisperX,
- tokenize with DistilBERT or RoBERTa,
- take last-layer hidden states,
- map each token to frame spans proportionally to character lengths,
- concatenate token embeddings to acoustic features per frame.

Resulting fusion tensor: `[T, Daudio + Dtext]`, with `T` frames.

All precomputed per-model/per-layer feature tensors are cached before fusion/classification experiments.

Two temporal linguistic variants are explicitly tested:

- TA: replace standard positional index with token-start-time positions (20 ms frame units),
- TA-PAD: insert `[PAD]` tokens between words and assign positions for inter-word silence.

Both variants perform similarly or worse than the standard setup in final tables.

### 2.3 Classification

Two-stage experimental flow:

- hyperparameter tuning,
- then 10 independent full runs for evaluation with different random seeds.

Classifier:

- transformer encoder with search over heads, width/depth, dropout, optimizer/scheduler settings.
- sequence pooling by mask-weighted mean or learnable attention pooling.
- 3-class softmax output for CN/MCI/ADRD.

Optimization:

- Optuna with TPE sampler,
- up to 150 trials per config,
- early stopping using validation log loss,
- report metrics on final seed-averaged test performance.

Evaluation metrics:

- log loss (calibration-heavy),
- F1 score.

Late fusion (LF) is posterior averaging of separately trained audio-only and text-only models. EF uses fused frame tensors directly with the shared transformer classifier.

## 3. Results

The best configuration summary is in a compact table of top-performing settings (Table 2 in the paper).

Best single rows:

- EF (Whisper + RoBERTa, layer 9): F1 `0.633`, log loss `0.687`.
- LF (Whisper + DistilBERT, layer 10): log loss `0.678`.
- Acoustic-only Whisper, layer 10: F1 `0.622`, log loss `0.686`.
- Text-only DistilBERT: F1 `0.491`, log loss `0.803`.
- TA-DistilBERT variant: F1 `0.492`, log loss `0.814`.

Across 48 evaluated settings:

- EF gave the highest F1 in `81.2%`.
- LF gave lowest log loss in `70.8%`.

Peak performance centers around acoustic layer range `8–10`.

### Table 2 (Best strategy-level results)

| Modeling strategy | Layer | F1 score | Log loss |
| --- | ---: | ---: | ---: |
| EF (Whisper + RoBERTa) | 9 | 0.633 | 0.687 |
| LF (Whisper + DistilBERT) | 9 | 0.596 | 0.679 |
| LF (Whisper + DistilBERT) | 10 | 0.585 | 0.678 |
| Acoustic-only (Whisper) | 10 | 0.622 | 0.686 |
| Text-only (TA-DistilBERT) | — | 0.492 | 0.814 |
| Text-only (DistilBERT) | — | 0.491 | 0.803 |

### Figures and interpretation

Figures in the source show layer-wise behavior for Whisper+RoBERTa:

- EF tends to peak around earlier layers for discrimination.
- LF and acoustic-only often peak slightly later.
- At one layer the metric-optimal strategy can differ (F1 vs log loss), indicating calibration/accuracy trade-off.

## 4. Discussion

The authors’ interpretation:

- EF is strongest when acoustic embeddings are in an acoustic-dominant regime, especially lower/mid layers.
- As layers deepen, acoustic embeddings become more lexical/semantic, reducing complementarity with text and shrinking EF gains.
- LF excels on log loss (probability calibration) because it averages independent posteriors.
- EF generally improves class separation and therefore F1.

The paper links these findings to prior work on layer-wise analysis in SSL encoders:

- early layers carry signal-driven cues;
- later layers carry more lexical/semantic patterns.

Evidence from cosine similarity checks on first vs last layer embeddings is used to support the “semantic leakage” explanation.

Model-specific behavior:

- For Whisper, acoustic-only final layer can sometimes outperform EF because text overlap is already high.
- This is interpreted as acoustic- and lexical-information coupling at deep layers.

Clinical framing:

- For screening, maximize F1.
- For risk estimation and threshold calibration, log loss is often more relevant.

## 5. Conclusion

The study concludes:

- use frame-aligned EF/LF with a controlled encoder-layer sweep;
- acoustic + linguistic multimodal fusion is useful but metric-dependent (EF vs LF strengths differ),
- best layers are usually in the mid acoustic depth (`~8–10`),
- text-only systems underperform both acoustic-only and multimodal settings,
- simple timing-based linguistic variants (TA, TA-PAD) do not solve the gap.

Practical direction is adaptive layer-aware fusion and better interpretability for clinical pipelines.

## 6. Acknowledgements

- Funded by LangInLife (CZ.02.01.01/00/23_025/0008726), co-funded by the EU.
- COST Action CA24128 (eVoiceNet).
- Fulbright Visiting Student Researcher Program.

## 7. Notes

The paper cites a challenge-derived preprocessing pipeline and makes available a GitHub repository for reproduction.
