---
title: "T5G2P: Using Text-to-Text Transfer Transformer for Grapheme-to-Phoneme Conversion"
authors:
  - "Markéta Řezáčková"
  - "Jan Švec"
  - "Daniel Tihelka"
citation_author: "Řezáčková et al"
year: 2021
doi: "10.21437/Interspeech.2021-546"
pages: "6-10"
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf; the existing local paper.md extraction was used only as a noisy cross-check."
extracted_at: "2026-03-07T20:03:41-08:00"
llm_friendly: true
---

# Title

T5G2P: Using Text-to-Text Transfer Transformer for Grapheme-to-Phoneme Conversion

## Metadata

- Authors: Markéta Řezáčková, Jan Švec, Daniel Tihelka
- Venue: Interspeech 2021
- Pages: 6-10
- DOI: 10.21437/Interspeech.2021-546
- Task: sentence-level grapheme-to-phoneme conversion for English and Czech

## TL;DR

The paper adapts T5 to sentence-level G2P and shows that it is much stronger than the authors' earlier encoder-decoder baseline, especially on English and especially for long words and homographs.

For Czech, a hand-engineered dictionary-plus-rules system remains slightly better overall, but T5 gets very close and is strong enough to support a useful hybrid workflow for detecting irregular loanwords.

## Abstract

The paper treats G2P as a text-to-text transformation problem rather than a word-only dictionary lookup or a smaller sequence-to-sequence model. By fine-tuning T5 on sentence-level orthography-to-phoneme pairs, the authors aim to capture contextual effects such as homograph disambiguation and cross-word phonological processes. They evaluate on English, where spelling is irregular, and Czech, where pronunciation is more regular but loanwords create exceptions.

## Research Question

Can a T5-based sentence-level text-to-text model outperform traditional dictionary/rule systems and earlier encoder-decoder neural G2P models, especially on long words, homographs, and irregular cases?

## Method

- Use `t5-base` as the main architecture, with about `220M` parameters.
- For English, start from Google's pretrained English T5 model.
- For Czech, pretrain a T5 model on Czech Common Crawl data using the original T5-style preprocessing.
- Fine-tune on sentence-level orthography-to-phoneme pairs rather than isolated words.
- Compare against:
- traditional dictionary-plus-rules G2P
- an earlier encoder-decoder DNN G2P system from the same research line
- Run an extra Czech experiment where T5 detects likely loanword exceptions so they can be handled by a hybrid pipeline.

## Data

- English fine-tuning data: `128,532` unique sentences with phonetic transcriptions
- Czech fine-tuning data: `442,029` unique sentences with phonetic transcriptions
- Train/validation/test split: `80% / 10% / 10%`
- Fine-tuning duration: 50 epochs, 2,000 steps per epoch
- Czech pretraining corpus: `47.9GB` of clean Common Crawl text, `20.5M` unique URLs, `6.7B` running words
- Baseline lexical resources:
- English dictionary of about `300,000` words plus more than `1,000` fallback rules
- Czech system with about `100` rules plus a dictionary of more than `170,000` irregular word forms

## Results

- English:
- dictionary + rules: `54.49%` sentence accuracy, `90.93%` word accuracy, `97.20%` phoneme accuracy
- encoder-decoder: `82.75%`, `95.72%`, `97.18%`
- T5G2P: `91.84%`, `99.04%`, `99.68%`
- Czech:
- rules only: `56.74%` sentence accuracy, `90.97%` word accuracy, `99.36%` phoneme accuracy
- dictionary + rules: `98.86%`, `99.99%`, `99.99%`
- encoder-decoder: `88.64%`, `98.69%`, `99.51%`
- T5G2P: `98.77%`, `99.89%`, `99.97%`
- The T5 model clearly fixes the earlier encoder-decoder model's long-word weakness.
- On isolated unseen words, the reported error rate is still non-trivial:
- English unseen-word error rate: `33.8%`
- Czech unseen-word error rate: `2.3%`
- Czech loanword detection is very strong:
- accuracy `99.97%`
- precision `99.51%`
- recall `99.72%`
- F1 `99.62%`
- English homograph handling is also strong in the reported examples:
- `live`: `100%` for both variants
- `read`: `90%` for `[rEd]`, `100%` for `[ri:d]`
- `record`: `100%` for both reported variants

## Limitations / Notes

- The training data are proprietary, so the exact recipe is not directly reproducible from public resources alone.
- The English unseen-word result shows that the model is still benefiting from large sentence-level overlap and context, not solving novel-word pronunciation perfectly in isolation.
- For Czech, the hand-engineered dictionary-plus-rules baseline remains slightly stronger overall, so T5 is not an unconditional replacement in a mature production stack.
- This is a G2P paper, not a pronunciation-scoring paper, so its relevance is upstream: canonical pronunciation generation, lexicon expansion, and contextual phoneme rendering.

## Relevance To Peacock

This paper is useful for the symbolic side of a pronunciation system. Peacock likely needs reliable canonical phoneme sequences for prompts, targets, lexicon expansion, and feedback explanations, and this paper shows that sentence-level transformer G2P is materially better than simpler word-level or encoder-decoder approaches.

The Czech loanword-detection idea is also broadly reusable: instead of forcing a model to fully solve every exception, Peacock could use a hybrid workflow where a strong default G2P handles most text and a model flags likely exception words for special treatment.
