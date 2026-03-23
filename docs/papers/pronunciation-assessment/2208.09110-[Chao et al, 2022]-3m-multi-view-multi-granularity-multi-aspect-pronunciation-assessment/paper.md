---
arxiv: 2208.09110
title: "3M: An Effective Multi-view, Multi-granularity, and Multi-aspect Modeling Approach to English Pronunciation Assessment"
authors:
  - "Fu-An Chao"
  - "Tien-Hong Lo"
  - "Tzu-I Wu"
  - "Yao-Ting Sung"
  - "Berlin Chen"
citation_author: "Chao et al"
year: 2022
venue: "arXiv preprint"
source_pdf: "paper.pdf"
extraction_method: "Manual summary from arXiv PDF."
extracted_at: "2026-03-22"
llm_friendly: true
---

# 3M: An Effective Multi-view, Multi-granularity, and Multi-aspect Modeling Approach to English Pronunciation Assessment

## Metadata

- Authors: Fu-An Chao, Tien-Hong Lo, Tzu-I Wu, Yao-Ting Sung, Berlin Chen
- Affiliation: National Taiwan Normal University (NTNU), Taiwan
- arXiv: 2208.09110
- Task: Automatic Pronunciation Assessment (APA) on SpeechOcean762

## TL;DR

3M augments GOPT with three additions: prosodic features (duration + energy), a vowel/consonant positional embedding, and frozen SSL features from wav2vec 2.0 + HuBERT + WavLM Large. Concatenating all three SSL models outperforms any single one and raises phone-level PCC from 0.612 (GOPT baseline) to 0.656, and utterance total PCC from 0.742 to 0.796. This paper establishes the canonical 3-SSL feature set that MuFFIN later inherits directly.

## Abstract

Automatic pronunciation assessment (APA) has two practical obstacles: studies focus on segmental (phone-level) features like goodness of pronunciation (GOP), neglecting suprasegmental (prosodic) patterns; and labeled L2 speech data is scarce. The paper proposes 3M — a multi-view, multi-granularity, multi-aspect model — that addresses both by integrating prosodic and SSL features alongside GOP into the GOPT Transformer architecture. A vowel/consonant positional embedding encodes phonological structure without requiring additional labeled data, and SSL features provide contextualized acoustic representations from three pre-trained models. Experiments on SpeechOcean762 show significant improvements over GOPT, particularly for fluency and prosody scores.

## Problem Statement

GOPT captures phone-level segmental features well but misses two things: (1) duration and energy, which are key to stress, fluency, and prosody; (2) richer contextualized acoustic information beyond log-posterior-probability-based GOP. Multi-aspect scoring (phone/word/utterance) is already supported by GOPT, but utterance-level prosodic accuracy is poor. Resource scarcity limits training data. The paper asks: can multi-view acoustic features fix the prosodic gap without needing more labeled data?

## Method

### Architecture

3M extends GOPT (a Transformer encoder with five CLS tokens for utterance-level aspects, plus per-phone outputs for word and phone levels). The input to the encoder is a concatenation of:

```text
x = [E_gop, E_dur, E_eng, E_w2v2, E_hubert, E_wavlm]           (Eq. 4)
E_multi-view = Dense(x)  →  24-dim per phone                    (Eq. 5)
```

A vowel/consonant (VC) positional embedding is looked up from the canonical phone sequence and projected to the same 24-dim space, then added before the Transformer.

### Feature Dimensions

| Feature | Dim | Notes |
|---------|-----|-------|
| GOP (E_gop) | 84 | DNN-HMM TDNN-F, trained on LibriSpeech 960h with Kaldi |
| Duration (E_dur) | 1 | per-phone duration in seconds |
| Energy (E_eng) | 7 | RMSE stats: mean, std, median, mad, sum, max, min |
| wav2vec 2.0 (E_w2v2) | 1,024 | wav2vec2-large-xlsr-53, last layer, time-averaged over phone |
| HuBERT (E_hubert) | 1,024 | hubert-large-ll60k, last layer, time-averaged |
| WavLM (E_wavlm) | 1,024 | wavlm-large, last layer, time-averaged |
| **Total** | **3,160** | projected to 24-dim |

SSL features use dropout p_drop=0.1 before concatenation to prevent overfitting (due to the large dimensionality mismatch).

### Vowel/Consonant Positional Embedding

A simple lookup table maps each canonical phone to V (vowel) or C (consonant). This embeds phonological structure (stress typically occurs on vowels) as a learnable vector, improving the word-level stress score.

### Granularity and Aspects

- **Phone level**: one accuracy score per phone
- **Word level**: {accuracy, stress, total}
- **Utterance level**: {accuracy, fluency, completeness, prosody, total} — five CLS tokens

## Data

SpeechOcean762: 5,000 English utterances from 250 non-native Mandarin speakers (adults and children). 2,500 train / 2,500 test. Labels annotated by 5 experts, then averaged. Phone scores [0, 2]; word and utterance scores [0, 10] (normalized to [0, 2] for training by dividing by 5, following GOPT).

## Results

**Table 1 (main comparison, phone PCC ↑ / MSE ↓):**

| Model | Phone MSE ↓ | Phone PCC ↑ | Utt Total PCC ↑ |
|-------|-------------|-------------|-----------------|
| RF | 0.130 | 0.440 | — |
| SVR | 0.160 | 0.450 | — |
| LSTM | 0.089 | 0.591 | 0.741 |
| GOPT | 0.085 | 0.612 | 0.742 |
| Kim et al. (SSL) | — | — | 0.780 (fluency only) |
| **3M** | **0.078** | **0.656** | **0.796** |

**Table 2 (ablation, starting from GOPT baseline):**

| Setting | Phone PCC | Utt Fluency | Utt Prosody | Utt Total |
|---------|-----------|-------------|-------------|-----------|
| Baseline (GOPT) | 0.612 | 0.753 | 0.760 | 0.742 |
| + Vowel embed | 0.616 | 0.758 | 0.756 | 0.745 |
| + Duration | 0.620 | 0.769 | 0.766 | 0.747 |
| + Energy | 0.626 | 0.779 | 0.778 | 0.749 |
| + wav2vec 2.0 alone | 0.626 | 0.779 | 0.776 | 0.757 |
| + WavLM alone | 0.639 | 0.802 | 0.804 | 0.770 |
| + HuBERT alone | 0.635 | **0.830** | 0.819 | 0.793 |
| **3M (all)** | **0.656** | 0.828 | **0.827** | **0.796** |

WavLM gives the biggest phone PCC boost when used alone (+0.027 over GOPT); HuBERT gives the biggest fluency boost. Combining all three gives the best overall result. Note the stress score degrades slightly when adding SSL features (as with prosodic features) — the authors attribute this to the severely imbalanced distribution of word stress labels in SpeechOcean762.

## Key Design Decisions

1. **Last-layer SSL features only**: The paper uses only the final layer of each SSL model, averaged over the time frames within each phone segment using forced-alignment timestamps. This is the simplest possible extraction strategy and already yields large gains.

2. **All three SSL models are from HuggingFace**:
   - `facebook/wav2vec2-large-xlsr-53` (note: multilingual model, not `wav2vec2-large-960h`)
   - `facebook/hubert-large-ll60k`
   - `microsoft/wavlm-large`

3. **Simple concatenation, not learned fusion**: SSL features are concatenated with GOP and prosodic features, then passed through a single Dense projection layer. No attention-based fusion.

4. **Dropout on SSL features**: p_drop=0.1 applied to each SSL feature block independently before concatenation. The authors found this empirically necessary to prevent overfitting given the large SSL dimension vs. training set size.

## Limitations

- Only last-layer features are used — all 24 intermediate layers are discarded, leaving potential information on the table.
- The three SSL models are treated independently (concatenated), not jointly (no cross-model interaction).
- The vowel/consonant embedding is static (V or C), not fine-grained by phone class.
- The stress score degrades slightly — an open problem with the skewed label distribution.
- No MDD (mispronunciation detection) component.

## Relevance to Peacock (P010)

This paper is MuFFIN's direct ancestor on the SSL side. MuFFIN (Yan et al. 2025, same NTNU lab) inherits the exact same three-SSL-model feature set from 3M, including:

- The wav2vec2 + HuBERT + WavLM combination
- Phone-level averaging over alignment timestamps
- The `output_hidden_states=True` extraction approach

The key gap 3M identifies but does not close: using only the last SSL layer. This is exactly the gap our P010 CHConv contribution addresses — learning optimal aggregation across all 24 layers using hierarchical 1D convolution. The 3M ablation (Table 2) showing WavLM > HuBERT > wav2vec2 individually but all-three being best is also relevant: it motivates fusing all three rather than selecting one.

**Note on authorship**: The first author is Fu-An Chao (not "Do" — the user was misremembering). Berlin Chen co-authors both this paper and MuFFIN. This is the same lab.
