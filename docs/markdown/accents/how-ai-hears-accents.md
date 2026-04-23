# How AI Hears Accents

### An Audible Visualization of Accent Clusters

**By Oscar Friedman and Ilya Usorov**
**October 9, 2025**

---

Today, we're going to go on a tour of the world's accents in English. Users of [BoldVoice](https://boldvoice.com), the American accent training app, speak more than 200 different languages, and it is our mission to help them speak English clearly and confidently. While building the accent strength metric we covered in the [previous blog post](https://accent-strength.boldvoice.com/), we needed to understand how our models clustered accents, dialects, native languages, and language families. Today, we will share some of our findings using a 3D latent visualization.

---

## Technical Approach

To begin, we finetuned [HuBERT](https://arxiv.org/abs/2106.07447), a pretrained audio-only foundation model for the task of accent identification using our in-house dataset of non-native English speech and self-reported accents. BoldVoice's own dataset of accented speech is one of the largest of its kind in the world.

### HuBERT + Classification Head Architecture

```
Model: boldvoice/hubert-accent-identifier
Total Parameters: 94.6M (all trainable)

ARCHITECTURE:
═════════════

                ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌───────────────┐
Raw Audio  →    │  Feature    │  →   │  Feature    │  →   │ Transformer │  →   │ Classification│
(16kHz)         │  Extractor  │      │  Projection │      │   Encoder   │      │      Head     │
                └─────────────┘      └─────────────┘      └─────────────┘      └───────────────┘
                7 CNN layers         LayerNorm→Linear     12 layers            768→256→50
                1→512, 320x ↓        512→768, Dropout     12 heads, dim=768
                                                           (89.8M params)

KEY DETAILS:
• Input: Raw waveform (no spectrograms)
• Downsampling: 320x (5×2×2×2×2×2×2)
• Transformer: 12 layers
```

This model receives **only the raw input audio and associated accent label**; it gets neither a text prompt nor a transcript.

### Training Details

- **Dataset size:** 30 million speech recordings comprising 25,000 hours of English speech (a small fraction of their total accent dataset)
- **Fine-tuning approach:** All layers of the pretrained base model were unfrozen due to the large dataset size (unlike a traditional fine-tune)
- **Training hardware:** Cluster of A100 GPUs
- **Training duration:** Approximately one week
- **Live demo:** [accentoracle.com](https://accentoracle.com)

---

## The Visualization

To observe how accents cluster, an audible latent space visualization was produced for a small subset of recordings. Points on the graph show language labels on hover.

The visualization is created by applying [UMAP](https://arxiv.org/abs/1802.03426) dimensionality reduction to reduce the **768-dimensional latent space** to just **3 dimensions**.

### From Audio to Latent Visualization

```
  FROM AUDIO TO LATENT VISUALIZATION
  ══════════════════════════════════

     ╱│     ╱│     ╱│     ╱│                                z ↑
    ╱ │    ╱ │    ╱ │    ╱ │       ┌─────────────┐            │ ●
   ╱  │   ╱  │   ╱  │   ╱  │       │ ●●●●●○●●●●● │            │  ●  ●
 ─────│──╱───│──╱───│──╱───│──     │ ●●●●●○●●●●● │            │   ○
      │ ╱    │ ╱    │ ╱    │       │ ●●●●●○●●●●● │            └────────→ y
      │╱     │╱     │╱     │       └─────────────┘           ╱  ●  ●
       Speech Audio (16kHz)       768-dim embedding         ╱ ●    ●
                                    (mean pooled)        x ╱
                                                               3D [x,y,z]
            │                              │                Interactive Plot
            │                              │
            │      ┌───────────────┐       │      ┌───────────┐    │
            └─────→│     Model     │──────→└─────→│ UMAP(n=3) │───→┘
                   │   Inference   │              └───────────┘
                   └───────────────┘
```

### Notes on Methodology

- **UMAP trade-off:** UMAP destroys much of the information in the full-dimensional latent space, but roughly preserves the global structure, including relative distances between clusters.
- **One point = one recording** inferenced by the model after fine-tuning; color corresponds to the true accent label.
- **Cherry-picking:** Only points for which the predicted and target accents match are shown — the purpose is to understand accent placement relative to one another, not to assess raw model performance.

---

## Innovative Privacy Protection

By clicking or tapping on a point, you hear a **standardized version** of the corresponding recording. Voice standardization is used for two reasons:

1. **Privacy:** Anonymizes the speaker in the original recordings.
2. **Clarity:** Projects each accent onto a neutral voice, making it easier to hear accent differences while ignoring extraneous factors like gender, recording quality, and background noise.

> **Caveat:** This approach does not perfectly preserve the source accent and introduces some audible phonetic artifacts. The voice standardization model is an in-house accent-preserving voice conversion model.

---

## Cluster Highlights

> Our team was most surprised to see that **geographic proximity, immigration, and colonialism** seem to affect this model's learned accent groupings more than language taxonomy.

### 1. Australian–Vietnamese Bridge

The Australian cluster is right next to the Vietnamese cluster, despite English and Vietnamese being taxonomically unrelated. Listening to the ~10 bridge points reveals what sounds like native Vietnamese speakers who speak English with an Australian accent — likely a result of Vietnamese immigration to Australia. These hybrid accents may explain the overall proximity of the two clusters.

### 2. French–Nigerian–Ghanaian Cluster

A similar pattern appears with the French/Nigerian/Ghanaian grouping, where colonial history and language overlap appear to exert more influence than strict language-family taxonomy.

### 3. Indian Subcontinent Cluster

The Indian subcontinent accents form a coherent cluster with notable internal structure:

| Sub-cluster | Languages | Geography |
|---|---|---|
| Southern end | Telugu, Tamil, Malayalam | Southern India |
| Northern end | Nepali, Bengali | Northwest India & Nepal |

This internal arrangement roughly mirrors real-world geography.

### 4. Korean–Mongolian Cluster

The nearest cluster to the Mongolian cluster is Korean. This aligns with long-standing expert observations of phonetic similarities between the two languages. Notably, the once-proposed "Altaic language family" hypothesis (now refuted) historically grouped them together. The model — with no knowledge of language families — independently picked up on these phonetic similarities, even as filtered through English as a second language.

---

## Key Takeaway

> The distances on this map are not an objective measure of the phonetic similarity between accents. They are a byproduct of a model which has successfully learned to **distinguish a variety of accents in L2 English speech from audio alone with no knowledge of language or linguistics.**

This raises an open question about the Korean–Mongolian proximity: is it a meaningless artifact of latent space visualization, or evidence of real phonetic features diffusing between the two languages?

---

## Conclusion

This exploration highlights how a large-scale speech model captures the shared phonetic landscape of global English. By studying how different accents organize in the model's latent space, pronunciation tools can be made more accurate and more effective — reflecting BoldVoice's mission to help every English learner be understood and confident.

---

## Credits

- **Authors:** Oscar Friedman, Ilya Usorov
- **Dialect Coach:** Ron Carlos (in-house)
- **Published:** October 9, 2025
- **Contact:** <research@boldvoice.com>
- **Related:** [Accent Strength blog post](https://accent-strength.boldvoice.com/) · [Accent Oracle demo](https://accentoracle.com) · [BoldVoice](https://boldvoice.com)
- **Source:** [accent-explorer.boldvoice.com](https://accent-explorer.boldvoice.com)
