# Technical Reference: Pronunciation Assessment Methods

Quick reference guide for key technical concepts used in the research.

---

## 1. Goodness of Pronunciation (GOP)

### Original Definition (Witt & Young, 2000)
```text
GOP(l_i) = log p(l_i | O_t1^t2) / (t_2 - t_1)
```

Where:
- `l_i` = target phoneme
- `O_t1^t2` = acoustic observations in segment
- `t_2 - t_1` = segment duration (for normalization)

### What it measures:
- How well does the spoken segment match the expected phoneme?
- Higher score = better pronunciation
- Score based on acoustic model posteriors

### Key Limitation:
- **Requires segmentation**: Needs forced alignment to know where phoneme starts/ends
- **Fails on mispronunciations**: Alignment models trained on correct speech
- **Not alignment-agnostic**: Sensitive to exact segment boundaries

---

## 2. Self-Alignment GOP (GOP-SA)

### Innovation:
Use the same CTC model for both alignment AND GOP computation

### How it works:
1. Run forward-backward algorithm on CTC model
2. Find best alignment for target phoneme using model's own posteriors
3. Compute GOP on that alignment
4. No external forced aligner needed

### Advantages:
- Avoids misalignment from training model on correct speech
- Uses modern CTC-based ASR models
- Simpler than traditional GOP pipeline

### Formula:
```text
GOP-SA(l_i) = log p(l_i | O_T, model=CTC) / (t_2 - t_1)
```

Where alignment (t_2 - t_1) comes from CTC model itself

---

## 3. Segmentation-Free GOP (GOP-SF)

### Core Innovation:
**Don't commit to a specific segmentation. Consider all possible segmentations.**

### How it works:
```text
GOP-SF(l_i) = log p(l_i | O_T, L_L, L_R)
```

Where:
- `O_T` = full utterance
- `L_L` = left context (phonemes before target)
- `L_R` = right context (phonemes after target)
- No explicit alignment boundaries

### Key Insight:
Instead of computing likelihood for specific segment (t_1, t_2):
```text
p(l_i | O_t1^t2)  <- OLD: specific segment
```

Compute likelihood given full utterance AND context:
```text
p(l_i | O_T, L_L, L_R)  <- NEW: all possible alignments
```

### Mathematical Foundation:
```text
p(l_i | O_T, L_L, L_R) =
    sum_all_alignments p(L_L, l_i, L_R, alignment | O_T)
    -----------------------------------------------------
    sum_all_alignments p(L_L, L_R, alignment | O_T)
```

### Computation via CTC:
```typescript
Numerator:   = CTC_loss(L_canonical, O_T)
Denominator: = CTC_loss(L_SDI, O_T)

where L_SDI allows:
  - Substitution: any phoneme instead of target
  - Deletion: skip target entirely
  - Insertion: insert any phoneme instead
```

### Advantages:
- Handles segmentation uncertainty automatically
- Works with peaky CTC models
- Can detect insertion/deletion errors (not just substitutions)
- Incorporates full utterance context

---

## 4. CTC: Connectionist Temporal Classification

### What is CTC?
Neural network layer for sequence labeling without frame-level annotations

### Key Insight:
- Maps variable-length input sequence to variable-length output sequence
- Uses special "blank" symbol for alignment
- Loss computed over all possible alignments

### CTC Loss:
```yaml
L_CTC(L) = -log sum p(pi | X)
           pi in Pi(B(pi)=L)

where:
- pi = alignment path (input length T)
- B(pi) = collapse function (remove blanks and repeats)
- L = target label sequence
```

### Forward-Backward Algorithm:
```text
alpha_t(s) = forward variable at timestep t, state s
beta_t(s) = backward variable at timestep t, state s

Used to:
1. Compute probability of path
2. Assign credit to each timestep
3. Extract expected durations
```

---

## 5. CTC Peaky Behavior

### Problem: "Peakiness"

#### Peaky Over Time (POT)
- Blank symbol active most timesteps
- Non-blank symbols active only briefly
- Poor temporal resolution

#### Peaky Over State (POS)
- Output distribution very concentrated
- One output dominates, others near-zero
- Entropy very low

### Why This Matters:
```text
Traditional GOP: uses Viterbi path (best single alignment)
-> Only one frame gets credited to each phoneme
-> Loses information about timing uncertainty

GOP-SF: considers all paths (forward-backward)
-> Distributes credit across possible alignments
-> Better handles peaky behavior
```

### Measurement:
```text
Blank Coverage (BC):
  BC = mean probability of blank symbol
  High BC = very peaky over time

Conditional Entropy (ConEn):
  ConEn = entropy of alignment distribution
  Low ConEn = very peaky over symbols
```

---

## 6. Forced Alignment

### What is it?
Automatically segment speech into phonemes using ASR model

### Traditional Process:
```text
1. HMM-GMM model trained on correct speech
2. Viterbi decoding finds best path
3. Extract timesteps for each phoneme
4. Use these boundaries for GOP
```

### Problems:
- **Mispronunciation mismatch**: Model trained on correct speech fails on errors
- **Coarticulation ambiguity**: Where does /w/ start and /a/ end?
- **Peaky CTC alignment**: CTC not designed for time alignment
- **Speaker variability**: Age, accent, dialect affect timing

### Segmentation-Free Solution:
- Don't rely on external forced aligner
- Use CTC's own alignment probabilities
- Or use all possible alignments (GOP-SF)

---

## 7. Feature Extraction from GOP

### Single Feature (Traditional)
```text
GOP_score = single numeric value [-inf to 0 or normalized]
```

### Feature Vector Approach
```text
FGOP(l_i) = {LPP, LPR(l_i)}

where:
  LPP = log p(L_canonical | O_T)
  LPR = log posterior ratio vector:
      LPR(i) = log p(L_canonical | O) / log p(L_with_error_i | O)
```

### With Normalization
```python
FGOP-Norm(l_i) = {LPP, LPR(l_i), E[duration]}

E[duration] = expected length of target phoneme activations
             from forward algorithm
```

### Usage:
Feed these features into classifier:
- Polynomial regression
- Support Vector Machine (SVM)
- Neural network
- GOPT (custom model)

---

## 8. Evaluation Metrics

### For Binary Tasks (Correct vs. Mispronounced)
```markdown
AUC-ROC: Area Under Receiver Operating Characteristic
  - Threshold-independent
  - Good for imbalanced datasets
  - Ranges 0-1 (0.5 = random, 1.0 = perfect)
```

### For Continuous Scores (Quality Rating)
```yaml
PCC: Pearson Correlation Coefficient
  - Correlation between predicted and true scores
  - Ranges -1 to 1 (0 = uncorrelated, 1 = perfect)
  - Used in speechocean762 evaluation
```

### For Sequence Tasks (Character/Word Error Rate)
```yaml
PER: Phoneme Error Rate
  - Edit distance on phoneme sequences
  - Measures recognition accuracy (not pronunciation)
```

---

## 9. Dataset Characteristics

### CMU Kids
```yaml
Dataset: 9.1 hours of speech
Speakers: 63 females + 24 males, ages 6-11
Utterances: 5,180 sentences
Labels: Marked with mispronunciation errors
Error Distribution:
  - 90.2% correct
  - 3.2% substitution
  - 2.3% deletion
  - 4.5% insertion
Task: Binary (Correct vs. Mispronounced)
```

### speechocean762
```yaml
Dataset: 5,000 utterances
Speakers: 2,670 speakers
Age Distribution: 50% children (5-15), 50% adults (12-43)
Language: English (non-native)
Labels: 3-way: Mispronounced(0), Strongly Accented(1), Correct(2)
Task: Ternary classification
Annotation: Phoneme-level
```

---

## 10. Key Equations at a Glance

### Traditional GOP
```text
GOP(l_i) = log p(l_i | O_t1^t2) / (t_2 - t_1)
```

### GOP with Normalization
```text
GOP_norm(l_i) = log p(l_i | O_t1^t2) / ln(t_2 - t_1)
```

### Segmentation-Free GOP
```text
GOP-SF(l_i) = log p(l_i | O_T, L_L, L_R)
```

### Segmentation-Free GOP with Normalization
```text
GOP-SF-Norm(l_i) = log p(l_i | O_T, L_L, L_R) / E[t_2 - t_1]
```

### CTC Loss
```text
L_CTC(L) = -log sum p(pi | X)  [summed over all paths pi]
```

### Expected Duration (from Forward-Backward)
```text
E[t_2 - t_1] = sum_t (normalized forward variable at nodes for l_i)
```

### Feature Vector
```text
FGOP-SF-Norm(l_i) = {LPP, LPR(l_i), E[t_2 - t_1]}
```

---

## 11. Common Acronyms

| Acronym | Meaning | Context |
|---------|---------|---------|
| **ASR** | Automatic Speech Recognition | Speech-to-text |
| **CAPT** | Computer-Assisted Pronunciation Training | Educational systems |
| **CALL** | Computer-Aided Language Learning | Language learning |
| **CTC** | Connectionist Temporal Classification | Neural network loss |
| **DNN** | Deep Neural Network | Traditional acoustic model |
| **E2E** | End-to-End | Single model without pipeline |
| **GMM-HMM** | Gaussian Mixture Model - Hidden Markov Model | Traditional ASR |
| **GOP** | Goodness of Pronunciation | Scoring method |
| **HMM** | Hidden Markov Model | Statistical model for sequences |
| **L1** | First Language | Native language |
| **L2** | Second Language | Foreign language learning |
| **LPP** | Log Posterior Probability | Feature |
| **LPR** | Log Posterior Ratio | Feature vector |
| **MDD** | Mispronunciation Detection & Diagnosis | Task |
| **OOV** | Out-of-Vocabulary | Unknown words |
| **PCC** | Pearson Correlation Coefficient | Evaluation metric |
| **PER** | Phoneme Error Rate | Evaluation metric |
| **POT** | Peaky Over Time | CTC characteristic |
| **POS** | Peaky Over State | CTC characteristic |
| **ROC** | Receiver Operating Characteristic | Evaluation curve |
| **SVM** | Support Vector Machine | Classifier |
| **TDNN** | Time Delay Neural Network | Acoustic model |
| **UQ** | Uncertainty Quantification | Estimating confidence |

---

## 12. Workflow Comparison

### Traditional GOP Workflow
```text
Utterance Audio
       |
[Forced Alignment (HMM-GMM)]
       |
Get segment boundaries (t_1, t_2)
       |
[Acoustic Model (HMM-GMM)]
       |
Compute posterior: p(l_i | O_t1^t2)
       |
Normalize by duration: (t_2 - t_1)
       |
GOP Score
```

### GOP-SA Workflow
```text
Utterance Audio
       |
[CTC-based ASR Model]
    /          \
Alignment    Posteriors
(Viterbi)    from same model
    \          /
Get boundaries + compute GOP
       |
GOP Score
```

### GOP-SF Workflow
```text
Utterance Audio + Canonical Transcription
       |
[CTC Forward-Backward Algorithm]
       |
Consider ALL possible segmentations
Weighted by alignment probability
       |
Marginalize over all alignments
       |
GOP Score (alignment-free)
```

---

## 13. Performance Improvements Summary

### Results (CMU Kids, Real Errors)
```text
Method              AUC    Improvement
-----------------------------------------
GOP-DNN-Avg         0.796  (baseline)
GOP-CTC-SA          0.908  +14.0%
GOP-CTC-SF-SD       0.914  +14.8% <- BEST
GOP-CTC-SF-S        0.891  +12.1%
```

### Key Insight:
- Self-alignment (SA) beats traditional approach significantly
- Segmentation-free (SF) with deletion+substitution (SD) performs best
- Allows insertions (SDI) performs worse

---

## 14. When to Use Each Method

### Traditional GOP
- Works well with HMM-GMM models
- When you have good forced aligner
- Struggles with mispronunciations
- Not compatible with modern CTC models

### GOP-SA
- Use CTC models for ASR
- Better on mispronounced speech
- Simpler than traditional GOP
- Still somewhat alignment-dependent

### GOP-SF
- Most robust to alignment errors
- Handles insertion/deletion errors
- Works with any peakiness level
- Best performance on benchmarks
- Higher computational cost than GOP-SA
- Makes uniform alignment assumption

---

## 15. Implementation Notes

### Computational Complexity
```text
Traditional GOP:   O(T x |L_C|)      [Linear in utterance length]
GOP-SA:            O(T x |L_C|)      [Same as traditional]
GOP-SF:            O(T x |L_C| + |V|) [Add vocabulary size term]
```

Where:
- T = utterance length (frames)
- |L_C| = canonical transcription length
- |V| = vocabulary size

### Numerical Stability
- Paper mentions numerical issues in GOP-CE-SF
- Solution: Use log-domain computations throughout
- Forward-backward algorithm already in log-space
- Avoid subtracting large log probabilities

### Implementation Tip:
```python
# DON'T:
p_num = exp(log_p_num)
p_den = exp(log_p_den)
gop = log(p_num / p_den)

# DO:
gop = log_p_num - log_p_den
```

---

*This reference guide complements 01_START_HERE.md. Use it while reading papers.*
