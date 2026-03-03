# Start Here: Pronunciation Assessment Research

Entry point for the research papers, reading curriculum, and full bibliography.

**Core Paper**: Segmentation-Free Goodness of Pronunciation (arXiv:2507.16838v3)

**Total Papers**: 29 unique papers across 7 research areas
**Year Range**: 2020--2026
**Estimated Reading Time**: 20--30 hours for all papers; 5--8 hours for core papers

### Paper Stats

| Year | Count | Notes |
|------|-------|-------|
| 2020 | 2 | CaGOP, early L2 work |
| 2021 | 5 | CoCA-MDD, PTeacher, datasets |
| 2022 | 4 | Wav2vec improvements, CAPT |
| 2023 | 2 | Reviews, dysarthria work |
| 2024 | 2 | Self-supervised approaches |
| 2025 | 9 | Recent wave of CTC+GOP work |
| 2026 | 1 | Latest (Jan 2026) |

### Most Recent (Last 3 Months)

- 2601.14744 -- Large audio-language models (Jan 2026)
- 2509.03256 -- Speech assessment model comparison (Sep 2025)
- 2509.02915 -- LoRA fine-tuning (Sep 2025)
- 2508.03937 -- LCS-CTC (Aug 2025)
- 2506.02080 -- CTC GOP improvements (Jun 2025)

---

## Get Papers Fast

### Method 1: Automatic (Recommended)

```bash
cd /Users/chiejimofor/Documents/Github/gopt

# Downloads organized into folders by research area
python3 download_papers.py

# Or specify custom directory
python3 download_papers.py --output-dir my_papers

# Quiet mode (no status updates)
python3 download_papers.py --quiet
```

**Result**: Creates `pronunciation_papers/` with folders:
```text
pronunciation_papers/
  core_segmentation_free/          (4 papers)
  mispronunciation_detection/      (8 papers)
  gop_methods/                     (4 papers)
  capt_systems/                    (7 papers)
  end_to_end_assessment/           (4 papers)
  child_speech_l2/                 (2 papers)
  datasets/                        (1 paper)
```

### Method 2: Manual (If Python Fails)

```bash
# Download just a few key papers
curl -o 2507.16838.pdf https://arxiv.org/pdf/2507.16838.pdf
curl -o 2310.13974.pdf https://arxiv.org/pdf/2310.13974.pdf
curl -o 2506.02080.pdf https://arxiv.org/pdf/2506.02080.pdf
```

### "I Just Want the Key Papers" (7 total)

```bash
for id in 2507.16838 2310.13974 2506.02080 2508.03937 \
          2206.07289 2203.15937 2104.01378; do
  curl -o "${id}.pdf" "https://arxiv.org/pdf/${id}.pdf"
done
```

### Troubleshooting

If the download script does not work:
```bash
# Try manual download with curl
curl -L -o 2507.16838.pdf https://arxiv.org/pdf/2507.16838.pdf

# Or visit arXiv directly:
# https://arxiv.org/pdf/2507.16838.pdf
```

---

## Paper Catalog

### Core Research (Segmentation-Free Methods)

Papers directly building on or extending segmentation-free approaches, CTC-based GOP, and alignment-free methods.

| Title | Authors | ID | Year | Key Contribution |
|-------|---------|-----|------|-----------------|
| Segmentation-Free Goodness of Pronunciation | Cao, Fan, Svendsen, Salvi | 2507.16838 | 2025 | **Main paper** -- GOP-SA & GOP-SF methods using CTC |
| Enhancing GOP in CTC-Based Mispronunciation Detection | - | 2506.02080 | 2025 | Improvements to GOP for CTC models |
| Towards Accurate Phonetic Error Detection Through Phoneme | - | 2507.14346 | 2025 | Phonetic error detection via phoneme-level assessment |
| LCS-CTC: Leveraging Soft Alignments to Enhance Phonetic | - | 2508.03937 | 2025 | Soft alignment methods for CTC speech recognition |

### Mispronunciation Detection & Diagnosis

| Title | Authors | ID | Year | Focus |
|-------|---------|-----|------|-------|
| Mispronunciation Detection and Diagnosis Without Model Training | - | 2511.20107 | 2025 | Zero-shot/few-shot MDD without retraining |
| Text-Aware End-to-end Mispronunciation Detection and Diagnosis | - | 2206.07289 | 2022 | End-to-end with text information |
| End-to-End Mispronunciation Detection and Diagnosis From Raw | - | 2103.03023 | 2021 | Direct from raw speech |
| Phonological Level wav2vec2-based Mispronunciation Detection | - | 2311.07037 | 2023 | Attribute-based error detection |
| An End-to-End Mispronunciation Detection System for L2 English | - | 2005.11950 | 2020 | L2 English specific system |
| CoCA-MDD: A Coupled Cross-Attention based | - | 2111.08191 | 2021 | Cross-attention architecture |
| A Full Text-Dependent End to End Mispronunciation Detection | - | 2104.08428 | 2021 | Text-dependent approach |

### Goodness of Pronunciation (GOP) Methods

| Title | Authors | ID | Year | Method |
|-------|---------|-----|------|--------|
| Goodness of Pronunciation Pipelines for OOV Problem | - | 2209.03787 | 2022 | Handling out-of-vocabulary words |
| Context-aware Goodness of Pronunciation for Computer-Assisted | - | 2008.08647 | 2020 | Context-aware GOP (CaGOP) |
| Speech Intelligibility Assessment of Dysarthric Speech | - | 2305.18392 | 2023 | Uncertainty quantification in GOP |
| Evaluating Logit-Based GOP Scores for Mispronunciation Detection | - | 2506.12067 | 2025 | Alternative to softmax posteriors |

### Computer-Assisted Pronunciation Training (CAPT)

| Title | Authors | ID | Year | Focus |
|-------|---------|-----|------|-------|
| MuFFIN: Multifaceted Pronunciation Feedback Model | - | 2510.04956 | 2025 | Multi-faceted feedback system |
| Towards Efficient and Multifaceted Computer-assisted Pronunciation | - | 2502.07575 | 2025 | Efficient multi-task CAPT (HMamba) |
| PTeacher: a Computer-Aided Personalized Pronunciation Training | - | 2105.05182 | 2021 | Personalized audio-visual feedback |
| JCAPT: A Joint Modeling Approach for CAPT | - | 2506.19315 | 2025 | Joint modeling for pronunciation feedback |
| Computer-assisted Pronunciation Training -- Speech synthesis | - | 2207.00774 | 2022 | Speech synthesis for corrective feedback |
| Unlocking Large Audio-Language Models for CAPT | - | 2601.14744 | 2026 | Large models for pronunciation training |
| V(is)owel: An Interactive Vowel Chart to Understand | - | 2507.06202 | 2025 | Interactive visualization for CAPT |

### End-to-End Pronunciation Assessment

| Title | Authors | ID | Year | Approach |
|-------|---------|-----|------|----------|
| Improving Mispronunciation Detection with Wav2vec2 | - | 2203.15937 | 2022 | Wav2Vec2 foundation model |
| Comparison of End-to-end Speech Assessment Models | - | 2509.03256 | 2025 | Comparative analysis of models |
| Automatic Pronunciation Assessment using Self-Supervised Speech | - | 2204.03863 | 2024 | Self-supervised learning approach |
| Automatic Pronunciation Assessment - A Review | - | 2310.13974 | 2023 | Comprehensive review of methods |
| LoRA Fine-tuned Speech Multimodal LLM | - | 2509.02915 | 2025 | Large language models for assessment |

### Child Speech & L2 Learners

| Title | Authors | ID | Year | Focus |
|-------|---------|-----|------|-------|
| Joint Pronunciation Transcription and Feedback for Evaluating Kids | - | 2507.03043 | 2025 | Kid-specific evaluation system |
| Self-Supervised Models for Phoneme Recognition | - | 2503.04710 | 2025 | Child speech phoneme recognition in French |

### Self-Supervised & Foundation Models

| Title | ID | Year | Model |
|-------|-----|------|-------|
| Automatic Pronunciation Assessment using Self-Supervised Speech | 2204.03863 | 2024 | SSL models fine-tuning |
| Self-Supervised Models for Phoneme Recognition | 2503.04710 | 2025 | wav2vec 2.0, HuBERT, WavLM |

### Datasets & Resources

| Title | Authors | ID | Year | Dataset |
|-------|---------|-----|------|---------|
| An Open-Source Non-native English Speech Corpus For Pronunciation Assessment | Zhang et al. | 2104.01378 | 2021 | **speechocean762** (5000 utterances, 2670 speakers) |

**Note**: The main paper (2507.16838) also uses:
- **CMU Kids** (9.1 hours, children 6-11 years, 5180 sentences)
- **speechocean762** (mentioned above)

### Bulk Download Script

```bash
mkdir -p pronunciation_papers
cd pronunciation_papers

arxiv_ids=(
  2507.16838  # Main paper
  2511.20107 2206.07289 2103.03023 2311.07037 2005.11950
  2111.08191 2203.15937 2104.08428 2209.03787 2008.08647
  2305.18392 2506.12067 2506.02080 2509.03256 2508.03937
  2507.14346 2310.13974 2207.00774 2510.04956 2502.07575
  2105.05182 2506.19315 2204.03863 2601.14744 2507.06202
  2104.01378 2507.03043 2503.04710 2509.02915
)

for id in "${arxiv_ids[@]}"; do
  echo "Downloading $id..."
  curl -o "${id}.pdf" "https://arxiv.org/pdf/${id}.pdf"
  sleep 1  # Be respectful to arXiv servers
done
```

### Direct arXiv Links by Category

**Core Segmentation-Free Methods**
- https://arxiv.org/pdf/2507.16838.pdf -- Main paper
- https://arxiv.org/pdf/2506.02080.pdf -- Enhancing GOP in CTC
- https://arxiv.org/pdf/2507.14346.pdf -- Phonetic Error Detection
- https://arxiv.org/pdf/2508.03937.pdf -- LCS-CTC

**Mispronunciation Detection**
- https://arxiv.org/pdf/2511.20107.pdf -- MDD Without Model Training
- https://arxiv.org/pdf/2206.07289.pdf -- Text-Aware End-to-End
- https://arxiv.org/pdf/2103.03023.pdf -- E2E from Raw Speech
- https://arxiv.org/pdf/2311.07037.pdf -- Phonological Level wav2vec2
- https://arxiv.org/pdf/2005.11950.pdf -- End-to-End L2 English
- https://arxiv.org/pdf/2111.08191.pdf -- CoCA-MDD
- https://arxiv.org/pdf/2203.15937.pdf -- Improving with Wav2vec2
- https://arxiv.org/pdf/2104.08428.pdf -- Full Text-Dependent

**GOP Methods**
- https://arxiv.org/pdf/2209.03787.pdf -- GOP for OOV
- https://arxiv.org/pdf/2008.08647.pdf -- Context-Aware GOP
- https://arxiv.org/pdf/2305.18392.pdf -- GOP with UQ for Dysarthric
- https://arxiv.org/pdf/2506.12067.pdf -- Logit-Based GOP

**CAPT Systems**
- https://arxiv.org/pdf/2510.04956.pdf -- MuFFIN
- https://arxiv.org/pdf/2502.07575.pdf -- HMamba
- https://arxiv.org/pdf/2105.05182.pdf -- PTeacher
- https://arxiv.org/pdf/2506.19315.pdf -- JCAPT
- https://arxiv.org/pdf/2207.00774.pdf -- Speech Synthesis for CAPT
- https://arxiv.org/pdf/2601.14744.pdf -- Large Audio-Language Models
- https://arxiv.org/pdf/2507.06202.pdf -- V(is)owel

**End-to-End Assessment**
- https://arxiv.org/pdf/2509.03256.pdf -- Comparison of Models
- https://arxiv.org/pdf/2310.13974.pdf -- Review Paper
- https://arxiv.org/pdf/2204.03863.pdf -- Self-Supervised
- https://arxiv.org/pdf/2509.02915.pdf -- LoRA Fine-tuned LLM

**Child Speech & L2**
- https://arxiv.org/pdf/2507.03043.pdf -- Joint Transcription for Kids
- https://arxiv.org/pdf/2503.04710.pdf -- Self-Supervised for Child Speech

**Datasets**
- https://arxiv.org/pdf/2104.01378.pdf -- speechocean762 Dataset

---

## Reading Path

A 5-phase reading plan designed for 6--8 weeks of focused reading. Adjust based on your background and goals.

### Phase 1: Foundation (Weeks 1--2)

Start with these core papers to understand the field fundamentals.

**1. Your Main Paper (Essential)**
- **Paper**: Segmentation-Free Goodness of Pronunciation (2507.16838)
- **Time**: 2--3 hours
- **Focus**: Read carefully, especially:
  - Section II-B: Alignment issues (why segmentation-free matters)
  - Section II-C: CTC peaky behavior (why it is a problem)
  - Section III: Methods (GOP-SA and GOP-SF definitions)
  - Section IV-V: Experiments (results comparison)

**2. Review Paper (Context)**
- **Paper**: Automatic Pronunciation Assessment -- A Review (2310.13974)
- **Time**: 1.5--2 hours
- **Why**: Broad overview of the field, historical context
- **Focus**: Understand different approaches and their evolution

**3. Original GOP Paper (Historical)**
- Referenced as [1]: Witt & Young (2000) in your paper
- Available in references, but understanding the original GOP concept is essential
- The paper explains it well in Section II-A

### Phase 2: CTC & Alignment (Weeks 2--3)

Understand the technical challenges your paper addresses.

**4. LCS-CTC (2508.03937)**
- **Time**: 1.5 hours
- **Focus**: Alternative soft alignment approaches for CTC
- **Connection**: Parallel solution to similar alignment problems

**5. Enhancing GOP in CTC-Based MDD (2506.02080)**
- **Time**: 1.5 hours
- **Connection**: Directly extends work on CTC+GOP
- **Compare**: With your paper's methods

**6. Towards Accurate Phonetic Error Detection (2507.14346)**
- **Time**: 1 hour
- **Focus**: Phoneme-level error detection (similar goal)
- **Connection**: Published same month as your paper

### Phase 3: Applications & Methods (Weeks 3--4)

See how the core ideas are applied in practice.

**7. Choose 2--3 from Mispronunciation Detection**
- 2203.15937: Improving with Wav2vec2 (foundational models)
- 2311.07037: Phonological Level detection (attribute-based)
- 2206.07289: Text-Aware E2E (including text context)
- Time per paper: 1--1.5 hours
- Why: Different approaches to the same problem

**8. Choose 1--2 from CAPT Systems**
- 2510.04956 (MuFFIN): Multi-faceted feedback
- 2502.07575 (HMamba): Efficient multi-task system
- 2105.05182 (PTeacher): Interactive feedback with visuals
- Time per paper: 1--1.5 hours
- Why: Real-world applications of pronunciation assessment

### Phase 4: Specialized Topics (Weeks 5--6)

Deep dive into specific areas of interest.

**Child Speech & L2 Learners**
- 2507.03043: Joint transcription for kids (newest)
- 2503.04710: Self-supervised for child speech
- 2005.11950: L2 English system (foundational)

**Foundation Models & Self-Supervised Learning**
- 2204.03863: Self-supervised pronunciation assessment
- 2509.02915: LoRA fine-tuning for large models
- 2601.14744: Unlocking large audio-language models

**Alternative GOP Methods**
- 2008.08647: Context-aware GOP (CaGOP)
- 2209.03787: GOP for out-of-vocabulary words
- 2506.12067: Logit-based alternatives to softmax

### Phase 5: Datasets & Resources (Throughout)

Reference these to understand benchmark datasets.

**Essential Datasets**
- **speechocean762** (2104.01378): 5000 utterances, 2670 speakers, non-native English
- **CMU Kids**: Referenced in your main paper (9.1 hours, children 6-11)

**Key Characteristics**
- Both have ternary labels (correct, accented, mispronounced) or binary
- Ground truth annotations available
- Widely used for benchmarking

### Week-by-Week Schedule

**Week 1--2: Foundation**
```text
Day 1-2:   Your main paper (2507.16838) - Deep read
Day 3-4:   Review paper (2310.13974) - Skim & reference
Day 5-7:   LCS-CTC (2508.03937) & Enhancing GOP (2506.02080)
```

**Week 3--4: Methods**
```text
Day 8-9:   Phonetic Error Detection (2507.14346)
Day 10-11: Pick 2 MDD papers (e.g., 2203.15937, 2311.07037)
Day 12-14: Pick 2 CAPT papers (e.g., 2510.04956, 2502.07575)
```

**Week 5--6+: Deep Dives**
```text
Choose one specialization based on your interests:
- Child speech development
- CAPT system design
- Foundation model adaptation
- Alternative scoring methods
```

---

## Paper Relationships

### Direct Dependencies (Read in Order)

```text
Witt & Young (2000) [Original GOP]
  |
  v
Your Paper: Segmentation-Free GOP (2507.16838)
  |
  v
Related Extensions:
  - Enhancing GOP in CTC (2506.02080)
  - LCS-CTC (2508.03937)
  - Phonetic Error Detection (2507.14346)
```

### Parallel Development (Read Together for Comparison)

```text
MDD Methods:
  - Text-Aware E2E (2206.07289)
  - Improving with Wav2vec2 (2203.15937)
  - Phonological Level (2311.07037)

CAPT Applications:
  - MuFFIN (2510.04956)
  - HMamba (2502.07575)
  - PTeacher (2105.05182)
```

### Foundation (Read for Background)

```text
Self-Supervised Learning:
  - wav2vec2.0 (referenced as [39])
  - HuBERT (referenced as [11])

CTC & Alignment:
  - CTC paper (referenced as [13])
  - Other CTC improvements ([29-31])
```

---

## Key Themes & Research Gaps

### Key Themes

**1. Alignment Problem**
- Traditional GOP requires pre-segmentation via forced alignment
- Mispronunciations make alignment unreliable
- CTC models have poor alignment information ("peaky behavior")
- Solution: Segmentation-free methods (main paper's contribution)

**2. CTC Challenges**
- CTC models exhibit "peakiness over time" (POT) and "peakiness over state" (POS)
- Not designed for alignment-critical tasks
- Recent work (2506.02080, 2507.14346) explores how to use CTC for GOP

**3. Foundation Models in Speech**
- wav2vec2.0, HuBERT, WavLM becoming standard
- Better phonetic recognition than traditional HMM-GMM
- Enable end-to-end approaches without explicit alignment

**4. End-to-End Revolution**
- Moving away from pipeline approaches (ASR then GOP calculation)
- Direct pronunciation assessment from speech
- Multi-task learning (MDD + other tasks)

**5. Child Speech Challenges**
- Different acoustic characteristics than adult speech
- Mispronunciations more common in learners
- Datasets like CMU Kids and speechocean762 essential

**6. Feedback & Interaction**
- Beyond binary classification toward detailed feedback
- Audio-visual approaches (PTeacher)
- Interactive visualization (V(is)owel)
- Multi-faceted feedback (MuFFIN)

### Research Gaps & Future Directions

Based on the papers reviewed:

1. **Real-time processing** -- Most papers focus on batch evaluation
2. **Personalization** -- Limited work on learner-specific models
3. **Multi-lingual** -- Mostly English-focused; more work needed for other languages
4. **Explainability** -- Why models make certain assessments (partially addressed)
5. **Interaction design** -- How to best present feedback to learners
6. **Cross-lingual transfer** -- Using models trained on one language for another
7. **Dysarthria & speech disorders** -- Growing area (2305.18392)

### Literature Review Stats

- **Total papers identified**: 29
- **Year range**: 2020--2026
- **Most recent wave**: 2025--2026 (14 papers)
- **Core focus areas**: MDD (7), GOP methods (4), CAPT (7), Assessment (5)
- **Largest dataset**: speechocean762 (5000 utterances)

---

## Quick Reference by Interest

**CTC & Alignment Issues**
- Start: 2508.03937, 2506.02080, 2507.14346

**Mispronunciation Detection**
- Start: 2206.07289, 2203.15937, 2311.07037

**CAPT Systems**
- Start: 2510.04956, 2502.07575, 2105.05182

**Child Speech**
- Start: 2507.03043, 2503.04710

**Self-Supervised Models**
- Start: 2204.03863, 2509.02915

**Alternative GOP Methods**
- Start: 2008.08647, 2209.03787, 2506.12067

**Review & Overview**
- Start: 2310.13974

### Common Tasks

**"I want to understand the technical details"**
1. Read `02_CONCEPTS.md` (30 min)
2. Read your paper section-by-section (2--3 hours)
3. Read 2506.02080, 2508.03937, 2507.14346 (parallel work)

**"I want to build a CAPT system"**
1. Read 2507.16838 (methods)
2. Read 2310.13974 (overview)
3. Read 2510.04956, 2502.07575, 2105.05182 (systems)

**"I'm interested in child speech"**
1. Read 2507.16838 (methods use CMU Kids)
2. Read 2507.03043 (kids specific)
3. Read 2503.04710 (child phoneme recognition)

**"I don't understand CTC"**
- Read `02_CONCEPTS.md` Section 4
- Then read your paper Section II-C

**"What's the difference between GOP-SA and GOP-SF?"**
- `02_CONCEPTS.md` Sections 2--3
- Your paper Section III-A and III-B
- Quick answer: SA uses same model for alignment; SF considers all alignments

---

## Concepts to Master

As you read, track understanding of these core concepts.

### Technical Concepts

- [ ] **CTC (Connectionist Temporal Classification)**: What is it? Why does it cause "peaky" behavior?
- [ ] **Forced Alignment**: How does it work? What are its failure modes?
- [ ] **Posteriorgram / Posterior Probability**: What does it represent? How is it computed?
- [ ] **Goodness of Pronunciation (GOP)**: Original definition and variants
- [ ] **Segmentation Problem**: Why is it hard? What does segmentation-free solve?
- [ ] **Self-Alignment**: How does using same model for alignment help?
- [ ] **Normalization**: Why normalize by activation length instead of segment length?

### Methodological Concepts

- [ ] **Binary vs. Ternary Classification**: Different task formulations
- [ ] **Feature Engineering**: How GOP scores become features for classifiers
- [ ] **End-to-End vs. Pipeline**: Trade-offs between approaches
- [ ] **Foundation Models**: Pre-training + fine-tuning paradigm
- [ ] **Cross-lingual Transfer**: Using models across languages

### Practical Concepts

- [ ] **Threshold Selection**: How to convert scores to binary/ternary decisions
- [ ] **Evaluation Metrics**: AUC-ROC, PCC, PER, etc.
- [ ] **Dataset Characteristics**: What makes a good benchmark dataset
- [ ] **Generalization**: Performance on different speaker types/ages

---

## Questions to Answer While Reading

### For Your Main Paper (2507.16838)

1. What are the two main problems with traditional GOP?
2. How does GOP-SA differ from traditional GOP? When does it help most?
3. What is the key insight of GOP-SF? Why does it not need explicit segmentation?
4. What is the "uniform alignment distribution" assumption? How reasonable is it?
5. How does the normalization in Eq. 6 differ from standard GOP normalization?
6. What is the "peakiness" problem in CTC, and how do the methods address it?

### For MDD Papers

1. What is the task definition (binary vs. ternary vs. multi-level)?
2. How does the method handle mispronunciation detection?
3. What datasets are used? How do results compare?
4. What is novel about this approach compared to prior work?

### For CAPT Papers

1. What feedback modalities are used (text, audio, visual)?
2. How is personalization handled?
3. What is the full system pipeline?
4. What aspects of pronunciation are addressed?

---

## Study Tips

### Note-Taking

Use Cornell Note-Taking System:
- Left column: Key concepts
- Right column: Detailed notes
- Bottom: Summary & questions

### For Managing References

```bash
# Export BibTeX from arXiv
# For paper 2507.16838:
# Click "BibTeX" on https://arxiv.org/abs/2507.16838
```

### Common Challenges & Solutions

**"The Math is Too Dense"**
- Focus on intuition first (read abstract, intro, conclusion)
- Then go back for mathematical details
- Use diagrams/figures as guides to understanding

**"Too Many Papers to Read"**
- Phase 1 alone provides a solid foundation
- Do not try to read everything immediately
- Use the Paper Catalog section to identify which are essential vs. optional

**"How Do These Papers Relate?"**
- Check references -- papers that cite each other are related
- Look at citation counts on arXiv (more cited = more influential)
- Use the "Related" section on arXiv papers

**"Where's the Original GOP Paper?"**
- Referenced as [1] in your paper (Witt & Young, 2000)
- Can find citation-formatted version in references
- Core idea explained well in Section II-A of your paper

### After Reading

**Create Summaries**

For each paper, create a 1-page summary:
- Problem: What is being addressed?
- Approach: What is the main contribution?
- Results: Key findings and benchmarks
- Connections: How does it relate to other papers?

**Build a Comparison Table**

Create a spreadsheet comparing papers on:
- Task (binary/ternary/utterance-level)
- Model (HMM, DNN, CTC, Transformer, etc.)
- Features (GOP, LPP, LPR, embeddings)
- Datasets used
- Key metrics achieved

**Identify Research Gaps**

Based on all papers read, note:
- What problems remain unsolved?
- What would be interesting to explore next?
- Where is the field heading?

---

*Last updated: 2026-02-22*
