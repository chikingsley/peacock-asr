# Pronunciation Annotation: Research & Design Knowledge

Research synthesis for P007 dataset-making. Covers the landscape of pronunciation assessment datasets, crowdsourced annotation methodology, gamification for annotator engagement, and quality control mechanisms.

---

## 1. The Dataset Landscape

### What exists for pronunciation assessment

| Dataset | Size | Granularity | Annotation Type | License | Notes |
|---------|------|-------------|-----------------|---------|-------|
| **SpeechOcean762** | 5000 utts, 250 spk | Phone + Word + Sentence | Quality scores (0-2, 0-10) | CC-BY 4.0 | Only free multi-level scored dataset. 300+ citations. Chinese L1 speakers. |
| **L2-ARCTIC** | 27K utts, 24 spk | Phone-level | Error labels (sub/del/ins) | CC BY-NC 4.0 | 6 L1 backgrounds. Error detection only, no quality gradient. |
| **CMU-Kids** | ~5180 utts | Phone-level | Binary correct/incorrect | Academic | Children ages 6-11. Used for GOP-SF evaluation. |
| **ISLE** | ~18h | Phone + Word | Error tags + prosody | Academic (paid commercial) | German/Italian L1. Early standard (2000). |
| **ERJ** | 202 speakers | Phone + Prosodic | Phonemic/prosodic symbols | Academic | Japanese L1. Used for ASR-free GOP research. |
| **TL-school** | Northern Italy students | Utterance-level | Proficiency scores | Research | English + German L2. |
| **L2-ARCTIC** annotations | ~3600 utts annotated | Phone-level | Sub/Del/Ins labels | CC BY-NC 4.0 | Subset with detailed error annotations. |

### Gap analysis

SpeechOcean762 is the **only** free dataset with all of:

- Multi-level quality scores (phone/word/sentence)
- Multiple expert annotators (5 per utterance)
- Both children and adults
- Free for commercial use

Everything else is either paywalled, only annotates at one level, or only does binary error detection. This monopoly is why SO762 has 300+ citations despite its limitations.

### What the field needs

1. **Larger scale** — 5000 utterances from one L1 (Mandarin) is small
2. **More L1 diversity** — Current datasets are dominated by Chinese and Japanese speakers
3. **Mispronunciation diagnosis** — SO762 has this (`mispronunciations` field) but most papers ignore it
4. **Audio pairs** — Learner pronunciation + reference pronunciation for contrastive learning
5. **Quality + error labels** — Combine SO762-style quality scores with L2-ARCTIC-style error annotations

---

## 2. Expert vs. Crowd Annotation

### The SpeechOcean762 approach (Expert panel)

- 5 expert annotators scored every utterance independently
- Voting on canonical phone sequences: each expert independently selected a sequence, majority wins
- Notation: `()` = score 0 (incorrect), `{}` = score 1 (accented), no symbol = score 2 (correct), `[]` = insertion
- Per-expert detail preserved in `scores-detail.json` (arrays of 5 values)
- Final scores in `scores.json` are averaged/median across experts
- Tool ("SpeechOcean uTrans") had consistency checks — warned if word accuracy contradicted phone scores

**Strengths:** High quality, consistent, detailed phonetic knowledge.
**Weaknesses:** Expensive, slow, doesn't scale. Only 5000 utterances in the final dataset.

### The Loukina (ETS) finding — Transcription beats error-marking

**Paper:** Loukina et al. (2015). "Expert and crowdsourced annotation of pronunciation errors for automatic scoring systems." Interspeech 2015. [ISCA Archive](https://www.isca-archive.org/interspeech_2015/loukina15b_interspeech.html)

Key findings:

- Compared expert linguists (detailed phonetic transcription) vs. naive crowd workers (AMT) doing simpler tasks
- **Low agreement even between experts** on what constitutes a pronunciation error
- Breakthrough: instead of "mark the errors" (subjective), ask crowd workers to **transcribe what they heard** in normal English spelling
- If crowd transcribes "bear" as "beer" → pronunciation error detected automatically via diff against canonical text
- Transcription-based approach produced **more valid training data** for automatic scoring systems than expert annotation

**Implication:** The task matters more than the annotator's expertise. Hide the linguistic complexity; let errors emerge from the mismatch between canonical and transcribed text.

**Data availability:** NOT publicly available. ETS proprietary (TOEFL test data). The methodology is what we take, not the data.

### The Common Voice cautionary tale

**Paper:** Hjortnaes et al. (2024). "Evaluating Automatic Pronunciation Scoring with Crowd-sourced Speech Corpus Annotations." NLP4CALL Workshop. [ACL Anthology](https://aclanthology.org/2024.nlp4call-1.6.pdf)

Key finding: Common Voice's crowdsourced annotations are a **poor substitute** for mispronunciation annotations because annotators flag audio quality issues and misreadings rather than actual pronunciation errors.

**Lesson:** Not all crowdsourced annotations are equal. The task framing determines what you get. "Is this recording okay?" ≠ "Is this pronunciation correct?"

### The Duolingo/Cai approach — Intelligibility over imitation

**Paper:** Cai et al. (2025). "Developing an Automatic Pronunciation Scorer: Aligning Speech Evaluation Models and Applied Linguistics Constructs." Language Learning. [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/lang.70000)

Shift in philosophy: score pronunciation based on **intelligibility** (can you understand it?) rather than **native-likeness** (does it sound like a native speaker?). Trained on bespoke human-rated dataset aligned to CEFR descriptors. Their scorer outperforms models trained on accent-matching approaches.

**Implication:** Our scoring rubric should prioritize "can a listener understand this?" over "does this match a reference accent?" This aligns perfectly with crowd annotation — non-experts are ideal judges of intelligibility.

---

## 3. Gamification for Annotation

### The foundational work: Games With A Purpose (GWAPs)

**Creator:** Luis von Ahn (Carnegie Mellon, later Duolingo/reCAPTCHA). [Wikipedia](https://en.wikipedia.org/wiki/Luis_von_Ahn)

The [ESP Game](https://edutechwiki.unige.ch/en/ESP_game) (2003) established three templates for gamified annotation:

| Template | How it works | Quality mechanism | Pronunciation application |
|----------|-------------|-------------------|--------------------------|
| **Output-agreement** | Two players independently label same item, score when they match | Consensus = correctness | Two listeners transcribe same word; agreement = gold label |
| **Input-agreement** | Two players may see same or different item, must determine which | Forces careful attention | "Did these two speakers say the same word?" |
| **Inversion-problem** | One player sees label, finds item; other sees item, guesses label | Asymmetric challenge | One player hears word, types it; other sees spelling, rates pronunciation |

Key insight: **quality emerges from the game mechanic itself**, not from post-hoc verification. You can't cheat because you don't know what your partner will say.

### The comprehensive review: Quest for Quality

**Paper:** "Quest for quality: a review of design knowledge on gamified AI training data annotation systems." Behaviour & Information Technology. [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/0144929X.2025.2568932)

Reviewed **56 gamified annotation systems** and synthesized 13 solution streams. Key findings:

#### What works

| Mechanism | Evidence | How it fights apathy |
|-----------|----------|---------------------|
| **Progressive difficulty** | Consistent across studies | Builds confidence early, prevents dropout |
| **Output agreement / consensus** | ESP Game, GWAPs literature | Makes accuracy the winning strategy |
| **Micro-sessions** | Fatigue research | 5-10 items per round prevents cognitive overload |
| **Streak mechanics** | Engagement research | Directly combats "click through without looking" |
| **Narrative framing** | "Dr. Detective" (CEUR 2013) | Purpose > points. "You're helping learners improve" |
| **Immediate feedback** | Gold/honeypot items | "Nice catch!" on known items reinforces attention |
| **Variety rotation** | Fatigue research | Alternate task types to prevent habituation |

#### What backfires

| Anti-pattern | Why it fails |
|-------------|-------------|
| **Leaderboards alone** | Speed-gamers race through without quality. Need quality-weighted scoring. |
| **Points without meaning** | If points don't unlock anything, motivation drops fast |
| **Competitive elements for non-competitive users** | Some annotators find leaderboards stressful. Personalization matters. |
| **Static gamification** | [Gamification fatigue](https://www.sciencedirect.com/science/article/abs/pii/S1567422324000140) — even good designs lose effectiveness if novelty wears off. Rotate mechanics. |
| **Overly complex UIs** | Annotation tool complexity is the #1 predictor of annotator dropout |

### Exception-based annotation ("default correct")

Used by SpeechOcean762 for phonemes: default score = 2 (correct), annotator only clicks if wrong.

**Why it works:**

- ~85% of phonemes in SO762 are correct → massive click reduction
- Psychologically: "find the bad ones" is more engaging than "rate everything"
- Prevents fatigue from repetitive positive labeling
- Amazon MTurk research confirmed: explicit positive labeling of every item leads to lower quality (people auto-click "correct" without attention)

**Two-pass extension:**

1. Coarse sweep: everything green (correct) by default, tap to reject
2. Targeted drill-down: only for rejected items, do detailed scoring

---

## 4. Quality Control Mechanisms

### Industry standard patterns (from Kili, Dataloop, Scale AI, etc.)

#### Honeypot / Gold Standard items

**How it works:** Insert pre-labeled items with known correct answers into the annotation stream. Measure annotator accuracy against ground truth.

**Implementation (from [Kili docs](https://docs.kili-technology.com/docs/honeypot-overview)):**

- Mark 3-10% of assets as honeypot before labelers start
- Attach a "Review" label containing the correct answer
- System intersperses gold items throughout the annotation queue
- Track scores at 4 levels: label, asset, annotator, project
- Honeypot score = agreement between annotator's label and ground truth

**Best practices:**

- Set up honeypot **before** annotators start working
- Use honeypot to onboard new annotators — their first session is heavily honeypotted
- Score threshold: below 70% agreement → flag for retraining or removal
- Rotate honeypot items periodically so annotators don't memorize them

#### Consensus / Inter-Annotator Agreement

**How it works:** Multiple annotators label the same item. Measure agreement.

**Implementation (from [Kili docs](https://docs.kili-technology.com/docs/consensus-overview)):**

- Configure consensus percentage (e.g., 20% of items get multiple annotators)
- Set number of labelers per consensus item (e.g., 3)
- Agreement calculated per item, per annotator, and per project
- For classification: exact match or category overlap
- For transcription: Dice coefficient (text overlap)
- For bounding boxes: IoU (intersection over union)

**For pronunciation annotation, consensus applies at:**

- Word-level binary ("sounds correct?" → yes/no) → exact match across annotators
- Transcription ("what did you hear?") → Dice coefficient or edit distance
- Phone-level scoring (0/1/2) → Cohen's kappa or weighted agreement

#### Review workflow

**Multi-step process (from [Kili docs](https://docs.kili-technology.com/docs/reviewing-labeled-assets)):**

1. Annotator labels asset
2. Asset enters review queue (automatic or sampled)
3. Reviewer either approves or sends back with issues
4. Issues can be annotation-level (specific label) or global (entire asset)
5. Corrector fixes and resubmits
6. Review queue automation: random sampling to eliminate reviewer bias

#### Quality metrics formulas

From [Kili's calculation rules](https://docs.kili-technology.com/docs/calculation-rules-for-quality-metrics):

- **Classification (single-choice):** Binary 0/1 match
- **Classification (multi-choice):** `overlapping_classes / selected_classes`
- **Transcription consensus:** Dice coefficient (text overlap, 1 = perfect)
- **Multi-job:** Weighted average (same level) or geometric average (nested)

### The Dawid-Skene model (for aggregating noisy labels)

When using many non-expert annotators, individual labels are noisy. The [Dawid-Skene model](https://en.wikipedia.org/wiki/Dawid%E2%80%93Skene_model) (1979) is the standard for aggregating:

- Estimates each annotator's confusion matrix (error tendencies)
- Jointly estimates true labels and annotator reliability
- More reliable annotators get more weight in the final label
- Works iteratively (EM algorithm)
- Used by most crowdsourcing platforms internally

**For pronunciation:** If annotator A always marks /θ/ as incorrect (maybe they can't distinguish it), the model learns to downweight A's opinions on /θ/ while still trusting them on vowels.

---

## 5. Build vs. Buy: Platform Analysis

### Option A: Paid annotation platform (Kili, Labelbox, Dataloop)

**Pros:**

- Built-in honeypot, consensus, review workflows
- Analytics dashboards for annotator performance
- Workforce management, onboarding tools
- Proven at scale

**Cons:**

- **Cost:** Kili starts at ~$500/month for teams, enterprise pricing for serious use
- **Audio support:** Most platforms optimize for vision/text; audio annotation UIs are basic
- **Pronunciation-specific:** None of these have phoneme-level scoring built in
- **Lock-in:** Data in their format, workflow in their system
- **Customization:** Can't do output-agreement games or custom gamification

**Verdict:** Skip. The quality *ideas* are gold; the platforms aren't built for our task.

### Option B: Custom app (what P007 already is) + stolen patterns

**Pros:**

- Full control over annotation UX (gamification, audio playback, phone-level interaction)
- Pronunciation-specific workflows (ABX comparison, TTS reference, phone scoring)
- SQLite → easy to snapshot, version, share
- Can implement the exact quality patterns we want
- No recurring cost

**Cons:**

- Must build quality management ourselves
- No built-in workforce management

**Verdict:** This is the path. We build it, but we steal every good idea from Kili's playbook.

### What to steal from Kili

1. **Honeypot infrastructure** — Pre-label 5-8% of items with known answers, intersperse in queue, track per-annotator scores
2. **Consensus sampling** — 15-20% of items get 3+ annotators, measure agreement
3. **Review workflow** — Reviewer role that can approve or send-back with issues
4. **Per-annotator analytics** — Track accuracy (via honeypot), consistency (via consensus), speed, and completion rate
5. **Score thresholds** — Auto-flag annotators below 70% honeypot accuracy for review
6. **Queue randomization** — Don't let reviewers cherry-pick what to review

---

## 6. Proposed Annotation Architecture

### Three-phase pipeline

```text
Phase 1: Coarse Sweep          Phase 2: Targeted Drill-Down       Phase 3: Expert Review
(Non-experts, gamified)         (Semi-expert, ABX comparison)       (Experts, full uTrans)

 ┌─────────────────────┐       ┌──────────────────────────┐       ┌─────────────────────┐
 │ Listen to sentence   │       │ Play TTS reference word   │       │ Full annotation UI   │
 │ All words green      │       │ Play learner's word       │       │ Phone-level scoring  │
 │ Tap to flag bad ones │──────▶│ Same / Different / Very   │──────▶│ Word accuracy 0-10   │
 │ 10 items per round   │ only  │ Optional: "What did it    │ only  │ Sentence scores      │
 │ Output-agreement     │ red   │   sound like?" transcribe │ ambig │ Mispronunciation dx  │
 └─────────────────────┘ items  └──────────────────────────┘ items └─────────────────────┘

 5-10 annotators/item            3-5 annotators/item                1-2 experts/item
 ~3 min/session                  ~5 min/session                     ~10 min/session
 Binary: correct/flagged         3-point: same/slight/very diff     Full rubric
 Honeypot: 8% gold items         Honeypot: 5% gold items            Consensus: 20%
```

**Volume reduction at each phase:**

- Phase 1 processes 100% of data → flags ~15-25% of words
- Phase 2 processes only flagged words → confirms ~60%, escalates ~40%
- Phase 3 processes only escalated items → ~6-10% of original data

### Quality control at each phase

| Phase | QC Mechanism | Implementation |
|-------|-------------|----------------|
| 1 | Honeypot (8%) | Pre-scored items from SO762 test set |
| 1 | Output agreement | 2+ annotators per item, flag disagreements |
| 1 | Streak tracking | Reward consecutive agreements with gold items |
| 2 | ABX gold items | TTS pairs where we know the answer |
| 2 | Transcription diff | Compare crowd transcription to canonical text |
| 3 | Inter-expert consensus | 20% overlap between expert reviewers |
| 3 | Consistency checks | Warn if word score contradicts phone scores (à la SO762) |

### Anti-fatigue mechanisms

| Mechanism | Phase(s) | Implementation |
|-----------|----------|----------------|
| **Default correct** | 1 | All words green, tap to reject |
| **Micro-sessions** | 1, 2 | 10 items (Phase 1) or 5 items (Phase 2) per round |
| **Progressive difficulty** | 1 | Start with high-accuracy speakers, unlock harder ones |
| **Immediate feedback** | 1, 2 | "Nice catch!" on honeypot items |
| **Variety rotation** | 1, 2 | Alternate between listen-and-flag, transcribe, and compare tasks |
| **Session limits** | All | Max 15 minutes, encouraged breaks, show fatigue warning |
| **Purpose framing** | All | "You're helping language learners improve their English" |
| **Progress visualization** | All | Completion %, contribution stats, impact metrics |

---

## 7. References

### Core papers

- Zhang et al. (2021). "speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment." Interspeech 2021. [arXiv:2104.01378](https://arxiv.org/abs/2104.01378)
- Loukina et al. (2015). "Expert and crowdsourced annotation of pronunciation errors for automatic scoring systems." Interspeech 2015. [ISCA Archive](https://www.isca-archive.org/interspeech_2015/loukina15b_interspeech.html)
- Hjortnaes et al. (2024). "Evaluating Automatic Pronunciation Scoring with Crowd-sourced Speech Corpus Annotations." NLP4CALL. [ACL Anthology](https://aclanthology.org/2024.nlp4call-1.6/)
- Cai et al. (2025). "Developing an Automatic Pronunciation Scorer." Language Learning. [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/lang.70000)
- El-Kheir et al. (2023). "Automatic Pronunciation Assessment — A Review." [arXiv:2310.13974](https://arxiv.org/html/2310.13974)

### Gamification & crowdsourcing

- Von Ahn & Dabbish (2004). "Labeling images with a computer game." CHI 2004. [ResearchGate](https://www.researchgate.net/publication/2956916_Games_with_a_Purpose)
- "Quest for quality: a review of design knowledge on gamified AI training data annotation systems." Behaviour & Information Technology. [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/0144929X.2025.2568932)
- "Understanding the failing of social gamification: A perspective of user fatigue." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1567422324000140)
- "Gamified Expert Annotation Systems: Meta-Requirements and Tentative Design." [Springer](https://link.springer.com/chapter/10.1007/978-3-031-06516-3_12)
- Nagao (2019). "Developing and validating a methodology for crowdsourcing L2 speech ratings in AMT." [John Benjamins](https://www.jbe-platform.com/content/journals/10.1075/jslp.18016.nag)

### Quality control

- Kili Technology. "Honeypot overview." [Docs](https://docs.kili-technology.com/docs/honeypot-overview)
- Kili Technology. "Consensus overview." [Docs](https://docs.kili-technology.com/docs/consensus-overview)
- Kili Technology. "Calculation rules for quality metrics." [Docs](https://docs.kili-technology.com/docs/calculation-rules-for-quality-metrics)
- Kili Technology. "Reviewing labeled assets." [Docs](https://docs.kili-technology.com/docs/reviewing-labeled-assets)
- Daniel & Karnin (2018). "Quality control in crowdsourcing: A survey." ACM Computing Surveys. [ACM](https://dl.acm.org/doi/10.1145/3148148)
- Dawid & Skene (1979). "Maximum likelihood estimation of observer error-rates." Applied Statistics.

### Datasets

- SpeechOcean762: [OpenSLR](https://www.openslr.org/101/) | [HuggingFace](https://huggingface.co/datasets/mispeech/speechocean762) | [GitHub](https://github.com/jimbozhang/speechocean762)
- L2-ARCTIC: [TAMU PSI Lab](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- ISLE: [ISLE project](http://www.isle-project.org/)
- CMU Kids: Referenced in Cao et al. (2025), "Segmentation-Free Goodness of Pronunciation"
