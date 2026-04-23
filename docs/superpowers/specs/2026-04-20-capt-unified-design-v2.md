# P015 CAPT-Unified v2

**Date:** 2026-04-20  
**Status:** Draft v2 for implementation  
**Supersedes:** `2026-04-20-capt-unified-design.md` for planning and execution  
**Fork base:** `projects/P013-hmamba-faithful`  
**Working directory:** `projects/P015-capt-unified`

## 1. Decision

P015 will be implemented as **one codebase with two research tracks**:

1. **Scripted CAPT track**
   Audio + reference text.
   Target tasks: `APA + MDD`.
   Backbone family: `HMamba`.

2. **Unscripted spoken-language track**
   Audio only at inference.
   Target task: `APA` only for v1.
   Backbone family: `HiPPO-style hierarchical APA`.

This split is required by the supervision available in the cited papers.
`MDD` in the current paper set depends on canonical phones derived from a reference text.
No paper in this stack shows paper-backed no-reference `MDD` in free-speaking.

## 2. Why v2 exists

The v1 spec bundled several good ideas together, but it overstated what the papers actually support.
v2 keeps the ambition and fixes the attribution and validation logic.

The main corrections are:

- `HMamba`, `Bao`, and `JCAPT` stay in the scripted track.
- `HiPPO` anchors the unscripted track.
- `HConv` and `CHConv` are treated according to the Shih papers, not as interchangeable labels.
- `JCAPT` phonological attributes and think tokens are separated correctly.
- `Bao` proficiency conditioning stays near the matching module, not a made-up FiLM head.
- `Zhao` word timing is treated as a candidate transplant, not as proof that one module alone reaches `0.483` word-stress PCC.
- `xlsr-espeak -> reverse G2P -> words` is treated as an open hypothesis, not an established component.

## 3. Non-negotiables

1. **Paper-faithful claims.**
   The implementation can extend the papers, but the spec must label extensions as hypotheses, not citations.

2. **Shared codebase, separate task contracts.**
   Shared infrastructure is encouraged.
   Shared claims are not assumed.

3. **WandB on every run.**
   No Trackio in P015.
   Every screening run, full run, ablation, and transcript study logs to W&B.

4. **P013 remains the scripted reference baseline.**
   Promotion gates compare against the reproduced P013 means:
   Phone PCC `0.7153`, Utt Total PCC `0.8083`, Word Total PCC `0.6991`, MDD F1 `0.5818`.

5. **The unscripted track must start with simulated free-speaking before real free-speaking.**
   This matches the HiPPO evaluation protocol and removes one confound at a time.

6. **ZIPA is not a drop-in production fallback for current phoneme scoring.**
   In this repo it is structurally incompatible with the current phone-level scoring path without additional work.

## 4. Paper-faithful component map

### 4.1 Scripted track

- **Chao et al. 2025 / HMamba**
  Hierarchical BiMamba backbone for `APA + MDD`, canonical phone input, `deXent` for MDD.

- **Bao et al. 2026**
  Add explicit suprasegmental features:
  formants, spectral balance, pitch, duration, energy.
  Add proficiency embeddings inside the feature matching module.
  Expect stress/prosody gains with a possible phone-score tradeoff.

- **Yang et al. 2025 / JCAPT**
  Add phonological attribute features as symbolic phone-side input.
  Evaluate think tokens as separate learnable appended tokens.
  Do not reinterpret think tokens as predicted attributes.

- **Zhao et al. 2026**
  Add 4D word timing features `[mu, sigma, range, delta]` and test word-level timing injection.
  Do not treat the full Cwacformer result as the expected gain from timing alone.

- **Shih et al. 2024 / 2025**
  If testing layer fusion inside one upstream model, test **HConv** first.
  Reserve **CHConv** for genuine multi-model fusion cases.

### 4.2 Unscripted track

- **Yan et al. 2025 / HiPPO**
  Speech foundation model for word transcription, then G2P for phones.
  Hierarchical pronunciation assessment only.
  `CONO` regularizer.
  Curriculum where the probability of drawing the hard free-speaking task increases over training.

### 4.3 Informing studies, not direct modules

- **Liang et al. 2025**
  Use this as evidence that layer choice matters.
  Do not claim it proves specific prosody vs phoneme layer bands for CAPT.

## 5. Architecture

## 5.1 Shared infrastructure

Shared across both tracks:

- dataset loaders
- feature caching
- W&B logging
- metric computation
- seed management
- ablation runner
- result manifests
- transcript proxy tooling

### Shared repo layout

```text
projects/P015-capt-unified/
├── conf/
│   ├── scripted/
│   └── unscripted/
├── src/capt_unified/
│   ├── shared/
│   ├── scripted/
│   ├── unscripted/
│   ├── data/
│   ├── eval/
│   └── tools/
├── docs/
│   ├── EXPERIMENT_LOG.md
│   ├── CLAIMS_LEDGER.md
│   ├── RUNBOOK.md
│   └── RESULT_MANIFEST_SCHEMA.md
└── tests/
```

## 5.2 Scripted track architecture

```text
audio + reference text
  |
  +--> canonical phones / word boundaries / BIES
  |
  +--> acoustic features
        - GOP
        - optional SSL
        - duration + energy
        - optional explicit prosody features
  |
  +--> HMamba spine
        phone BiMamba -> word BiMamba -> utt BiMamba
  |
  +--> APA heads
  |
  +--> MDD head with deXent
```

Candidate modules for scripted ablation:

- `S-C1` single-upstream `HConv` layer fusion
- `S-C2` Bao DSP stream
- `S-C3` Bao ABM-like matching with proficiency embedding
- `S-C4` JCAPT phonological attribute features
- `S-C5` JCAPT think tokens
- `S-C6` Zhao word timing injection

## 5.3 Unscripted track architecture

```text
audio only
  |
  +--> transcript proxy study
        candidates:
        - xlsr-espeak phone-first path
        - POWSM
        - w2v-bert-derived system
        - Whisper-large-v3 control for HiPPO-faithful comparison
  |
  +--> words
  |
  +--> G2P
  |
  +--> phones + phone/word mapping
  |
  +--> HiPPO-style hierarchical APA encoder
  |
  +--> phone / word / utt APA heads
  |
  +--> CONO
  |
  +--> curriculum over read-aloud vs free-speaking tasks
```

Track-U v1 does **not** include `MDD`.

## 6. Open hypotheses

Each item below is a hypothesis until it clears the validation tiers in Section 8.

| ID | Hypothesis | Track | Why it matters |
|---|---|---|---|
| H1 | P015 scaffold can reproduce P013 within tolerance | Scripted | Prevents false progress caused by scaffold drift |
| H2 | Single-upstream HConv can replace 3-SSL concat without scripted regression | Scripted | Cuts memory and simplifies the front end |
| H3 | Explicit DSP features improve word-stress and utt-prosodic with bounded phone loss | Scripted | Tests Bao-style prosody gains directly |
| H4 | Proficiency embeddings improve scripted fusion beyond DSP alone | Scripted | Tests Bao conditioning separately from DSP |
| H5 | Phonological attributes help phone and word scoring | Scripted | Tests JCAPT symbolic grounding |
| H6 | Think tokens help MDD more than APA | Scripted | Tests JCAPT reasoning mechanism as its own factor |
| H7 | Word timing injection lifts word-stress in HMamba | Scripted | Tests Zhao-style timing transfer |
| H8 | Transcript proxy quality is sufficient for unscripted APA | Unscripted | This is the main bottleneck for Track U |
| H9 | HiPPO-style curriculum beats direct free-speaking training | Unscripted | Tests the central HiPPO claim |
| H10 | Shared lower-level infrastructure can support both tracks without metric drift | Both | Enables a unified codebase without collapsing the tasks |

## 7. Metrics and promotion gates

## 7.1 Scripted primary metrics

- Phone PCC
- Phone MSE
- Word Accuracy PCC
- Word Stress PCC
- Word Total PCC
- Utt Prosodic PCC
- Utt Total PCC
- MDD precision / recall / F1

## 7.2 Unscripted primary metrics

- transcript proxy WER or CER
- phone-level APA PCC
- word-level APA PCC
- utt total PCC
- robustness across transcript quality buckets

## 7.3 Scripted promotion gates

| Stage | Requirement |
|---|---|
| S0 parity | Within `0.005` on Phone PCC and Utt Total PCC vs P013 mean |
| S1 layer fusion | No worse than `-0.005` Phone PCC and `-0.010` MDD F1 vs P013 |
| S2 DSP | Word Stress PCC improves by at least `+0.02` with Phone PCC drop no larger than `0.01` |
| S3 proficiency | Beats S2 on either Word Stress PCC or Utt Prosodic PCC without further phone regression |
| S4 phonological attrs | Improves Phone PCC or Word Accuracy PCC |
| S5 think tokens | Improves MDD F1 or correct diagnosis rate |
| S6 word timing | Improves Word Stress PCC over best prior scripted model |

## 7.4 Unscripted promotion gates

| Stage | Requirement |
|---|---|
| U0 transcript proxy study | Select one primary transcript path and one control path |
| U1 simulated free-speaking baseline | Reproduce a stable HiPPO-style baseline on simulated free-speaking |
| U2 curriculum | Beat no-curriculum baseline on utt total PCC |
| U3 real unscripted pilot | Hold at least `0.70` utt total PCC on the internal pilot set |

## 8. Validation tiers

Every hypothesis passes through three tiers.

### Tier A - Screening

- `1` seed
- `20-30%` of full training budget or shortened epochs
- Purpose: kill bad ideas cheaply
- Hardware: 5070 or spare DGX slot

### Tier B - Candidate

- `3` seeds
- full training schedule
- Purpose: estimate whether the effect is real
- Hardware: DGX Spark preferred

### Tier C - Promotion

- `5` seeds
- full training schedule
- report mean and std
- update `CLAIMS_LEDGER.md`
- Purpose: allow the result to change the roadmap

Promotion rule:

- the mean must improve the target metric, and
- the regression on protected metrics must stay inside the declared bound

Protected metrics:

- scripted track: Phone PCC, Utt Total PCC, MDD F1
- unscripted track: utt total PCC and transcript quality

## 9. Experiment lanes

The goal is not to run many experiments randomly.
The goal is to run orthogonal experiments in parallel so each machine answers a different question.

## 9.1 Lane layout

### Lane A - Canonical baseline lane

Runs on DGX Spark.

Responsibilities:

- P013 parity in P015 scaffold
- best scripted baseline
- 5-seed promotion runs

Rule:
Only one canonical run at a time.
This lane defines the number everyone compares against.

### Lane B - Module ablation lane

Runs on DGX Spark.

Responsibilities:

- HConv vs baseline
- DSP
- DSP + proficiency matching
- phonological attrs
- think tokens
- word timing

Rule:
Change one module family at a time.
Do not stack unvalidated modules together.

### Lane C - Transcript and data lane

Runs on 5070 and CPU.

Responsibilities:

- transcript proxy study
- G2P and reverse-map tooling
- simulated free-speaking score reassignment
- feature extraction checks
- short unscripted screening runs

Rule:
This lane reduces data and supervision risk before DGX time is spent on full training.

## 9.2 Machine roles

### DGX Spark

Use for:

- full scripted training
- 3-seed and 5-seed runs
- unscripted curriculum runs
- parallel long jobs with isolated configs

Avoid using it for:

- trivial smoke tests
- format bugs
- cache-shape debugging
- one-file dataloader mistakes

### 5070 12GB

Use for:

- unit tests and smoke tests
- one-seed shortened runs
- transcript proxy evaluation
- feature cache generation
- ablation preflight
- overnight bug reproduction

Keep models conservative on this box:

- smaller batch sizes
- shorter sequence limits
- gradient accumulation
- BF16 or FP32 when wav2vec2-bert stability is questionable

## 10. Execution order

## Phase 0 - Scaffold and reproducibility

1. Fork P013 into P015.
2. Replace logging with W&B.
3. Reproduce P013 scripted baseline exactly.
4. Freeze this as `scripted-baseline-v1`.

## Phase 1 - Scripted interface study

1. Baseline front end: current P013 3-SSL concat.
2. Single best-layer probe.
3. Single-upstream HConv.
4. Optional multi-model HConv or CHConv only after single-upstream HConv has cleared promotion.

Decision:
If HConv does not clear Tier C, keep P013 front end and move on.

## Phase 2 - Scripted prosody study

1. Add Bao DSP features alone.
2. Add DSP + proficiency matching.
3. Compare against scripted baseline and interface winner.

Decision:
Promote only if stress or prosodic gains are real and phone loss stays bounded.

## Phase 3 - Scripted symbolic study

1. Add phonological attributes only.
2. Add think tokens only.
3. Add both if each clears candidate tier.

Decision:
Keep the two mechanisms separate until both have individual evidence.

## Phase 4 - Scripted timing study

1. Add Zhao word timing to the current best scripted model.
2. Evaluate especially on Word Stress PCC.

## Phase 5 - Unscripted transcript study

1. Build the simulated free-speaking setup first.
2. Compare transcript paths:
   - xlsr-espeak primary research candidate
   - POWSM candidate
   - Whisper-large-v3 control
3. Measure transcript quality and downstream APA impact separately.

Decision:
Choose one primary unscripted transcript path and one control.

## Phase 6 - Unscripted APA baseline

1. Implement HiPPO-style baseline with chosen transcript path.
2. Train with and without curriculum.
3. Train with and without CONO.

Decision:
If curriculum or CONO do not reproduce the expected direction on simulated free-speaking, stop expanding Track U.

## Phase 7 - Real unscripted pilot

1. Build a small real unscripted evaluation set.
2. Run zero architecture changes first.
3. Only after that, test whether any scripted module transfers into Track U.

## 11. Result recording

Every promoted run must record:

- git commit
- exact config
- dataset revision
- seeds
- hardware
- wall time
- primary metrics
- protected metrics
- decision: promoted / rejected
- why the decision changed or did not change the roadmap

Required docs:

- `docs/EXPERIMENT_LOG.md`
- `docs/CLAIMS_LEDGER.md`
- `docs/ABLATION_RESULTS.md`

## 12. Out of scope for v2

- no-reference free-speaking `MDD`
- full multilingual production training from day one
- train-from-scratch CAPT backbones
- real-time latency optimization
- LLM scoring or chat-based feedback generation

## 13. Final operating rule

P015 does not win by looking maximally unified on paper.
P015 wins by establishing a clean scripted baseline, a clean unscripted baseline, and then merging only the pieces that survive controlled ablations.
