# Peacock-ASR Research Meta-Plan

*Lab management document. Tracks dependencies, status, and cross-track breakthroughs.*

Last updated: 2026-03-03

---

## Track Index

| Track | Title | Status | Workspace | Key Metric |
|-------|-------|--------|-----------|------------|
| 05 | CTC Phoneme Posteriors for Segmentation-Free Pronunciation Assessment | **Active** — Phase 1 complete, Phase 2 in progress | `track05_paper/` | Phone PCC 0.677 (baseline) |
| 06 | LLM-Based Pronunciation Assessment (Phi-4, Qwen2-Audio) | Planned | `track06_llm_pronunciation/` | Phone PCC vs GOP baseline |
| 07 | Training CTC Models from Scratch (Conformer, Zipformer) | Planned | `track07_training_from_scratch/` | PER on LibriSpeech |
| 08 | Real-Time Streaming Pronunciation Assessment | Planned | `track08_realtime_streaming/` | Latency (ms) + PCC |
| 09 | ConPCO Regularization for CTC-Based Scoring | **Next** — workspace ready, code cloned | `track09_conpco_scoring/` | Phone PCC (loss function gain) |
| 10 | Compact CTC Backbones (95M, 10M vs 300M) | Planned — workspace ready | `track10_compact_backbones/` | PCC vs backbone params |

Narrative docs (legacy, kept for reference):

- `05_PHONEME_HEADS.md` — Original Track 05 narrative (900 lines)
- `06_LLM_PRONUNCIATION.md` — Original Track 06 notes
- `07_TRAINING_FROM_SCRATCH.md` — Original Track 07 notes
- `08_REALTIME_STREAMING.md` — Original Track 08 notes

---

## Current Scoreboard (SpeechOcean762, Phone-Level PCC)

| System | PCC | Source |
|--------|-----|--------|
| GOPT (Kaldi) | 0.612 | Gong et al. 2022 |
| HiPAMA | 0.616 | Do et al. 2023 |
| Gradformer | 0.646 | Pei et al. 2023 |
| HIA | 0.657 | Han et al. 2026 |
| **Ours (xlsr-53 + GOPT + GOP-SF)** | **0.677** | Track 05 Phase 1 |
| HierCB + ConPCO | 0.743 | Yan et al. 2025 |

Our system already beats all published GOP-based methods. The gap to close is 0.677 → 0.743 (ConPCO/HierCB uses SSL embeddings directly, not GOP).

---

## Dependency Graph

```text
Track 05 (baseline) ──────────────────────────┐
  │                                            │
  ├── Track 09 (ConPCO loss)                   │  independent
  │     └── Can the ConPCO loss improve        │  of each other
  │         GOPT? (loss swap only)             │
  │                                            │
  ├── Track 10 (compact backbones) ────────────┘
  │     └── Do 95M models match 300M?
  │         Needs CTC fine-tuning (shared with Track 07)
  │
  ├── Track 06 (LLM pronunciation) ── independent
  │     └── Skip GOP entirely. Phi-4 / Qwen2-Audio.
  │         No dependency on backbone or scorer.
  │
  ├── Track 07 (training from scratch) ── feeds Track 10
  │     └── Build CTC models from TIMIT/LS.
  │         Conformer/Zipformer backbone → used in Track 10.
  │
  └── Track 08 (real-time streaming) ── depends on Track 05/10
        └── Can GOP-SF work incrementally?
            Needs a working backbone first.
```

### Key Dependencies

| Upstream | Downstream | What Flows |
|----------|------------|------------|
| Track 05 | Track 09 | Frozen GOP-SF features, GOPT baseline, eval protocol |
| Track 05 | Track 10 | Backend interface, GOP-SF algorithm, GOPT scorer |
| Track 07 | Track 10 | CTC fine-tuning recipe for wav2vec2-base / HuBERT-base |
| Track 05/10 | Track 08 | Working backbone + scorer to stream-ify |
| None | Track 06 | Independent paradigm (end-to-end LLM) |

### Cross-Track Breakthroughs

If a breakthrough happens in one track, here's how it propagates:

| Breakthrough | Propagates to |
|-------------|---------------|
| ConPCO loss improves GOPT by >3% PCC (Track 09) | Track 10 (test ConPCO + small backbone), Track 08 (use ConPCO in streaming scorer) |
| 95M backbone matches 300M within CI (Track 10) | Track 08 (streaming with small model), Track 09 (re-run ConPCO with small backbone) |
| LLM scoring beats GOP pipeline (Track 06) | All tracks — may shift entire research direction |
| Custom Conformer matches wav2vec2-base (Track 07) | Track 10 (add to Pareto plot), Track 08 (purpose-built streaming model) |
| GOP-SF works incrementally (Track 08) | Track 05 (add streaming results to paper) |

---

## Execution Priority

### Tier 1: Run Now (infrastructure ready)

1. **Track 09 — ConPCO loss integration**
   - Why first: Directly improves our best system (0.677 → ???). Smallest implementation effort (loss function swap). Code already cloned (`references/ConPCO/`).
   - Effort: ~3 days (port loss, run GOPT with ConPCO, 3 seeds)
   - Blocked by: Nothing

2. **Track 05 — Phase 2 completion (logit scalar experiments)**
   - Already in progress. Runs exist in `runs/2026-03-03_*`.
   - Effort: ~1 day (finish remaining logit scalar variants)
   - Blocked by: Nothing

### Tier 2: Run Next (needs some setup)

1. **Track 10 — Phase 1: backbone swap (wav2vec2-base, HuBERT-base)**
   - Why: Publishable regardless of outcome (first systematic comparison).
   - Effort: ~5 days (fine-tune 2 models on LibriSpeech, create backend adapters)
   - Blocked by: CTC fine-tuning recipe (overlap with Track 07)

### Tier 3: Longer-Term

1. **Track 06 — LLM pronunciation (Phi-4 LoRA)**
   - Independent track. Requires RunPod GPU (VRAM > 12GB for Phi-4).
   - Effort: ~7 days
   - Blocked by: RunPod availability, unclear scoring rubric design

2. **Track 07 — Training from scratch**
   - Research track. TIMIT sandbox first, then Conformer on LibriSpeech.
   - Effort: ~10+ days
   - Blocked by: icefall setup, TIMIT license

3. **Track 08 — Real-time streaming**
   - Hardest track. Only one paper (CoCA-MDD) addresses streaming + pronunciation.
   - Effort: ~14+ days
   - Blocked by: Track 05/10 results (need a stable backbone first)

---

## Infrastructure Notes

### Compute Resources

| Resource | Location | VRAM | Best For |
|----------|----------|------|----------|
| gmk-server | Local | 12 GB (RTX 5070) | GOPT training (31K params), GOP feature extraction, quick experiments |
| RunPod L4 | Cloud | 24 GB | CTC fine-tuning (wav2vec2, HuBERT), Phi-4 LoRA |
| RunPod A100 | Cloud | 40/80 GB | Full Phi-4 training, large-batch CTC fine-tuning |

### Shared Infrastructure Across Tracks

- **SpeechOcean762 dataset**: Pinned HF revision, 2500/2500 split — all tracks use this
- **GOP-SF features**: Cached at `~/.cache/peacock-asr/features/{backend}/{split}.pt`
- **MLflow**: `mlflow.peacockery.studio` — all experiments logged here
- **Backend interface**: `src/peacock_asr/backends/` — new backbones plug in here
- **GOPT model**: `src/peacock_asr/gopt_model.py` — scorer shared by Tracks 05/09/10

### BF16 Training (validated 2026-03-03)

- L4 GPU supports BF16 natively (Ada Lovelace, 242 TFLOPS)
- Fix merged to master: `torch.cuda.is_bf16_supported()` runtime check
- 21% throughput gain vs FP32 (2.29 vs 1.89 samples/sec)
- Gradient checkpointing re-enabled for memory efficiency

---

## Track Status Details

### Track 05: Segmentation-Free GOP + GOPT (Active)

**Phase 1** (complete): Baseline reproduction

- xlsr-53 + GOPT: PCC 0.6774 ± 0.0127 (3 seeds)
- xlsr-espeak backend: PCC 0.648 (lower, as expected)

**Phase 2** (in progress): Logit scalar experiments

- Testing logit-margin and raw-logit GOP variants
- Runs in `runs/2026-03-03_*`

**Phase 3** (planned): Ablation of GOP-SF feature components

### Track 09: ConPCO Scoring (Next Up)

- Workspace ready, code cloned at `references/ConPCO/`
- Phase 1: Port ConPCO loss to GOPT trainer
- Phase 2: Hierarchical aggregation (phone → word → utterance)
- Phase 3: CLAP contrastive alignment

### Track 10: Compact Backbones (Planned)

- Phase 1: wav2vec2-base (95M), HuBERT-base (95M) as GOP backbones
- Phase 2: Citrinet-256 (10M) — extreme compression
- Phase 3: Scoring head comparison (GOPT vs HMamba vs HiPAMA)
- Phase 4: Pareto analysis (PCC vs params)

### Track 06: LLM Pronunciation (Planned)

- Phase 1: Phi-4 multimodal + LoRA on SpeechOcean762
- Phase 2: Qwen2-Audio-7B comparison
- Paradigm shift: skip GOP entirely, end-to-end scoring

### Track 07: Training from Scratch (Planned)

- Phase 1: TIMIT sandbox with icefall
- Phase 2: Conformer phoneme CTC on LibriSpeech
- Phase 3: Compare vs wav2vec2-base (Track 10 crossover)

### Track 08: Real-Time Streaming (Planned)

- Phase 1: Chunked GOP-SF feasibility
- Phase 2: Streaming backbone evaluation
- Hardest track — least prior work to build on

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-02 | Use PCC (not MSE) as primary phone-level metric | Matches all published work on SpeechOcean762 |
| 2026-03-02 | Minimum 3 seeds per configuration | Lab methodology standard |
| 2026-03-03 | BF16 by default on L4 (merged bf16-fix) | 21% throughput gain, no accuracy loss |
| 2026-03-03 | ConPCO before compact backbones | Higher potential PCC gain, less implementation effort |
| 2026-03-03 | Keep narrative docs as legacy references | Don't delete — track workspaces supersede them |

---

## Publishing Strategy

Papers can be submitted independently. Likely order:

1. **Track 05** — Baseline paper (GOP-SF + GOPT, already beats SOTA GOP methods)
2. **Track 09** — ConPCO integration (if loss improves PCC meaningfully)
3. **Track 10** — Compact backbones (publishable even with negative results)
4. **Track 06** — LLM pronunciation (if competitive with GOP pipeline)
5. **Track 07/08** — Longer-term, depends on breakthroughs

Target venues: ICASSP, Interspeech, ACL (for LLM track), TASLP (journal for comprehensive study).
