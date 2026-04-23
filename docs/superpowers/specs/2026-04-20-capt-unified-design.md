# P015 CAPT-Unified — Design Spec

**Date:** 2026-04-20
**Status:** Design approved, implementation plan pending
**Fork base:** `projects/P013-hmamba-faithful`
**Working directory:** `projects/P015-capt-unified`

## 1. Requirements (non-negotiable)

1. **Multilingual.** The system must accept any L1/L2 pair without retraining. Phone inventory is IPA, not CMU-39. Evaluation must include at least one non-English L2 corpus.
2. **Free-speaking (unscripted).** The system must not require a reference transcript at inference time. Scoring is conditioned on a hypothesized transcript produced by a multilingual speech foundation model (SFM).
3. **W&B tracking on every run.** No Trackio, no bare stdout. Training, eval, and sweeps all log.
4. **No Whisper.** The SFM is `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (387-token IPA, 60+ languages). Secondary candidates: POWSM, ZIPA, `facebook/w2v-bert-2.0`. Whisper is excluded from the SFM slot and every other slot.
5. **HMamba spine retained.** P013's 3-level BiMamba backbone + deXent loss is the reference baseline. All additions must beat or match P013's Phone PCC 0.7153 / Utt 0.8083 / Word 0.6991 / MDD F1 0.5818 on scripted SpeechOcean762 before promotion.

## 2. Target metrics

| Aspect | P013 baseline | P015 target | P015 stretch |
|---|---|---|---|
| Phone PCC | 0.7153 | ≥ 0.720 | ≥ 0.736 (Bao HMamba repro) |
| Word Accuracy PCC | 0.6991 | ≥ 0.700 | ≥ 0.715 |
| **Word Stress PCC** | (weak) | **≥ 0.45** | **≥ 0.483 (Zhao Cwacformer)** |
| Utt Total PCC | 0.8083 | ≥ 0.810 | ≥ 0.829 (HMamba paper) |
| Utt Prosodic PCC | — | ≥ 0.850 | ≥ 0.862 (Bao) |
| MDD F1 | 0.5818 | ≥ 0.60 | ≥ 0.64 |
| Free-speaking Phone PCC | — | ≥ 0.42 | ≥ 0.48 (HiPPO) |
| Free-speaking Utt Total PCC | — | ≥ 0.70 | ≥ 0.754 (HiPPO) |

## 3. Architecture

### 3.1 Data flow

```text
Audio (16 kHz, any L1/L2, any length)
  │
  ├──▶ [A] Multilingual IPA SFM ─────────────────▶ p̂ (hypothesized IPA phones)
  │        xlsr-espeak (primary)                   w̑ (word-grouped via G2P reverse-map)
  │        POWSM / w2v-bert-2.0 (fallback)
  │
  ├──▶ [B] DSP prosody stream (parallel)
  │        • F1/F2 formants (parselmouth)
  │        • 4-band spectral balance
  │        • PyWORLD DIO pitch → 256-Mel bins
  │        → 6-dim + Mel-pitch per frame
  │
  └──▶ [C] SSL encoder + CHConv layer fusion
            • Primary: wav2vec2-bert-2.0 OR xlsr-espeak hidden states
            • CHConv: 1D conv over the LAYER axis
                - prosody-lane taps L3–L7
                - phoneme-lane taps L9–L12
            • Output: two frame streams (prosody-H, phoneme-H)

     [B], [C]-prosody, [C]-phoneme
           │
           ▼
     [D] ABM (Attention-Based Matching, Bao 2026)
         Query: phoneme-H. Keys/values: prosody-H ⊕ DSP stream.
         → fused frame features F

     F, p̂
       │
       ▼
     [E] JCAPT phonological attribute head (Yang 2025)
         14-dim articulatory vector per phone
         trained as auxiliary "think token" prediction
         → attribute-conditioned phone embeddings

                 ▼
     [F] HMamba 3-level BiMamba (Chao 2025) — the spine
         Phone-Mamba → Word-Mamba → Utt-Mamba
         deXent loss at phone-head for MDD

     At Word-Mamba input:
     [G] Zhao 2026 word-timing injection
         per-word [μ, σ, r, δ] (frame-timing stats)
         multi-scale 1D conv (k=1, 3, 5) over word embeddings

     At Utt-head:
     [H] Proficiency FiLM (Bao 2026)
         3-level proficiency embedding from utt-pooled features
         γ, β modulate the utt-regression head

     Loss orchestration:
     [I] HiPPO curriculum (Yan 2025)
         Phase 1: L_read (scripted, forced text)
         Phase 2: 0.5·L_read + 0.5·L_free
         Phase 3: L_free (hypothesized text from [A])
         + λ_CONO · L_CONO (contrastive ordinal regularizer) on all phases
```

### 3.2 Component specs

**[A] SFM.** Input: raw waveform. Output: IPA phone sequence with frame timings; words reconstructed by G2P reverse-map (espeak phonemizer). No language ID required — `xlsr-espeak` handles 60+ languages natively.

**[B] DSP stream.** Computed on-the-fly in dataset `__getitem__`. parselmouth for formants; PyWORLD for pitch. Cached to disk at `~/.cache/peacock-asr/p015/dsp/{split}.pt` after first epoch. 6-dim + 256-dim Mel-pitch, concatenated to `(T, 262)`.

**[C] CHConv layer fusion.** This is the fix for the P011 failure. P011 applied conv *after* phone pooling; P015 applies conv *inside* the SSL layer stack. For an encoder with L layers and D hidden, stack to `(T, L, D)` and run `Conv1d(in=L, out=K, kernel=1)` over the layer axis. Two output heads: `prosody-H` (taps L3–L7 with K=32) and `phoneme-H` (taps L9–L12 with K=64). This replaces HMamba's three-SSL stacking — expect large VRAM win if it holds.

**[D] ABM.** Cross-attention: query = phoneme-H, K/V = `concat(prosody-H, DSP)`. Single-head, 4 layers, dim 256. Output replaces naive concat.

**[E] JCAPT attributes.** 14-dim articulatory feature lookup (place, manner, voicing, height, backness, rounding, nasality, aspiration, ...) — derived from IPA symbols via Panphon. Auxiliary cross-entropy loss on attribute prediction; "think tokens" are these predictions fed back as inputs before phone scoring.

**[F] HMamba spine.** Copied verbatim from P013. No architectural change; only its *inputs* change.

**[G] Zhao word-timing.** For each word span, compute 4-dim stats over its frame-level features. Multi-scale conv with k=1, 3, 5, channels 64 each, concatenated → 192-dim word prior, injected at Word-Mamba input.

**[H] Proficiency FiLM.** Small MLP predicts 3-way proficiency from utt-pooled features. Proficiency embedding (8-dim) projected to γ, β; applied to utt-head MLP.

**[I] HiPPO curriculum.** Three training phases, each a separate W&B run tagged `phase={1,2,3}`. Phase transitions gated by validation loss plateau (patience 3 epochs). CONO contrastive loss ramps from 0 to λ=0.1 across phases.

### 3.3 Multilingual training mix

- **Scripted English:** SpeechOcean762 (existing, primary eval).
- **Scripted multilingual:** L2-Arctic (6 L1 accents), optionally CommonVoice L2 subsets.
- **Free-speaking:** Curate from SpeakOcean free-form subset if available, else record small internal corpus. Phase 3 only.

## 4. Implementation stages

Each stage ends with a W&B-logged ablation and a written go/no-go note in `docs/EXPERIMENT_LOG.md`. Failing ablations halt promotion; the offending component is pulled or redesigned before the next stage.

1. **Scaffold** — fork P013 → P015. Wire W&B (replace Trackio). Reproduce P013 numbers under the new scaffold. Gate: PCC within 0.005 of P013.
2. **CHConv layer fusion [C]** — replace HMamba's SSL-stacking with single-SSL + CHConv. Gate: Phone PCC ≥ P013.
3. **DSP + ABM [B, D]** — add parallel prosody stream and ABM fusion. Gate: Utt-Prosodic ≥ 0.84, Phone PCC not regressed by > 0.010.
4. **Zhao word-timing [G]** — multi-scale conv at Word-Mamba. Gate: Word-Stress PCC ≥ 0.45.
5. **JCAPT attributes [E]** — articulatory auxiliary head. Gate: no regression + MDD F1 ≥ 0.60.
6. **Proficiency FiLM [H]** — add if prior gates held. Gate: Utt Total PCC ≥ 0.81.
7. **HiPPO curriculum [I] + SFM-hypothesized transcripts [A]** — switch to free-speaking training. Gate: free-speaking Phone PCC ≥ 0.42, Utt Total ≥ 0.70.
8. **Multilingual extension** — retrain on L2-Arctic + English. Gate: held-out L2-Arctic Phone PCC ≥ 0.60.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CHConv underperforms 3-SSL stacking | Keep P013 3-SSL path as a config flag; revert if CHConv regresses. |
| DSP stream adds phone-PCC regression (as in Bao) | ABM [D] is the designed fix. If it doesn't recover, gate stage 3 on phone-PCC parity. |
| xlsr-espeak transcripts too noisy for free-speaking | POWSM fallback; or train a distilled multilingual SFM from wav2vec2-bert. |
| JCAPT attribute head hurts convergence | Loss weight schedule; start λ_attr=0, ramp to 0.1 over 5 epochs. |
| Free-speaking corpus too small | Curriculum phase 3 uses mixed scripted+free; phase 3 runs until data allows. |
| VRAM blowup from SSL+DSP+HMamba+Mel-pitch | Mel-pitch is 256 not 512; DSP cached; single SSL via CHConv (not three); gradient checkpointing on HMamba. |

## 6. Repo layout (P015)

```text
projects/P015-capt-unified/
├── pyproject.toml
├── conf/                      # Hydra configs per stage
│   ├── stage1_scaffold.yaml
│   ├── stage2_chconv.yaml
│   ├── ...
│   └── stage8_multilingual.yaml
├── src/capt_unified/
│   ├── models/
│   │   ├── chconv.py          # [C]
│   │   ├── dsp_stream.py      # [B]
│   │   ├── abm.py             # [D]
│   │   ├── jcapt_head.py      # [E]
│   │   ├── hmamba_spine.py    # [F] (lifted from P013)
│   │   ├── word_timing.py     # [G]
│   │   └── proficiency.py     # [H]
│   ├── training/
│   │   ├── curriculum.py      # [I]
│   │   ├── losses.py          # CONO, deXent, attribute-CE
│   │   └── trainer.py         # W&B-wired
│   ├── data/
│   │   ├── speechocean.py
│   │   ├── l2arctic.py
│   │   ├── free_speaking.py
│   │   └── sfm_transcribe.py  # [A] hypothesized transcript prep
│   └── eval/
│       ├── scripted.py
│       └── free_speaking.py
├── docs/
│   ├── EXPERIMENT_LOG.md      # W&B run table + stage gates
│   ├── RUNBOOK.md
│   └── ABLATION_RESULTS.md
├── runs/                      # local artifacts
└── tests/
```

## 7. Out of scope

- **Train-from-scratch backbones (P003/P004 direction).** Not ruled out long-term, but orthogonal to this spec.
- **LLM-based scoring (Liu 2026 ALM).** Revisit only if hierarchical stack plateaus.
- **Real-time streaming.** Latency optimization is a follow-on once PCC targets hold.
- **Semantic/coherence judgment.** Layer 3 of the north-star product, separate spec.

## 8. Naming of referenced work

- **Chao et al 2025** — HMamba (backbone).
- **Bao et al 2026** — DSP stream [B], ABM [D], Proficiency FiLM [H].
- **Zhao et al 2026** — Cwacformer word-timing [G].
- **Yang et al 2025** — JCAPT phonological attributes [E].
- **Yan et al 2025 (HiPPO)** — curriculum [I], CONO loss.
- **Shih 2024/2025** — HConv/CHConv layer fusion [C].
- **Liang 2025** — SSL layer-role analysis (prosody early, phoneme late).
- **Han et al 2026 (HIA)** — considered; bidirectional granularity interaction may appear as stage-9 refinement if Utt↔Phone coupling still lags.
