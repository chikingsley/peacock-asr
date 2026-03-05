# Track 08 Paper Workspace: Real-Time and Streaming Pronunciation Scoring

Working title:

- **Chunked GOP-SF for Low-Latency Pronunciation Scoring: Toward Real-Time Feedback**

Purpose:

- Determine whether GOP-SF can be applied word-by-word (chunked) as utterances arrive.
- Evaluate streaming ASR models (Voxtral, Moshi, Moonshine v2) as CTC posterior sources.
- Characterize latency and score stability under incremental processing.
- Establish what "real-time" pronunciation scoring actually requires in practice.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`

Draft files:

- `manuscript.md` (primary writing file)

Citation convention:

- Use Pandoc/Quarto citekeys, e.g. `[@shi2021coca_mdd]`.
- All citekeys are in `./refs.bib`.

Process:

1. Phase 1 (chunked GOP-SF) first — no new model needed, reuses existing pipeline.
2. Lock latency measurement protocol before any streaming model experiments.
3. Write Results only from reproducible logs/artifacts.
4. Run evidence audit before finalizing claims.

Papers (PDFs in `./papers/`):

- `[Shi et al, 2021]-coca-mdd-streaming-coupled-cross-attention.pdf` — CoCA-MDD: only paper directly on streaming + pronunciation
- `[Nachmani et al, 2026]-voxtral-realtime-streaming-asr.pdf` — Voxtral: streaming ASR, vLLM serving, word boundaries
- `[Defossez et al, 2024]-moshi-speech-text-foundation-model-for-realtime-dialogue.pdf` — Moshi: full-duplex speech, real-time by design
- `[Labiausse et al, 2026]-simultaneous-speech-to-speech-translation-without-aligned-data.pdf` — Hibiki Zero: simultaneous streaming translation via GRPO
- `[Labiausse et al, 2025]-high-fidelity-simultaneous-speech-to-speech-translation.pdf` — Hibiki: decoder-only simultaneous translation
- `[Zeghidour et al, 2025]-streaming-seq2seq-with-delayed-streams-modeling.pdf` — delayed-stream seq2seq formulation
- `[Kudlur et al, 2026]-moonshine-v2-ergodic-streaming-encoder-asr.pdf` — Moonshine v2: ergodic sliding-window ASR
- `[Wan et al, 2026]-streaming-speech-recognition-with-decoder-only-llms-and-latency-optimization.pdf` — streaming ASR with decoder-only LLM
- `[Sekoyan et al, 2025]-canary-parakeet-efficient-multilingual-asr-and-ast.pdf` — Canary 1B v2 / Parakeet-TDT

Key references (in `./refs.bib`):

- `[@shi2021coca_mdd]` — CoCA-MDD: streaming coupled cross-attention MDD (2111.08191)
- `[@defossez2024moshi]` — Moshi: real-time speech-text foundation model
- `[@nachmani2025voxtral]` — Voxtral: Mistral streaming ASR
- `[@kyutai2025hibiki_zero]` — Hibiki Zero: zero-alignment streaming translation
- `[@kyutai2024hibiki]` — Hibiki: simultaneous speech translation
- `[@timkey2025delayed_streams]` — delayed streams seq2seq streaming formulation
- `[@rybakov2025moonshine_v2]` — Moonshine v2: ergodic streaming encoder
- `[@tang2025streaming_llm_asr]` — streaming ASR with decoder-only LLMs
- `[@jain2025canary]` — Canary 1B v2 / Parakeet-TDT

Upstream dependencies:

- Track 05: GOP-SF pipeline (chunked approach reuses this directly)
- Track 07: from-scratch streaming model is the long-horizon version of Phase 3/4 here
