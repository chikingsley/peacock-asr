# Track 08 Workspace: Unscripted / ASR-Conditioned CAPT

Working titles:

- **Unscripted Pronunciation Assessment via ASR-Conditioned Phoneme Scoring**
- **Chunked GOP-SF for Low-Latency Pronunciation Scoring: Toward Real-Time Feedback**

Purpose:

- Define the unscripted / free-speaking CAPT problem for this repo.
- Evaluate whether ASR output can serve as the temporary transcript for
  pronunciation scoring when canonical text is not known in advance.
- Measure how ASR transcript errors propagate into pronunciation scoring.
- Evaluate low-latency and streaming variants as deployment constraints inside
  that broader unscripted setting.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`
- Unscripted note: `./UNSCRIPTED_ASR_CAPT_PLAN.md`
- Data taxonomy: `./DATA_TAXONOMY.md`

Draft files:

- `manuscript.md` (primary writing file)

Citation convention:

- Use Pandoc/Quarto citekeys, e.g. `[@shi2021coca_mdd]`.
- All citekeys are in `./refs.bib`.

Process:

1. Define the unscripted / ASR-conditioned task contract first.
2. Separate pronunciation scoring, ASR recognition, and semantic judging into
   distinct metrics.
3. Treat low-latency or streaming variants as system constraints, not as the
   only research question.
4. Write Results only from reproducible logs/artifacts.
5. Run evidence audit before finalizing claims.

Papers (PDFs in `./papers/`):

- `[Yan et al, 2025]-hippo-hierarchical-apa-unscripted-speech.pdf` — HiPPO: unscripted / spoken-language APA
- `[Chen et al, 2023]-multipa-multitask-open-response-pronunciation.pdf` — MultiPA: open-response pronunciation assessment
- `[Chen et al, 2025]-textpa-zero-shot-pronunciation-llm.pdf` — Read to Hear: textual descriptions + coherence-aware LLM scoring
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

- `[@yan2025hippo]` — HiPPO: hierarchical APA for spoken / unscripted language
- `[@chen2024multipa]` — MultiPA: open-response pronunciation assessment
- `[@chen2025read_to_hear]` — Read to Hear: zero-shot pronunciation scoring from textual speech descriptions
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

- Track 05: transcript-known GOP-SF pipeline
- Track 06: multimodal / LLM-based pronunciation scoring
- Track 07: from-scratch phoneme backbones

Scope note:

- The earlier chunked-streaming draft remains relevant, but it is now treated as
  one sub-path inside Track 08 rather than the entire definition of the track.
