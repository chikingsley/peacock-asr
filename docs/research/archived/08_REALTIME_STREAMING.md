# 08: Real-Time and Streaming Pronunciation

Status: stub -- papers and repos collected, no analysis or validation.

This track covers real-time / streaming pronunciation scoring where
audio is processed as it arrives, not after the full utterance.

Last updated: 2026-03-02

## The Idea

Score pronunciation while the person is still speaking. This requires:

- chunked audio processing (not full-utterance)
- low latency (scores within ~500ms of word completion)
- score stability (provisional scores that don't jump around)

Two sub-problems:
a) known transcript -- score against expected pronunciation in real time
b) open-ended -- no transcript, model must recognize and score together

This is the hardest track. None of the other tracks (05, 06, 07) solve
streaming. They all assume you have the complete utterance first.

## Papers We Have

### Streaming speech models

    2602.11298 (rl_alignment_speech/)
    "Voxtral Realtime"
    Mistral's streaming ASR model. Sub-second latency, matches
    offline quality. Native streaming with word-boundary behavior.
    vLLM serving support. Previously identified as streaming backbone
    candidate.

    2410.00037 (streaming_realtime/)
    "Moshi: speech-text foundation model for real-time dialogue"
    Kyutai's full-duplex speech model. Real-time by design.
    Previously identified as implementation substrate candidate
    (logit access, alignment tooling, fine-tuning hooks).

    2602.11072 (streaming_realtime/)
    "Hibiki Zero: Simultaneous Speech-to-Speech Translation"
    Zero-alignment simultaneous translation using GRPO.
    Demonstrates streaming speech processing without aligned data.

    2502.03382 (streaming_realtime/)
    "Hibiki: High-Fidelity Simultaneous Speech Translation"
    Decoder-only model for simultaneous multimodal translation.

    2509.08753 (streaming_realtime/)
    "Streaming Sequence-to-Sequence Learning with Delayed Streams"
    Formulation for streaming multimodal seq2seq. Defines chunk
    cadence and delayed commit behavior.

    2602.12241 (streaming_realtime/)
    "Moonshine v2: Ergodic Streaming Encoder ASR"
    Sliding-window attention for low-latency edge deployment.
    Interesting for on-device pronunciation scoring.

    2601.22779 (streaming_realtime/)
    "Streaming ASR with Decoder-Only LLMs"
    Streaming ASR using LLM decoder with latency optimization.

### Multimodal speech models (could be adapted for streaming)

    2503.20215 (streaming_realtime/)
    "Qwen2.5-Omni Technical Report"
    End-to-end multimodal model for speech and text.

    2505.08699 (streaming_realtime/)
    "Granite-speech: open-source speech-aware LLMs"
    IBM's speech-aware LLMs. Strong English ASR.

    2511.23404 (streaming_realtime/)
    "LFM2 Technical Report"
    Liquid Foundation Models for efficient on-device deployment.
    Interesting for edge inference.

### Streaming pronunciation specifically

    2111.08191 (mispronunciation_detection/)
    "CoCA-MDD: Streaming Mispronunciation Detection"
    Coupled cross-attention for streaming end-to-end MDD.
    Only paper we have that directly addresses streaming + pronunciation.

    2509.03256 (streaming_realtime/)
    "End-to-End Speech Assessment Models for NOCASA 2025"
    Children's word-level assessment. Not streaming but relevant
    for real-time word-level scoring.

## Code References (Not Validated)

These repos were identified in the adjacent repo scan but have NOT
been tested or evaluated for quality:

    references/ADVANCED-transcription/speech-to-text/kyutai
    Kyutai (Moshi) code. Previously identified as best place to
    add a first scoring module. Has alignment and timestamp tooling.

    references/voxtral-finetune
    Voxtral training collator/prompt formatting patterns.

    references/Finetune-Voxtral-ASR
    Minimal LoRA fine-tuning scripts for Voxtral.

    Kyutai public repos (moshi, moshi-finetune, hibiki*)
    Streaming operational patterns. Not cloned locally.

## CAPT Systems (Full Systems With Real-Time Aspects)

    2105.05182 (capt_systems/)
    "PTeacher: Personalized Pronunciation Training"
    Personalized CAPT with exaggerated corrective feedback.

    2207.00774 (capt_systems/)
    "CAPT -- Speech Synthesis is Almost All You Need"
    Using speech synthesis as primary CAPT component.

    2502.07575 (capt_systems/)
    "Prior Efforts in Building CAPT"
    Joint modeling for CAPT combining APA and MDD.

    2506.19315 (capt_systems/)
    "JCAPT: Joint Modeling Approach for CAPT"
    Unified system for assessment + diagnosis.

    2507.06202 (capt_systems/)
    "V(is)owel: Interactive Vowel Chart"
    Visual feedback for pronunciation improvement.

## What We Don't Have Yet

- no streaming pronunciation prototype of any kind
- no latency measurements on any streaming model
- no validation that any of these repos/models actually work for
  pronunciation scoring (they're ASR/translation models)
- no understanding of how to attach a pronunciation scoring head
  to a streaming model
- the Voxtral/Kyutai choices came from what code happened to exist
  nearby, not from systematic evaluation
- no comparison between streaming approaches

## Open Questions

- can GOP-SF scoring work incrementally on partial utterances, or
  does it fundamentally need the full sequence?
- what's the minimum viable latency for useful pronunciation feedback?
- is streaming pronunciation scoring even necessary for MVP, or is
  ~1-2 second full-utterance processing fast enough?
- for known-transcript scoring, can we just run CTC + GOP on each
  word as it completes (chunked non-streaming)?
- how do provisional scores behave? do they converge or oscillate?

## Relationship to Other Tracks

- track 1 (05): the CTC + GOP pipeline could potentially be chunked
  to process word-by-word, which might be "good enough" for real-time
  without full streaming architecture
- track 2 (06): LLM-based scoring is too slow for true streaming
  but could work for per-sentence scoring with acceptable latency
- track 3 (07): a from-scratch streaming model (like Moshi) could
  be trained with pronunciation scoring built in from the start
