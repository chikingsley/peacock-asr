# 06: Multimodal LLM Pronunciation Assessment

Status: stub -- papers collected, no analysis done yet.

This track explores using multimodal LLMs (audio + text in, scores out)
for pronunciation assessment, instead of the CTC + GOP pipeline in 05.

Last updated: 2026-03-02

## The Idea

Feed audio + text prompt into a multimodal LLM. The model directly
outputs pronunciation scores. No CTC posteriors, no GOP computation,
no phoneme head. The LLM just rates the pronunciation.

Key differences from track 1 (05):

- no explicit phoneme model -- the LLM handles everything internally
- more flexible -- can give open-ended feedback, not just scores
- heavier inference -- full LLM forward pass vs small CTC head + GOP
- less transparent -- harder to explain why a score is what it is

The leading approach is LoRA fine-tuning on an existing multimodal LLM
(e.g. Phi-4 Multimodal) using pronunciation-scored datasets like
SpeechOcean762.

## Papers We Have (with .md summaries)

### Core -- directly about LLM-based pronunciation scoring

    2509.02915 (end_to_end_assessment/)
    "English Pronunciation Evaluation without Complex Joint Training"
    LoRA fine-tuned Phi-4 multimodal for APA + MDD.
    THE key paper for this track. Reports strong score correlation.
    Also in: rl_alignment_speech/2509.02915v1.md (duplicate)
    Also in: rl_alignment_speech/2509.02915_lora_pronunciation_assessment.md (duplicate)

    2407.09209v2 (rl_alignment_speech/)
    "Pronunciation Assessment with Multi-modal Large Language Models"
    MLLM scoring system design and prompt strategy.

    2503.11229 (rl_alignment_speech/)
    "Exploring the Potential of Large Multimodal Models as Effective
    Alternatives for Pronunciation Assessment"
    Direct comparison of LMMs vs traditional APA methods.

    2508.12591 (rl_alignment_speech/)
    "Beyond Modality Limitations: A Unified MLLM Approach to Automated
    Speaking Assessment"
    Unified MLLM with curriculum learning for speaking assessment.

    2509.15701 (rl_alignment_speech/)
    "Fine-Tuning Large Multimodal Models for Automatic Pronunciation
    Assessment"
    Fine-tuned LMMs for multi-granularity pronunciation scores.

    2601.16230 (rl_alignment_speech/)
    "Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech"
    Zero-shot (no fine-tuning) LLM approach for L2 evaluation.

    2308.12490 (rl_alignment_speech/)
    "MultiPA: A Multi-task Speech Pronunciation Assessment Model for
    Open Response Scenarios"
    Multi-task model for open-ended pronunciation scoring.

    2509.14187 (rl_alignment_speech/)
    "Read to Hear: Zero-Shot Pronunciation Assessment Using Textual
    Descriptions and LLMs"
    Zero-shot via text descriptions -- no audio-scored training data.

    2512.04964 (rl_alignment_speech/)
    "HIPPO: Hierarchical Pronunciation Assessment"
    Hierarchical model with interpretability for APA.

### Related -- ASR-free and end-to-end approaches

    2005.11902v1 (rl_alignment_speech/)
    "ASR-Free Pronunciation Assessment"
    Pronunciation scoring from raw speech, no ASR pipeline.
    Useful as non-ASR-confidence control baseline.

    2204.03863 (end_to_end_assessment/)
    "Automatic Pronunciation Assessment using Self-Supervised Speech
    Representation Learning"
    SSL-based (wav2vec, HuBERT) approach for APA. Bridge between
    track 1 SSL encoders and track 2 end-to-end scoring.

    2310.13974 (end_to_end_assessment/)
    "Automatic Pronunciation Assessment - A Review"
    Comprehensive review of APA methods. Good for understanding
    the landscape.

    2509.03256 (end_to_end_assessment/)
    "Comparison of End-to-End Speech Assessment Models"
    E2E models for children's word-level pronunciation. NOCASA 2025.

### RL/alignment methods (applicable to this track)

    2509.01939 (rl_alignment_speech/)
    "GRPO for Speech Recognition"
    Group relative policy optimization for LLM-based ASR.

    2509.03526 (rl_alignment_speech/)
    "Enhancing Speech LLMs through Reinforced Behavior Alignment"
    RL-based behavior alignment for speech LLMs.

    2507.09929 (rl_alignment_speech/)
    "DPO for Speech Enhancement"
    DPO with perceptual feedback. Method reference.

    2602.13891 (rl_alignment_speech/)
    "GSRM: Generative Speech Reward Model"
    Generative reward model for speech RLHF.

    2511.07931 (rl_alignment_speech/)
    "SpeechJudge: Towards Human-Level Judgment for Speech Naturalness"
    Generative reward model for speech quality.

    2510.00743 (rl_alignment_speech/)
    "From Scores to Preferences: Redefining MOS Benchmarking"
    Preference-based benchmark for speech quality reward models.

### CAPT systems (full systems, some use LLMs)

    2601.14744 (capt_systems/)
    "Unlocking Large Audio-Language Models for Interactive Language
    Learning"
    ALMs for chat-based pronunciation training with instruction tuning.

    2510.04956 (capt_systems/)
    "MuFFIN: Multifaceted Pronunciation Feedback Model"
    Hierarchical neural model for comprehensive feedback.

## Papers We Have But Haven't Read Closely

The rl_alignment_speech/ folder also contains these TTS-focused papers.
They're about RL/alignment methods applied to speech synthesis, not
pronunciation assessment. Useful as methodology references only:

    2409.10157 -- DPO for emotional TTS
    2505.04113 -- Zero-shot TTS intelligibility via preference alignment
    2507.05911 -- DiffRO for LLM-based TTS rewards
    2509.18798 -- GRPO for TTS
    2509.18928 -- DPO for speech diffusion models
    2510.20210 -- Multi-level TTS evaluator

    2025_interspeech_gop2vec_sirigiaju -- GoP2Vec few-shot (more track 1)
    2509.16876 -- Multi-task pretraining with handcrafted features (hybrid)

## What We Don't Have Yet

- no reproduction of any LLM-based pronunciation results
- no comparison between LLM scores and our GOP-based scores on the
  same dataset (SpeechOcean762)
- no understanding of inference cost (LLM forward pass vs CTC + GOP)
- no analysis of which papers are actually good vs just published
- the paper .md summaries are auto-generated and may be low quality

## Open Questions

- does LoRA fine-tuning on SpeechOcean762 overfit (only ~5h of data)?
- how do LLM scores compare to GOP PCC 0.662 on the same eval set?
- can an LLM give per-phone scores or only word/utterance level?
- what's the inference latency? can it work for real-time feedback?
- is the LLM learning pronunciation quality or just ASR confidence?

## Code References

None validated yet. The ADVANCED-transcription repo has some
multimodal speech code but nothing specific to pronunciation LLMs.
