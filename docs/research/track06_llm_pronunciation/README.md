# Track 06 Paper Workspace: Multimodal LLM Pronunciation Assessment

Working title:

- **Multimodal LLM Pronunciation Assessment: LoRA Fine-Tuning vs CTC+GOP Pipelines on SpeechOcean762**

Purpose:

- Evaluate whether LoRA-fine-tuned multimodal LLMs (Phi-4, Qwen2-Audio) can match or
  exceed our CTC+GOP+GOPT pipeline on SpeechOcean762.
- Produce a fair head-to-head comparison on the same eval set and same metric (phone-level PCC).
- Quantify the inference cost trade-off: LLM forward pass versus CTC+GOP+GOPT.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`
- Track narrative: `../06_LLM_PRONUNCIATION.md`

Draft files:

- `manuscript.md` (primary writing file)

Citation convention:

- Use Pandoc/Quarto citekeys, e.g. `[@ryu2025phi4_lora_pronunciation]`.
- All citekeys are in `./refs.bib`.

Process:

1. Reproduce Phi-4 LoRA result on SpeechOcean762 (Phase 1).
2. Evaluate Qwen2-Audio-7B as alternative (Phase 2).
3. Compare LLM scores with our GOP PCC on identical eval set (Phase 3).
4. Measure and report inference cost (Phase 4).
5. Write Results only from reproducible logs/artifacts.
6. Run evidence audit before finalizing claims.

Papers (PDFs in `./papers/`):

- `[Ahn et al, 2025]-lora-mllm-apa-mdd-joint.pdf` — Phi-4 LoRA (KEY paper)
- `[Fu et al, 2024]-pronunciation-assessment-multimodal-llm.pdf` — Fu et al.: MLLM pronunciation with Data2vec2+Qwen
- `[Wang et al, 2025]-lmm-pronunciation-assessment-gpt4o.pdf` — Wang et al.: GPT-4o zero-shot for pronunciation (Microsoft)
- `[Fang et al, 2025]-mllm-automated-speaking-assessment-sfmt.pdf` — Fang et al.: unified MLLM for speaking assessment (SFMT)
- `[Wang et al, 2025]-fine-tuning-lmm-automatic-pronunciation-assessment.pdf` — fine-tuning Qwen2-Audio-7B for APA, multi-granularity
- `[Parikh et al, 2026]-zero-shot-speech-llms-l2-multi-aspect-evaluation.pdf` — Parikh et al.: zero-shot Qwen2-Audio for L2 evaluation
- `[Chen et al, 2023]-multipa-multitask-open-response-pronunciation.pdf` — Chen et al.: MultiPA multi-task model
- `[Chen et al, 2025]-textpa-zero-shot-pronunciation-llm.pdf` — Chen et al.: TextPA zero-shot
- `[Yan et al, 2025]-hippo-hierarchical-apa-unscripted-speech.pdf` — Yan et al.: HiPPO
- `[Kim et al, 2022]-ssl-pronunciation-assessment-wav2vec-hubert.pdf` — Kim et al.: SSL-based APA (bridge paper)

Key references (in `./refs.bib`):

- `[@ryu2025phi4_lora_pronunciation]` — Ryu et al.: Phi-4 LoRA, PCC 0.668-0.675 (Accuracy)
- `[@fu2024mllm_pronunciation]` — Fu et al.: Data2vec2+Qwen MLLM, PCC 0.713-0.777
- `[@wang2025gpt4o_pronunciation]` — Wang et al.: GPT-4o zero-shot (Microsoft)
- `[@fang2025sfmt_speaking_assessment]` — Fang et al.: SFMT unified MLLM
- `[@yang2025qwen2audio_finetuning]` — Qwen2-Audio fine-tuning, reported PCC 0.77 (accuracy)
- `[@parikh2026zeroshot_speech_llm]` — Parikh et al.: zero-shot Qwen2-Audio
- `[@chen2024multipa]` — Chen et al.: MultiPA open-response
- `[@chen2025read_to_hear]` — Chen et al.: TextPA zero-shot
- `[@yan2025hippo]` — Yan et al.: HiPPO hierarchical (PCC 0.83 utterance-level)
- `[@kim2022ssl_pronunciation_assessment]` — Kim et al.: SSL-based APA
- `[@gong2022gopt_transformer_pronunciation_assessment]` — GOPT baseline
