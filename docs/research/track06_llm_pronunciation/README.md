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

- `2509.02915_lora_pronunciation_assessment.pdf` — Phi-4 LoRA (KEY paper)
- `2509.02915.pdf` — same paper, end_to_end_assessment copy
- `2407.09209v2.pdf` — Fu et al.: MLLM pronunciation with Data2vec2+Qwen
- `2503.11229.pdf` — Wang et al.: GPT-4o zero-shot for pronunciation (Microsoft)
- `2508.12591.pdf` — Fang et al.: unified MLLM for speaking assessment (SFMT)
- `2509.15701.pdf` — fine-tuning Qwen2-Audio-7B for APA, multi-granularity
- `2601.16230.pdf` — Parikh et al.: zero-shot Qwen2-Audio for L2 evaluation
- `2308.12490_multipa_open_response.pdf` — Chen et al.: MultiPA multi-task model
- `2509.14187_read_to_hear_zero_shot_pronunciation.pdf` — Chen et al.: TextPA zero-shot
- `2512.04964_hippo_hierarchical_pronunciation_assessment.pdf` — Yan et al.: HiPPO
- `2204.03863.pdf` — Kim et al.: SSL-based APA (bridge paper)

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
