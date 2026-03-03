# Track 10 Paper Workspace: Compact Backbones for Pronunciation Assessment

Working title:

- **Do You Need 300M Parameters? Compact CTC Backbones for GOP-Based Pronunciation Scoring**

Purpose:

- Compare smaller CTC backbones (wav2vec2-base 95M, HuBERT-base 95M, Citrinet 10M) against our xlsr-53 300M baseline as GOP feature extractors.
- Measure the compute-accuracy tradeoff at the backbone level.
- Include HMamba (Mamba-based scoring head) as an alternative to GOPT transformer.
- Follow lab methodology: one change at a time, compute-fair, reproducible.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`

Draft files:

- `manuscript.md` (primary writing file)

Citation convention:

- Use Pandoc/Quarto citekeys, e.g. `[@kim2022ssl_pronunciation]`.
- All citekeys are in `./refs.bib`.

Process:

1. Freeze Methods and dataset/eval protocol (inherit from Track 05).
2. Lock experiment table schema and report all runs in one format.
3. Write Results only from reproducible logs/artifacts.
4. Run evidence audit before finalizing claims.

Key references:

- HMamba repo: <https://github.com/Fuann/hmamba>
- HMamba paper: NAACL 2025 (arXiv: 2502.07575)
- HiPAMA repo: <https://github.com/doheejin/HiPAMA>
- HIA paper: AAAI 2026 (arXiv: 2601.01745)
- Kim et al. SSL pronunciation: Interspeech 2022 (arXiv: 2204.03863)
- Citrinet-256: <https://huggingface.co/nvidia/stt_en_citrinet_256_ls>
- GOPT repo: <https://github.com/YuanGongND/gopt>

Key insight:

- No published paper has used wav2vec2-base or HuBERT-base as a CTC backbone for GOP-based pronunciation assessment on SpeechOcean762. This is a genuine gap.
