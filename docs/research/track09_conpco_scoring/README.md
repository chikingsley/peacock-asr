# Track 09 Paper Workspace: Contrastive Ordinal Scoring for Pronunciation Assessment

Working title:

- **ConPCO Regularization for CTC-Based Pronunciation Scoring: An Incremental Integration Study**

Purpose:

- Integrate ConPCO (Contrastive Phonemic Ordinal) regularization into our existing GOPT pipeline.
- Measure the isolated effect of the loss function versus the full HierCB architecture.
- Follow lab methodology: one change at a time, compute-fair, reproducible.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`
- Runbook: `./RUNBOOK.md`

Draft files:

- `manuscript.md` (primary writing file)

Citation convention:

- Use Pandoc/Quarto citekeys, e.g. `[@yan2025conpco]`.
- All citekeys are in `./refs.bib`.

Process:

1. Freeze Methods and dataset/eval protocol (inherit from Track 05).
2. Lock experiment table schema and report all runs in one format.
3. Write Results only from reproducible logs/artifacts.
4. Run evidence audit before finalizing claims.

Papers (PDFs in `./papers/`, index at `./papers/INDEX.md`):

- See `./papers/INDEX.md` for the full Yan & Chen dependency chain
- ConPCO repo: <https://github.com/bicheng1225/ConPCO> (cloned at `references/ConPCO/`)
- Precomputed features on HF: a2d8a4v/SpeechOcean762_for_ConPCO

Key references (in `./refs.bib`):

- `[@yan2023pco]` — PCO: ordinal entropy loss (ASRU 2023)
- `[@yan2024hiertfr]` — HierTFR/HierCB: hierarchical architecture (ACL 2024)
- `[@yan2024hiergat]` — HierGAT: graph attention approach (TASLP 2024, **PDF missing — paywalled**)
- `[@yan2025conpco]` — ConPCO: + CLAP contrastive alignment (ICASSP 2025)
- `[@li2025multitask_pretraining]` — Multi-task pretraining on HierCB (Interspeech 2025)
- `[@yan2025muffin]` — MuFFIN: unified MDD+APA (TASLP 2025)
- `[@yan2025hippo]` — HiPPO: hierarchical + interpretable (AACL 2025)
- `[@chen2025read_to_hear]` — Read to Hear: zero-shot LLM scoring (EMNLP 2025, different group)
