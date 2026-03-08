# Track 10 Paper Workspace: Compact Backbones for Pronunciation Assessment

Working title:

- **Do You Need 300M Parameters? Compact CTC Backbones for GOP-Based Pronunciation Scoring**

Purpose:

- **Phase 0 (complete):** Validate the CTC fine-tuning recipe on a larger SSL
  backbone so the same training path can be reused for smaller backbones.
- Compare smaller CTC backbones (wav2vec2-base 95M, HuBERT-base 95M, Citrinet
  10M) against our xlsr-53 300M baseline as GOP feature extractors.
- Measure the compute-accuracy tradeoff at the backbone level.
- Include HMamba (Mamba-based scoring head) as an alternative to GOPT transformer.
- Follow lab methodology: one change at a time, compute-fair, reproducible.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`

Draft files:

- `manuscript.md` (primary writing file)
- `RUNBOOK.md` (canonical execution paths)

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

## Current project-local configs

- Training sweep:
  `../experiments/sweeps/final/train_wav2vec2_base.yaml`
- Training sweep:
  `../experiments/sweeps/final/train_hubert_base.yaml`
- Eval sweep:
  `../experiments/sweeps/final/eval_wav2vec2_base.yaml`
- Eval sweep:
  `../experiments/sweeps/final/eval_hubert_base.yaml`
- Eval sweep:
  `../experiments/sweeps/final/eval_w2v_bert.yaml`
- Local launcher:
  `../code/launch_hubert_base_local.py`
- Local benchmark wrapper:
  `../code/benchmark_hubert_base_local.py`
- Scoring benchmark:
  `../code/p003_compact/bench_scoring.py`
- Citrinet workstream:
  `./CITRINET_WORKSTREAM.md`

Project-local sweeps and manifests are canonical. Historical leftovers now live
under `../experiments/legacy/`.

## Launch commands

Phase 1: wav2vec2-base (95M) CTC fine-tuning:

```bash
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/train_wav2vec2_base.yaml
uv run --project projects/P003-compact-backbones wandb agent <sweep-id>
```

Phase 1: wav2vec2-base evaluation:

```bash
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/eval_wav2vec2_base.yaml
uv run --project projects/P003-compact-backbones wandb agent <sweep-id>
```

Phase 0: w2v-BERT-2.0 (600M) evaluation:

```bash
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/eval_w2v_bert.yaml
uv run --project projects/P003-compact-backbones wandb agent <sweep-id>
```

Phase 1: HuBERT-base local training:

```bash
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/launch_hubert_base_local.py
```

Scoring optimization loop:

```bash
uv run --project projects/P003-compact-backbones peacock-benchmark-scoring \
  --backend hf:Peacockery/w2v-bert-phoneme-en \
  --limit 16 \
  --split both \
  full
```

That command is the fast checkpoint for testing scoring fixes. It writes
phase-by-phase timing JSON under
`../experiments/benchmarks/scoring/reports/`.

Warm the persistent `k2` topology cache before full eval runs:

```bash
uv run --project projects/P003-compact-backbones python -m p003_compact.cli \
  prewarm-k2 \
  --backend hf:Peacockery/wav2vec2-base-phoneme-en \
  --split both
```

No `--preprocessed-dataset` is needed for `wav2vec2-base` because its feature
extractor is just audio normalization, so `set_transform()` on-the-fly is
faster than loading preprocessed features.

Phase 2 (`Citrinet-256`) is intentionally isolated from the Hugging Face CTC
path. Use the dedicated workstream note and directories under:

- `../code/citrinet/`
- `../env/citrinet/`
- `../experiments/citrinet/`
- `../third_party/citrinet/`

Current result snapshot:

- `wav2vec2-base` (95M): `0.640 +/- 0.009` PCC
- `HuBERT-base` (95M): `0.6489 +/- 0.0093` PCC
- `w2v-BERT-2.0` (600M): `0.6755 +/- 0.0066` PCC
- `Citrinet-256` (10M, NeMo branch): `0.5574 +/- 0.0133` PCC

Key insight:

- No published paper has used wav2vec2-base or HuBERT-base as a CTC backbone
  for GOP-based pronunciation assessment on SpeechOcean762. This is a genuine
  gap.
- Citrinet now has a real branch result rather than a pure feasibility note,
  but it is not competitive with the SSL CTC backbones in the current form.
