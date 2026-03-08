# P003 Runbook (Compact Backbones)

Canonical W&B sweep specs now live under:

- [`projects/P003-compact-backbones/experiments/sweeps/final/train_wav2vec2_base.yaml`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/sweeps/final/train_wav2vec2_base.yaml)
- [`projects/P003-compact-backbones/experiments/sweeps/final/train_hubert_base.yaml`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/sweeps/final/train_hubert_base.yaml)
- [`projects/P003-compact-backbones/experiments/sweeps/final/eval_wav2vec2_base.yaml`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/sweeps/final/eval_wav2vec2_base.yaml)
- [`projects/P003-compact-backbones/experiments/sweeps/final/eval_hubert_base.yaml`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/sweeps/final/eval_hubert_base.yaml)
- [`projects/P003-compact-backbones/experiments/sweeps/final/eval_w2v_bert.yaml`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/sweeps/final/eval_w2v_bert.yaml)

Project-local wrapper scripts now live under:

- [`projects/P003-compact-backbones/code/launch_hubert_base_local.py`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/launch_hubert_base_local.py)
- [`projects/P003-compact-backbones/code/benchmark_hubert_base_local.py`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/benchmark_hubert_base_local.py)

## 1) Train wav2vec2-base phoneme head

```bash
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/train_wav2vec2_base.yaml
```

Run agent:

```bash
mkdir -p projects/P003-compact-backbones/experiments/logs
nohup uv run --project projects/P003-compact-backbones wandb agent <ENTITY/PROJECT/SWEEP_ID> > projects/P003-compact-backbones/experiments/logs/sweep_<SWEEP_ID>.log 2>&1 &
```

## 2) Evaluate wav2vec2-base as a GOP backend

This sweep is the same scoring contract as `P001`: GOP-SF feature extraction
followed by the GOPT scoring head. The only thing that changes is the backend
that produces phoneme posteriors.

```bash
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/eval_wav2vec2_base.yaml
```

Run agent:

```bash
mkdir -p projects/P003-compact-backbones/experiments/logs
nohup uv run --project projects/P003-compact-backbones wandb agent <ENTITY/PROJECT/SWEEP_ID> > projects/P003-compact-backbones/experiments/logs/sweep_eval_<SWEEP_ID>.log 2>&1 &
```

## 3) Backend used by evaluation

Evaluation uses the generic HF backend path:

```text
hf:Peacockery/wav2vec2-base-phoneme-en
```

That path now resolves to the project-local HF backend implementation in
`projects/P003-compact-backbones/code/p003_compact/backends/hf_ctc.py`, loaded directly
by the shared backend resolver. Future compact-backbone experiments should
follow the same `hf:<repo_id>` pattern. The eval sweeps also set `gopt=true`
explicitly so reruns do not depend on any historical CLI default.

For the 600M validation point, use:

```text
hf:Peacockery/w2v-bert-phoneme-en
```

Launch the evaluation sweep with:

```bash
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/eval_w2v_bert.yaml
```

## 4) Launch HuBERT-base locally

```bash
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/launch_hubert_base_local.py
```

The launcher uses the canonical repo-local HF cache and writes checkpoints to:

```text
projects/P003-compact-backbones/experiments/checkpoints/hubert-base-phoneme-en
```

The target HF repo is:

```text
Peacockery/hubert-base-phoneme-en
```

## 5) Launch HuBERT-base on a generic cloud GPU

On any remote machine with the repo checked out and
`uv sync --project projects/P003-compact-backbones` completed:

```bash
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/launch_hubert_base_local.py
```

That command is intentionally cloud-agnostic. It uses the `P003` project-local
trainer instead of any provider-specific path.

## 6) Run the local HuBERT benchmark wrapper

```bash
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/benchmark_hubert_base_local.py
```

## 7) Benchmark scoring optimizations without a rewrite

Use the project-local scoring benchmark to measure one phase at a time on a
small fixed subset before changing production code:

```bash
uv run --project projects/P003-compact-backbones peacock-benchmark-scoring \
  --backend hf:Peacockery/w2v-bert-phoneme-en \
  --limit 16 \
  --split both \
  --workers 1 \
  full
```

Save only prepared posterior matrices for later CPU experiments:

```bash
uv run --project projects/P003-compact-backbones peacock-benchmark-scoring \
  --backend hf:Peacockery/w2v-bert-phoneme-en \
  --limit 16 \
  --split both \
  --transport-dtype float32 \
  prepare
```

Then benchmark only the scalar GOP phase against that prepared bundle:

```bash
uv run --project projects/P003-compact-backbones peacock-benchmark-scoring \
  --backend hf:Peacockery/w2v-bert-phoneme-en \
  --limit 16 \
  --split both \
  --transport-dtype float32 \
  --workers 8 \
  scalar
```

Benchmark reports are written under:

```text
projects/P003-compact-backbones/experiments/benchmarks/scoring/reports/
```

Prepared posterior bundles are written under:

```text
projects/P003-compact-backbones/experiments/benchmarks/scoring/prepared/
```

## 8) Prewarm the k2 topology cache

`k2` is now the default scalar backend for `P003`, but first-run cold starts are
slower because denominator topologies must be built and cached. Prewarm the
cache once per backend before a full eval sweep:

```bash
uv run --project projects/P003-compact-backbones python -m p003_compact.cli \
  prewarm-k2 \
  --backend hf:Peacockery/wav2vec2-base-phoneme-en \
  --split both
```

This reuses prepared posterior caches when available and populates persistent
topology files under:

```text
projects/P003-compact-backbones/.cache/k2_topologies/
```
