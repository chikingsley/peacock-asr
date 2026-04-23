# P014 Implementation Notes

`P014-hippo-reproduction` now has a runnable read-aloud path under `src/p014`.
It still does not reproduce the full HiPPO paper.

## Sources checked

- ACL paper: `https://aclanthology.org/2025.ijcnlp-long.45/`
- ACL PDF: `https://aclanthology.org/2025.ijcnlp-long.45.pdf`
- local extracted paper notes:
  `/home/simon/github/peacock-asr/docs/markdown/[Yan et al, 2025]-hippo-exploring-a-novel-hierarchical-pronunciation-assessment-approach-for-spoken-languages/paper.md`
- official public repo checked on `2026-04-20`:
  `https://github.com/bicheng1225/HIPPO`

The public GitHub repo currently exposes only a README on `main`. The paper is
still the operative reference.

## What runs now

The supported path is the standardized package in `src/p014`.

- real Speechocean762 annotations are loaded from the Hugging Face dataset
- cached ConPCO SSL and prosody features are loaded from
  `a2d8a4v/SpeechOcean762_for_ConPCO`
- ModernBERT word embeddings are computed and cached locally
- the training CLI runs end to end and writes reproducible artifacts under
  `artifacts/`

The preserved legacy branch under `hippo/`, `data/`, and `scripts/` still
contains placeholder logic and should not be used for reproduction work.

## Latest run

The latest full read-aloud run completed on `2026-04-20` on the Spark GPU.

- command:
  `uv run --extra train p014 train-read-aloud --epochs 100 --batch-size 25 --device auto --run-name read_aloud_spark_seed22_e100 --json`
- artifact:
  [artifacts/read_aloud_spark_seed22_e100/summary.json](../artifacts/read_aloud_spark_seed22_e100/summary.json)
- train/test split:
  `2500 / 2500`
- best epoch:
  `9`

### Table 2 comparison

| Metric | Current run | Paper Table 2 | Gap |
| --- | ---: | ---: | ---: |
| Phone MSE | 0.124 | 0.080 | +0.044 |
| Phone PCC | 0.137 | 0.657 | -0.520 |
| Word Accuracy PCC | 0.172 | 0.630 | -0.458 |
| Word Total PCC | 0.166 | 0.643 | -0.477 |
| Utt Accuracy PCC | 0.218 | 0.791 | -0.573 |
| Utt Fluency PCC | 0.228 | 0.845 | -0.617 |
| Utt Prosody PCC | 0.197 | 0.837 | -0.640 |
| Utt Total PCC | 0.200 | 0.816 | -0.616 |

The current run is operationally valid. It is not close to the paper's Table 2
performance yet.

## Main remaining gaps

1. The active model still omits the paper's GOP feature stream.
   Paper consequence:
   Section 2.2 uses CTC-based GOP features as part of the pronunciation input.
   Current code:
   [src/p014/data.py](/home/simon/github/peacock-asr/projects/P014-hippo-reproduction/src/p014/data.py)
   and [src/p014/model.py](/home/simon/github/peacock-asr/projects/P014-hippo-reproduction/src/p014/model.py)
   only use cached SSL, duration, energy, phone IDs, and ModernBERT features.

2. The active hierarchy is still approximate.
   Paper consequence:
   Sections 2.2 to 2.4 describe a specific Conv-LLaMA and attention-pooling
   design.
   Current code:
   [src/p014/blocks.py](/home/simon/github/peacock-asr/projects/P014-hippo-reproduction/src/p014/blocks.py)
   and [src/p014/model.py](/home/simon/github/peacock-asr/projects/P014-hippo-reproduction/src/p014/model.py)
   implement a close scaffold, but not a block-for-block match.

3. The paper reports five-trial averages.
   Paper consequence:
   Appendix B averages metrics over five independent trials.
   Current code:
   [src/p014/training.py](/home/simon/github/peacock-asr/projects/P014-hippo-reproduction/src/p014/training.py)
   runs one seed per invocation, so the latest result is a single-trial number.

4. The free-speaking path from Appendix D is still missing.
   Paper consequence:
   raw speech must be transcribed with Whisper-large-v3, converted through
   `g2pE`, aligned, and scored with deletion-ignore, substitution-aligned, and
   insertion-zero rules.
   Current state:
   only the typed config exists in
   [src/p014/config.py](/home/simon/github/peacock-asr/projects/P014-hippo-reproduction/src/p014/config.py).
   There is no active `src/p014` implementation for that scenario.

## What is true now

1. The project is standardized around `uv`, `ruff`, `ty`, `pytest`, and
   `pydantic`.
2. The read-aloud path runs end to end on real data and writes artifacts.
3. The current results are still far below the paper.
4. The repo is a partial HiPPO reproduction scaffold, not a faithful paper
   reproduction.
