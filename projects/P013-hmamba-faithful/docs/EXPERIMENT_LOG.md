# P013 Experiment Log

## 2026-04-03 official-style rerun

Run root:

- `runs/repro-20260403-094834`

Goal:

- Re-run HMamba with the original repo's official `run.sh` settings:
  batch size `50`, warmup `300`, 20 epochs, seeds `824 17 2413 168 623`,
  `deXent` with `a=0.7`, and phone-MSE-based checkpoint selection.

Tracking:

- Trackio project: `p013-hmamba-faithful`
- Trackio runs: `seed824`, `seed17`, `seed2413`, `seed168`, `seed623`

Per-seed results:

| Seed | Phone PCC | Phone MSE | Utt Total PCC | Word Total PCC | PER | Inline MDD F1 | Corpus MDD Precision | Corpus MDD Recall | Corpus MDD F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 824 | 0.7149 | 0.0664 | 0.8051 | 0.7039 | 0.02753 | 0.5280 | 0.6824 | 0.4951 | 0.5738 |
| 17 | 0.7215 | 0.0652 | 0.8131 | 0.7017 | 0.02711 | 0.5335 | 0.7079 | 0.5049 | 0.5894 |
| 2413 | 0.7136 | 0.0667 | 0.8130 | 0.6973 | 0.02740 | 0.5266 | 0.6820 | 0.4859 | 0.5675 |
| 168 | 0.7172 | 0.0661 | 0.8111 | 0.7007 | 0.02704 | 0.5365 | 0.6556 | 0.5719 | 0.6109 |
| 623 | 0.7092 | 0.0676 | 0.7992 | 0.6919 | 0.02728 | 0.5193 | 0.6677 | 0.4935 | 0.5676 |

5-seed means:

- Phone PCC: `0.7153`
- Phone MSE: `0.0664`
- Utterance total PCC: `0.8083`
- Word total PCC: `0.6991`
- PER: `0.02727`
- Inline validation MDD F1 proxy: `0.5288`
- Corpus-level MDD precision: `0.6791`
- Corpus-level MDD recall: `0.5103`
- Corpus-level MDD F1: `0.5818`

Interpretation:

1. APA is stable and reasonably close to the paper's HMamba result.
2. PER is essentially on target.
3. MDD improved materially once post-recog corpus evaluation replaced the inline
   proxy, but it still trails the paper's reported `0.6385` F1.
4. The inline `result.csv` MDD field should not be used as the final MDD result
   for this branch.

Exact stage-3 parity check:

- On 2026-04-03, the original `P014-hmamba-original/eval_mdd/mdd_result.sh`
  path was re-run over all five seed outputs using repo-local Kaldi
  `align-text` and `compute-wer` binaries built under `third_party/kaldi`.
- The exact Kaldi outputs are saved as `mdd_result_kaldi_raw.txt` and
  `mdd_result_kaldi.txt` in each seed directory.
- The exact Kaldi MDD precision/recall/F1 values match the existing
  corpus-level `mdd_result.json` values to rounding error across all five
  seeds, so the remaining shortfall versus the paper is not caused by the
  evaluation path.
