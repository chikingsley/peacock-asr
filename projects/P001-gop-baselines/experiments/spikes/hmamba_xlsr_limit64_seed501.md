# HMamba Spike: xlsr-espeak, limit=64, seed=501

Date: 2026-03-08

Purpose:
- first runnable HMamba-style scorer result inside the `P001` feature contract
- compare against the existing `GOPT` path without rebuilding a new dataset stack

Run:
- command:
  - `uv run peacock-asr run --backend xlsr-espeak --hmamba --seed 501 --workers 8 --device cpu --limit 64`
- W&B:
  - `https://wandb.ai/peacockery/peacock-asr-p001-gop-baselines/runs/2yg0zy6g`
- checkpoint dir:
  - `/home/simon/github/peacock-asr/projects/P001-gop-baselines/.cache/checkpoints/2026-03-08_080324_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft_seed501`

Result:
- backend: `xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft)`
- mode: `hmamba`
- limit: `64 train / 64 test`
- seed: `501`
- phones scored: `1072`
- PCC: `0.2822`
- PCC 95% CI: `[0.2261, 0.3364]`
- MSE: `0.0540`

Interpretation:
- this is a real scorer result
- this is **not** a canonical `P001` table entry
- the run is a bounded pilot on a 64/64 subset
- the current HMamba adaptation is phone-level and uses the upstream transformer-mode path, not the full bimamba dependency stack

Runtime findings:
- the existing `P001` prepared cache for `xlsr-espeak` was stale/corrupt and unusable for this run
- the parallel scalar worker path in `P001` had two real bugs that blocked feature-based scorers:
  - exact-type checking on worker returns was too strict
  - the worker returned score-only GOP results without occupancies, but the feature path still needed occupancies
- both issues were fixed locally before this spike completed

Next step:
- repair/regenerate the full `P001` xlsr feature cache cleanly
- rerun HMamba on the full dataset
- only then compare HMamba against the canonical multi-seed `GOPT` line
