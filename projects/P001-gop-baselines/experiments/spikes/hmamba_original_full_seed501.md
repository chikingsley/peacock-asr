# HMamba Full Run: original, full split, seed=501

This run answers the next concrete `P001` question after the bounded
`xlsr-espeak` pilot: can the phone-level HMamba adaptation run end-to-end
against the canonical full `original` backend feature cache?

Command:

- `CACHE_DIR=/home/simon/github/peacock-asr/.cache uv run peacock-asr run --backend original --hmamba --seed 501 --device cuda --workers 8`

Key outcome:

- full dataset, no `--limit`
- reused canonical cached `original` features from the repo-root `.cache`
- both splits loaded from cache (`cache_hits=2`, `cache_misses=0`)
- HMamba training completed successfully on GPU

Result:

- backend: `original (checkpoint-8000)`
- mode: `hmamba`
- phones: `47369`
- `PCC = 0.6341`
- `PCC 95% CI = [0.6287, 0.6395]`
- `MSE = 0.0813`

Artifacts:

- checkpoint dir:
  `/home/simon/github/peacock-asr/.cache/checkpoints/2026-03-08_103353_original__checkpoint-8000_seed501`
- metrics:
  `/home/simon/github/peacock-asr/.cache/checkpoints/2026-03-08_103353_original__checkpoint-8000_seed501/eval_metrics.json`
- run info:
  `/home/simon/github/peacock-asr/.cache/checkpoints/2026-03-08_103353_original__checkpoint-8000_seed501/run_info.json`
- W&B:
  `https://wandb.ai/peacockery/peacock-asr-p001-gop-baselines/runs/fp6g466r`

Notes:

- this is still the `P001` phone-level HMamba adaptation, not a full upstream
  hierarchical HMamba reproduction
- a small compatibility shim was required so current `P001` code can load the
  legacy cached `UtteranceFeats` objects serialized under the old
  `peacock_asr.gopt_model` module path
- this is a real full-run score, but still a single-seed exploratory scorer
  result rather than a paper-close multi-seed canonical baseline
