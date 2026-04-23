# P014 HiPPO Reproduction

`P014-hippo-reproduction` now follows the same project shape as the other
experiment folders:

- `pyproject.toml` is the canonical dependency definition.
- `uv` is the package manager and lockfile owner.
- `ruff`, `ty`, and `pytest` are wired into the project config.
- `pydantic` validates the paper-faithful YAML configs.

## Current status

The active package under `src/p014` now runs a real read-aloud experiment on
Speechocean762 using cached ConPCO SSL and prosody features plus ModernBERT
word embeddings. The preserved legacy code under `hippo/`, `data/`, and
`scripts/` is still only for inspection.

The latest full run completed on `2026-04-20` on the Spark GPU and wrote
[artifacts/read_aloud_spark_seed22_e100/summary.json](./artifacts/read_aloud_spark_seed22_e100/summary.json).
Its best metrics were:

- phone MSE `0.124`
- phone PCC `0.137`
- utterance-total PCC `0.200`

The paper's Table 2 targets for HiPPO* are materially better:

- phone MSE `0.080`
- phone PCC `0.657`
- utterance-total PCC `0.816`

As checked on `2026-04-20`, the authors' public GitHub repository
`bicheng1225/HIPPO` exists, but its public `main` branch only contains a
README. The paper remains the primary implementation reference.

## Quick start

```bash
uv sync --extra dev --extra train
uv run p014-audit
uv run p014-show-config --scenario free_speaking
uv run --extra train p014 train-read-aloud --device auto --run-name read_aloud_local
uv run pytest
uv run ruff check .
uv run ty check
```

## Configs

The typed configs in `configs/` now track the paper defaults from Appendix B:

- hidden size `24` for phone, word, and utterance levels
- `1` MHSA head inside each Conv-LLaMA block
- `3` heads for word- and utterance-level attention pooling
- Adam with learning rate `0.001`
- batch size `25`
- `5` trials
- `100` epochs
- best epoch selected by minimum phone-level MSE

Scenario-specific overrides:

- `hippo_free_speaking.yaml`: Table 1 setting with curriculum learning and
  CONO enabled
- `hippo_read_aloud.yaml`: Table 2 setting with curriculum learning and CONO
  disabled

## Audit output

`p014-audit` reports the current gap between the paper and the active package.
The main findings are:

- the read-aloud path now runs end to end on real data, but its metrics are
  still far below Table 2
- the active model still omits the paper's GOP feature stream and simplifies
  parts of the hierarchical architecture
- the paper's five-trial reporting protocol is not yet reproduced
- the Appendix D free-speaking pipeline is still missing

Detailed notes live in [docs/IMPLEMENTATION_NOTES.md](./docs/IMPLEMENTATION_NOTES.md).
