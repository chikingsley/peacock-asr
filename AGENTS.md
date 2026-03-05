# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Peacock-ASR is a Python 3.13 research CLI for pronunciation assessment (GOP scoring) against SpeechOcean762. There is no web server or database; the product is a CLI tool (`peacock-asr`) and a test suite.

### System dependencies

- `espeak-ng` and `libsndfile1` must be installed (required by `phonemizer` and `soundfile` Python packages).

### Dev commands

All commands use `uv run` from the workspace root:

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Lint | `uv run ruff check src/ tests/` |
| Format check | `uv run ruff format --check src/ tests/` |
| Type check | `uv run ty check` |
| Tests | `uv run pytest` |
| CLI help | `uv run peacock-asr --help` |
| Quick eval | `uv run peacock-asr run --backend xlsr-espeak --limit 5 --device cpu` |

### Non-obvious caveats

- **Ruff scope**: Running `ruff check .` (without path restriction) will lint the `references/` directory which contains third-party code with thousands of pre-existing issues. Always scope lint to `src/` and `tests/`.
- **No GPU required**: All tests and the CLI run on CPU. The one CUDA-specific test is auto-skipped when no GPU is present.
- **Model downloads**: First run of `peacock-asr run` downloads HuggingFace models (~1 GB for xlsr-espeak). These are cached in `.cache/` under the workspace.
- **Dataset download**: `peacock-asr download` fetches SpeechOcean762 from HuggingFace Hub. No `HF_TOKEN` is required for this public dataset.
- **`references/` directory**: Contains third-party reference codebases (moshi, gopt-transformer, etc.) that are NOT part of the main project. Ignore for lint/test/build.
- **`justfile` recipes**: These are for RunPod remote GPU management and are not needed for local dev.
