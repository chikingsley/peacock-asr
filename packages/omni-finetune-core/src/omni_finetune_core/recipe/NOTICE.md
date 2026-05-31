# `recipe/` — Meta's wav2vec2-ASR recipe glue (owned in-house copy)

These files are copied verbatim from Meta's **omnilingual-asr** repository
(`workflows/recipes/wav2vec2/asr/`,
<https://github.com/facebookresearch/omnilingual-asr>), which is licensed under a
BSD-style license (see the per-file headers). They are **not** published on PyPI — the
pip `omnilingual-asr` package ships the models and dataset readers but not this recipe
entry point — which is the only reason this code is in-housed here instead of consumed
as a dependency.

Why owned-in-house rather than a gitignored vendored checkout:

- It is a small, stable surface (~1.2k lines) that imports only from the installed
  `fairseq2` and `omnilingual_asr` packages plus its own siblings — verified: it imports
  cleanly against the **pip** `omnilingual_asr`, with no need for the rest of the repo.
- A single tracked copy beats two gitignored checkouts silently drifting in the
  per-language projects.

**Policy:** treat this directory as third-party. It is excluded from our ruff + ty
strictness (see the root `pyproject.toml`). Keep changes minimal and clearly marked; our
own typed harness lives in the parent `omni_finetune_core` package, not here. To refresh
against upstream, re-copy the directory and re-apply any local patches.
