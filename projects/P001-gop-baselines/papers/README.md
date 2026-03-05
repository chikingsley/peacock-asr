# P001 Paper Sources

P001 uses the shared paper mirror under `docs/papers/`.
PDFs live in `docs/papers/pdf/`; extracted notes live in `docs/papers/markdown/`.

The bibliography file for this project is:
- `projects/P001-gop-baselines/docs/refs.bib`

`file = {...}` entries in `refs.bib` point to:
- `../../../docs/papers/pdf/...`

This means the paper package is self-contained in `P001/docs` while sharing one
central paper mirror across projects.

If later needed, selected PDFs can be copied into this folder for a fully local
project bundle.
