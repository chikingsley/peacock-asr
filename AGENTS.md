# Peacock ASR Agent Notes

## Markdown Formatting

- Format every active Markdown file with `mdformat 1.0.0`, `mdformat-gfm 1.0.0`, and `mdformat-frontmatter 2.1.2`.
- Always pass `--wrap no`; prose paragraphs stay on one physical line. Never hand-wrap Markdown prose to a fixed column width.
- The repository pre-commit hook is canonical. After Markdown edits, run `uvx --from mdformat --with mdformat-gfm --with mdformat-frontmatter mdformat --wrap no <paths>`.
- Preserve the exclusions in `.pre-commit-config.yaml` for generated environments, runtime data, vendored recipes, and archives.

## Python Tooling

- Use `uv` for Python execution, dependency management, Python versions, and Python-backed tools.
- Do not run bare `python`, `python3`, `pip`, or `python -m`.
- Use project console scripts through `uv run --project <project> <script> ...` for repo workflows.
- Add or reuse a console script when a workflow should be repeatable.

## Parakeet Training Metric Policy

- Before changing a Parakeet training recipe, read `packages/parakeet-finetune-core/README.md`.
- CTC recipes should monitor validation WER for best-checkpoint selection.
- TDT/RNNT recipes should enable `model.compute_eval_loss = True` after restoring the base model and changing vocabulary, then monitor `val_loss` for best validation checkpoints.
- BF16 TDT/RNNT validation WER can be noisy after tokenizer changes; use fp32 checkpoint eval for WER gates.
- New Parakeet recipes must inspect the restored model/YAML config rather than assuming `compute_eval_loss`; some NVIDIA transducer configs ship with it disabled.
