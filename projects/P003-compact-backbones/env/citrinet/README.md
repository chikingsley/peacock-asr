# Citrinet Environment

Keep Citrinet dependencies isolated from the main `P003` env.

Reason:

- the current `P003` environment is built around PyTorch + Transformers
- Citrinet work requires NeMo-specific tooling and tokenizer/model handling
- this branch is still experimental and should not pollute the stable backbone
  path

Baseline reference:

- NeMo installation guidance:
  <https://github.com/NVIDIA-NeMo/NeMo>
- Citrinet model card:
  <https://huggingface.co/nvidia/stt_en_citrinet_256_ls>

Initial target:

- a separate `uv`-managed or containerized environment dedicated to Stage 1
  preflight
- no changes to the main `projects/P003-compact-backbones/pyproject.toml`
  until Citrinet proves worth integrating

Required preflight outcomes:

- stock model loads
- one-file transcription works
- tokenizer metadata is inspectable
- output shape/stride can be measured
