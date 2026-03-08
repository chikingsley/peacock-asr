# Citrinet Branch

This directory is the isolated `P003` branch for NeMo/Citrinet work.

Why it exists:

- Citrinet is not a drop-in Hugging Face CTC backbone in this project
- the stock checkpoint is a NeMo `EncDecCTCModelBPE`
- the tokenizer/output path is SentencePiece-based and separate from our current
  HF phoneme-head workflow

Rules:

- keep Citrinet-specific code here until the branch produces a complete scored
  result
- do not add NeMo-specific code to `code/p003_compact/` unless it becomes a
  real shared dependency
- prefer small inspection/preflight scripts first

Planned contents:

- `scripts/inspect_stock_model.py`
- `scripts/export_dummy_manifest.py`
- `tokenizers/` for the 41-token ARPABET tokenizer assets
- `manifests/` for tiny NeMo-compatible manifests

Execution plan:

- see
  [../docs/CITRINET_WORKSTREAM.md](/home/simon/github/peacock-asr/projects/P003-compact-backbones/docs/CITRINET_WORKSTREAM.md)
