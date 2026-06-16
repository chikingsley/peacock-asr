# archive — superseded / completed-experiment scripts

These ran during the TDT investigation and are kept for provenance. The live training path is now
the `parakeet_finetune_core` package (`tajik-parakeet-train-tdt`); the live eval is `../eval_ckpt.py`;
the live manifest builder is `../gate1b_curator_manifest.py`.

| file | what it was | superseded by |
|---|---|---|
| `gate0a_repro.py` | reproduced the TDT-head stagnation on the 110M hybrid | (finding recorded in `../README.md`) |
| `gate0b_v3_inspect.py` | v3 preflight (tokenizer coverage, blank/duration indices, aux-CTC head) | (finding recorded) |
| `gate_decode_check.py` | proved fp32 decode works (stall was under-learning, not a bug) | (finding recorded) |
| `gate1_build_manifest.py` | small FLEURS→NeMo manifest builder | `../gate1b_curator_manifest.py` (full curator corpus) |
| `tdt_finetune.py` | A0-vs-B ablation that validated the loss-init fix | loss-init fix baked into `parakeet_finetune_core.tdt` |
| `run_big.py` | bespoke big-run harness (manual Trainer + hand-rolled logging) | `parakeet_finetune_core.tdt` (`tajik-parakeet-train-tdt`, proper recipe) |
