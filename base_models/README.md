# Shared base models

This is Peacock ASR's canonical local store for reusable pretrained weights. Model files are ignored by Git; this inventory records their source and exact Hub revision. Training and benchmark commands should use these local paths instead of creating another Hugging Face cache copy.

## Verified Hugging Face snapshots

| Local path                              | Hub repository                       | Revision                                   | Primary weight SHA-256                                             |
| --------------------------------------- | ------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------ |
| `whisper/c1tech-whisper-base-persian`   | `C1Tech/whisper_base_persian`        | `effb325ff62f5811d5eff9c3be093464b6637e50` | `34b1b4d4f6c4325720ccbb09e5ab3060876fdff359688e24e863e31b4faef7ea` |
| `whisper/openai-whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo`      | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1` |
| `qwen/qwen3-asr-0.6b-hf`                | `Qwen/Qwen3-ASR-0.6B-hf`             | `6aa69c382e2b426eee1f5870d4c95859a74b6445` | `d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35` |
| `qwen/qwen3-asr-1.7b-hf`                | `Qwen/Qwen3-ASR-1.7B-hf`             | `057a3b044fcd31c433e7971ab40d68d20e7eae6d` | `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1` |
| `parakeet/parakeet-ctc-109m-farsi`      | `Peacockery/parakeet-ctc-109m-farsi` | `27418c9ffa05a8a0fe66fc5900ca0181a3ad25a7` | `b11fc64a92e4cc457c2693356e85d296143b702bc8e25e2f7fe8ff485a4afa72` |

The revisions above were downloaded and verified with `hf cache verify --local-dir` on 2026-07-09. Recreate a snapshot with:

```bash
hf download REPOSITORY --revision REVISION --local-dir base_models/FAMILY/NAME
hf cache verify REPOSITORY --revision REVISION --local-dir base_models/FAMILY/NAME \
  --fail-on-missing-files
```

## Existing local weights

- `omni/`: Meta OmniASR v2 weights downloaded directly from `https://dl.fbaipublicfiles.com/mms/`:

  | Local file                           | SHA-256                                                            |
  | ------------------------------------ | ------------------------------------------------------------------ |
  | `omniASR-CTC-300M-v2.pt`             | `8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50` |
  | `omniASR-CTC-1B-v2.pt`               | `354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c` |
  | `omniASR-CTC-3B-v2.pt`               | `fa7f662c326842bb80561db97631ae3c48d911aec579654a1e8414c26caf9089` |
  | `omniASR-LLM-300M-v2.pt`             | `703105f3e4612a030110a1ba1ae419d5d4f75c6dc10175a38b2ddbb4bdf3817b` |
  | `omniASR-LLM-1B-v2.pt`               | `cceb4d9ebac3d168a6af6b26c62ce11bafc562b38976c6bfa87e7d60422c6da5` |
  | `omniASR-LLM-3B-v2.pt`               | `6c454275703c96e5f68ce3b6315345251dfc70028b1f795cd60972150bb2a044` |
  | `omniASR_tokenizer_written_v2.model` | `8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e` |

- `parakeet/`: Parakeet CTC/TDT NeMo and Hugging Face weights. `parakeet-ctc-109m-farsi/model.nemo` is the canonical Persian CTC checkpoint for evaluation and NeMo forced alignment. The older flat `ctc.nemo` has SHA256 `ab04147024caa2687e522eedba08b8d8a711a7c3813c1abe05771448871a72d2`; its archive identity differs from the Hub checkpoint, so it remains a local legacy artifact rather than a default.

These older local files predate this registry, so their source revisions still need provenance reconciliation. Keep them here; do not move them into project-specific model directories.
