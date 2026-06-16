# Parakeet Fine-Tune Core

Shared NeMo/Parakeet training glue for Peacock ASR language projects.

## Contract

Every language project should expose console scripts in its own `pyproject.toml` and call this
package from a small project adapter:

```python
from parakeet_finetune_core import ParakeetProject
from parakeet_finetune_core.ctc import train_ctc_main
from parakeet_finetune_core.eval import eval_main

PROJECT = ParakeetProject(
    name="tajik",
    language="tgk_Cyrl",
    root=ROOT,
    default_train_manifest=ROOT / "data/parakeet/train.jsonl",
    default_validation_manifest=ROOT / "data/parakeet/dev.jsonl",
    default_tokenizer_dir=ROOT / "data/tokenizers/parakeet/tgk_cyrl_spe_bpe_v1024",
)

def train_ctc(argv=None):
    return train_ctc_main(PROJECT, argv)

def evaluate(argv=None):
    return eval_main(PROJECT, argv)
```

Run it as a project command:

```bash
uv run --project projects/tajik-asr tajik-parakeet-train-ctc --max-steps 1000
uv run --project projects/tajik-asr tajik-parakeet-eval --limit 80 --device cpu
```

Do not launch project workflows through direct interpreters or helper-script paths. The reusable
surface is `uv run --project <language-project> <console-script> ...`.

## Common vs Language-Specific

Common in this package:

- cache defaults under the language project
- NeMo tokenizer helper invocation
- CTC trainer config and dry-run YAML emission
- generic NeMo fine-tune recipe wrapper
- TDT loss-init fix for duration-bin outputs
- fused TDT loss/WER setup
- TDT run logging and checkpoint defaults
- CTC and TDT/TDT+CTC checkpoint evaluation via NeMo `transcribe`
- WER scoring with project-provided normalization

Language-specific in each `<language>-asr` project:

- model artifact paths or model IDs
- train/dev/test NeMo manifests
- tokenizer artifact path and name
- default evaluation checkpoint and model kind
- output run root
- default CTC/TDT run names
- text normalization and curation policy outside the Parakeet trainer

## Training Metric Policy

Checkpoint and early-readout metrics are architecture-specific:

- CTC runs monitor validation WER. CTC validation loss is cheap, but WER is the selection signal
  used by the existing Parakeet CTC recipes and does not have the TDT/RNNT decode issue.
- TDT/RNNT runs enable `model.compute_eval_loss = True` after restoring the base model and
  changing vocabulary, then monitor `val_loss` for best validation checkpoints. BF16 TDT/RNNT
  validation WER can be noisy after tokenizer changes, so use fp32 checkpoint eval for WER gates.
- New Parakeet recipes must inspect the restored model/YAML config rather than assuming this flag.
  Some NVIDIA transducer configs ship with `compute_eval_loss: false`; the shared TDT harness
  overrides it intentionally.

## TDT Status

TDT is supported here as shared mechanics, not as a fully promoted production recipe. The reusable
pieces are the parts that must be identical across projects: the RNNT loss class-count fix, fused
joint loss/WER, optional encoder freeze/unfreeze, run logging, and checkpoint handling. Project
experiments still own their ablation choices, data gates, and promotion criteria.
