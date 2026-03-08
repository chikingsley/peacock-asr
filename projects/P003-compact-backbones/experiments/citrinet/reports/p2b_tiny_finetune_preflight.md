# Citrinet P2-B Tiny Fine-Tune Preflight

Date:

- 2026-03-07

Goal:

- validate the direct `P2-B` path, not the stock `P2-A` wrapper path
- prove that we can:
  - build a phoneme-target tokenizer
  - build NeMo manifests from the LibriSpeech-alignments source
  - swap the Citrinet decoder vocabulary
  - start training without touching the main `P003` HF path

Artifacts:

- tokenizer vocab:
  [code/citrinet/tokenizers/arpabet_41_wpe/vocab.txt](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/tokenizers/arpabet_41_wpe/vocab.txt)
- tokenizer metadata:
  [metadata.json](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/tokenizers/arpabet_41_wpe/metadata.json)
- preflight manifests:
  [train.jsonl](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/preflight/manifests/train.jsonl)
  [eval.jsonl](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/preflight/manifests/eval.jsonl)
- 1-step training report:
  [report.json](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/preflight/tiny_finetune_step1/report.json)

Commands used:

```bash
projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  projects/P003-compact-backbones/code/citrinet/scripts/build_p2b_assets.py
```

```bash
CUDA_VISIBLE_DEVICES='' \
projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  projects/P003-compact-backbones/code/citrinet/scripts/tiny_finetune_preflight.py \
  --max-steps 1 \
  --batch-size 2 \
  --output-dir projects/P003-compact-backbones/experiments/citrinet/preflight/tiny_finetune_step1
```

Observed facts:

- stock model:
  `nvidia/stt_en_citrinet_256_ls`
- direct adaptation route:
  `change_vocabulary(..., new_tokenizer_type='wpe')`
- LibriSpeech-alignments source works for NeMo manifests after exporting audio
  bytes to concrete files
- the direct training loop starts and completes the requested sanity step
- global step reached:
  `1`

Important caveat:

- the `wpe` path expands the effective decoder vocabulary to `44`, not `41`
- NeMo/BERT tokenizer injection adds:
  - `[CLS]`
  - `[SEP]`
  - `[MASK]`

Interpretation:

- the direct `P2-B` path is viable
- the next technical decision is whether to:
  - accept a 44-class first real run, or
  - invest a bit more in a cleaner tokenizer path to keep the decoder exactly at
    the repo-standard phoneme target

Current recommendation:

- use this preflight as the go/no-go gate for a first real Citrinet run
- treat the `44`-class tokenizer issue as a known risk, not a blocker
- the next real experiment should be a GPU-backed Citrinet fine-tune on a larger
  manifest, likely `train_clean_100` before attempting anything larger

Status after this preflight:

- go directly to `P2-B`
- do not spend more time on the stock `P2-A` wrapper path
- first real run command is:

```bash
projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  projects/P003-compact-backbones/code/citrinet/scripts/train_citrinet_p2b.py \
  --train-manifest projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/manifests/train.jsonl \
  --eval-manifest projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/manifests/eval.jsonl \
  --output-dir projects/P003-compact-backbones/experiments/citrinet/checkpoints/citrinet_256_p2b_train_clean_100 \
  --max-steps 5000 \
  --batch-size 16 \
  --accumulate-grad-batches 1 \
  --num-workers 4 \
  --lr 1e-4 \
  --seed 17
```
