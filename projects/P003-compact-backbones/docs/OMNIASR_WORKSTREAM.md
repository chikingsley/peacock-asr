# OmniASR CTC 300M Workstream

## Status

The stock OmniASR CTC 300M v2 probe is runnable, and the first phoneme
adaptation hook is now in place:

- local ARPABET SentencePiece tokenizer assets build cleanly
- a local Omni model/tokenizer card pair is generated under `code/omni/cards`
- direct fairseq2 custom-card loading still fails on the `final_proj` shape
  mismatch
- stock checkpoint load plus `final_proj` replacement to `41` classes works on
  CUDA and can be wrapped by the upstream inference pipeline

## Why It Is Not Drop-In

`omniASR_CTC_300M_v2` is not on the same training/eval contract as the current
`P003` backbones.

Key differences:

- stack: `fairseq2` / `omnilingual-asr`, not the current Transformers trainer
- checkpoint format: native fairseq2 model cards / `.pt`, not our existing `hf:` phoneme repos
- output target: written-text tokenizer vocab (`~10k` tokens in v2), not the 41-token ARPABET phoneme CTC head used by `P003`

That means the stock 300M CTC checkpoint cannot be evaluated as a phoneme
posterior backend without adaptation. It is not comparable to the existing
`wav2vec2-base`, `HuBERT-base`, `w2v-BERT`, or `wav2vec2-large` phoneme runs.

## Current Probe Surface

Use the project-local launcher, which creates the correct Python 3.12
fairseq2/Omnilingual runtime on demand:

```bash
uv run --project projects/P003-compact-backbones \
  python projects/P003-compact-backbones/code/launch_omniasr_ctc_300m_v2_local.py \
  --check-only
```

Under the hood it:

- runs the probe under Python 3.12
- installs the local `third_party/omnilingual-asr` checkout editably
- injects the Intel oneTBB runtime required by `fairseq2n`
- loads `omniASR_CTC_300M_v2` through the upstream Omnilingual inference stack

Entrypoints:

- `projects/P003-compact-backbones/code/launch_omniasr_ctc_300m_v2_local.py`
- `projects/P003-compact-backbones/code/launch_omniasr_ctc_300m_v2_probe.py`

It is intentionally a text-ASR probe first. It validates that:

- the local `third_party/omnilingual-asr` checkout is wired correctly
- the fairseq2 inference stack can load `omniASR_CTC_300M_v2`
- we can start measuring the stock model before touching phoneme adaptation

## Current Phoneme Adaptation Hook

Build local phoneme tokenizer/model assets and preflight the adapted load path:

```bash
uv run --project projects/P003-compact-backbones \
  python projects/P003-compact-backbones/code/launch_omniasr_ctc_300m_v2_phoneme_local.py \
  --device cuda
```

This now does three concrete things:

- builds a `41`-token ARPABET SentencePiece tokenizer under
  `projects/P003-compact-backbones/code/omni/tokenizers/arpabet_41_spm`
- writes local fairseq2 asset cards under
  `projects/P003-compact-backbones/code/omni/cards`
- proves that loading the stock `omniASR_CTC_300M_v2` checkpoint and replacing
  `final_proj` with a `41 x 1024` phoneme head is viable on CUDA

Current preflight result:

- tokenizer loaded: yes
- custom local card with `restrict: false`: still fails on `final_proj.*`
- fallback load mode: `stock-plus-head-replacement`
- pipeline readiness: yes

## Current Scoring Contract

Omni is no longer only “trainable.” The current scoring contract is:

- train into the canonical local run directory
- score through the `P003` runtime using an `omni:<run-dir>` backend
- load the fine-tuned checkpoint through a persistent Python 3.12 worker

The worker uses:

- stock `omniASR_CTC_300M_v2` architecture
- local `41`-token phoneme tokenizer assets
- `final_proj` replacement to the `41`-class phoneme head
- the latest fairseq2 checkpoint weights from the training output dir

The canonical eval sweep is:

- [`eval_omniasr_ctc_300m_v2_phoneme.yaml`](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/sweeps/final/eval_omniasr_ctc_300m_v2_phoneme.yaml)

The canonical backend form is:

```text
omni:/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/checkpoints/omniasr-ctc-300m-v2-phoneme-en
```

## Concrete Next Steps

1. Run the real phoneme fine-tuning launcher, not just preflight.
2. Verify the first output checkpoint resolves cleanly through the `omni:` backend.
3. Let the automatic post-train scorer create the canonical eval sweep.
4. Compare Omni against the other `P003` backbones under the same GOPT contract.

## Current Recommendation

- `Parakeet 0.6B`: already queued in the chain
- `omniASR_CTC_300M_v2`: no longer blocked on loader/debug work; the next task
  is the actual phoneme fine-tuning run

## Sources

- local checkout:
  `projects/P004-training-from-scratch/third_party/omnilingual-asr/README.md`
- local cards:
  `projects/P004-training-from-scratch/third_party/omnilingual-asr/src/omnilingual_asr/cards/models/rc_models_v2.yaml`
