# P003 Citrinet-256 Workstream

Purpose:

- keep the Citrinet branch isolated from the existing Hugging Face / Transformers
  backbone path
- make Phase 2 concrete before we touch NeMo code
- preserve a clean experiment boundary between:
  - `P1`: drop-in HF CTC backbones
  - `P2`: NeMo Citrinet adaptation

Canonical references:

- Phase plan: [./ABLATION_PLAN.md](/home/simon/github/peacock-asr/projects/P003-compact-backbones/docs/ABLATION_PLAN.md)
- Evidence ledger: [./EVIDENCE_LEDGER.md](/home/simon/github/peacock-asr/projects/P003-compact-backbones/docs/EVIDENCE_LEDGER.md)
- NeMo model docs: <https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/models.html>
- NeMo fine-tuning docs: <https://docs.nvidia.com/nemo/speech/nightly/asr/configs.html>
- Citrinet-256 model card: <https://huggingface.co/nvidia/stt_en_citrinet_256_ls>

## 1. What We Are Actually Testing

There are two different Citrinet questions. They should not be mixed.

### P2-A: Stock feasibility

Question:

- can stock `nvidia/stt_en_citrinet_256_ls` produce posteriors that are usable at
  all inside our GOP pipeline?

What this tests:

- tokenizer mismatch pain
- frame-rate / stride pain
- wrapper feasibility

What this does **not** test:

- whether a phoneme-retuned Citrinet can be competitive

### P2-B: Phoneme-retuned Citrinet

Question:

- if we replace the stock tokenizer/output vocabulary and fine-tune against our
  41-token ARPABET target, can a ~10M model remain competitive?

What this tests:

- true tiny-backbone viability
- deployment-grade size / speed tradeoff

This is the real experiment. `P2-A` only exists to reduce integration risk.

## 2. Why Citrinet Must Be Isolated

The stock checkpoint is not a drop-in HF CTC model.

Official constraints:

- Citrinet in NeMo is a subword CTC model instantiated via
  `EncDecCTCModelBPE`.
- The stock `stt_en_citrinet_256_ls` model uses a SentencePiece unigram
  tokenizer with vocabulary size `256`.
- NeMo supports fine-tuning by updating the tokenizer/vocabulary, including
  loading a pretrained model and swapping tokenizer/vocabulary during
  fine-tuning.

Practical implication:

- do **not** bolt NeMo-specific lifecycle into `p003_compact/backends/hf_ctc.py`
- do **not** pollute the existing HF training path
- keep a separate Citrinet adapter / env / manifests / runbook until the
  experiment earns promotion

## 3. Recommended Project Layout

Use these directories as the Citrinet boundary inside `P003`.

```text
projects/P003-compact-backbones/
  code/
    citrinet/
      README.md
      TODO.md
      manifests/
      tokenizers/
      scripts/
  env/
    citrinet/
      README.md
  experiments/
    citrinet/
      README.md
      manifests/
      sweeps/
      logs/
      reports/
  third_party/
    citrinet/
      README.md
```

Division of responsibility:

- `code/citrinet/`
  Citrinet-only code and wrappers
- `env/citrinet/`
  NeMo-specific environment/bootstrap notes
- `experiments/citrinet/`
  manifests, sweeps, logs, reports
- `third_party/citrinet/`
  pinned external references, config pointers, copied examples if needed

## 4. Work Sequence

### Stage 0: Freeze current P003 backbone table

Done:

- `wav2vec2-base` scored
- `HuBERT-base` scored
- `w2v-BERT-2.0` scored

This matters because Citrinet should now be run as a **new branch**, not mixed
into the tail of Phase 1.

### Stage 1: NeMo preflight

Deliverable:

- prove we can instantiate the stock checkpoint cleanly in an isolated env

Success criteria:

- load `nvidia/stt_en_citrinet_256_ls`
- run transcription on a tiny audio sample
- inspect tokenizer assets and decoder vocab size
- record time stride / posterior frame count on a known utterance

No scoring yet.

### Stage 2: Stock posterior feasibility (`P2-A`)

Deliverable:

- a wrapper that exports posterior-like outputs from stock Citrinet for a small
  batch of utterances

Success criteria:

- can produce a stable tensor per utterance
- can measure effective frames per second / stride
- can determine whether GOP-SF can even consume the outputs without retuning

Decision gate:

- if stride/tokenization mismatch makes the outputs meaningless, do not waste
  time on lexicon hacks; move directly to `P2-B`

Current outcome:

- Stage 2 did its job and is closed
- stock outputs are not a serious GOP backend path
- the branch stayed focused on `P2-B`
- `P2-B` now has a real 5-seed SpeechOcean eval result:
  - `PCC 0.5574 +/- 0.0133`
  - `MSE 0.0977 +/- 0.0029`
  - backend: `nemo:Peacockery/citrinet-256-phoneme-en`

### Stage 3: Phoneme tokenizer build

Deliverable:

- a 41-token ARPABET tokenizer directory usable by NeMo fine-tuning

Success criteria:

- tokenizer assets versioned under `code/citrinet/tokenizers/`
- explicit token order pinned
- blank handling documented

### Stage 4: Phoneme-retuned Citrinet (`P2-B`)

Deliverable:

- fine-tuned Citrinet checkpoint with 41-token ARPABET output

Preferred path:

- initialize from stock `stt_en_citrinet_256_ls`
- update tokenizer/vocabulary
- reinitialize decoder as needed
- fine-tune on LibriSpeech

Current practical decision:

- accept the `wpe` tokenizer path for the first real run even though it expands
  the decoder from `41` to `44` classes via `[CLS]`, `[SEP]`, and `[MASK]`
- treat that as a first-run engineering compromise, not the final ideal tokenizer
- use a real `train_clean_100` fine-tune to decide whether Citrinet is worth
  deeper cleanup
- use a project-local Vast control plane, not shared infra code and not RunPod
  scripts, for the first sustained GPU run

Readiness status:

- full `train_clean_100` / `dev_clean` manifests exported
- dataset size validated:
  - `28,538` train rows / `98.595h`
  - `2,703` eval rows / `5.133h`
- full-manifest 1-step dry run completed
- current blocker is only GPU placement, not NeMo integration

Operational surface:

- offer search:
  [vast_search_offers.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/vast_search_offers.py)
- instance status:
  [vast_show_instances.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/vast_show_instances.py)
- destroy:
  [vast_destroy_instance.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/vast_destroy_instance.py)
- full launch / sync / run / teardown:
  [orchestrate_vast_citrinet_trainclean100.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/orchestrate_vast_citrinet_trainclean100.py)
- template upsert:
  [vast_upsert_template.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/vast_upsert_template.py)
- volume inspection / creation:
  [vast_show_volumes.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/vast_show_volumes.py),
  [vast_create_volume.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/vast_create_volume.py)

Canonical Vast template:

- name: `p003_citrinet_nemo_ssh`
- hash: `ab21436ee2fe8894e2aef98578790fe9`

Rerun strategy:

- first run may still pay full bootstrap + data sync tax
- after a successful run, create a local volume from the finished instance
- relaunch with:
  - `--attach-volume-name <volume_name>`
  - default mount `/data`
- when the volume already contains:
  - `train_clean_100_full`
  - the Citrinet venv under `/data/p003-citrinet/env/citrinet/.venv`
  later runs can skip both the dataset sync and the NeMo install

### Stage 5: GOP/GOPT evaluation

Deliverable:

- either a NeMo-native backend for `P003` eval
- or an export/conversion path that gives us a stable posterior backend

Constraint:

- evaluation contract stays the same as the other `P003` backbones
- only the backbone changes

## 5. What To Build First

The first code should be small and disposable:

1. `code/citrinet/scripts/inspect_stock_model.py`
   - load the model
   - print tokenizer type, vocab size, decoder classes
   - run one forward pass
   - print output shape and effective frame count

2. `code/citrinet/scripts/export_dummy_manifest.py`
   - build a tiny LibriSpeech-style NeMo manifest from a few local samples

3. `experiments/citrinet/reports/stock_feasibility.md`
   - one page with findings from Stage 1 and Stage 2

This is the correct first slice. Do not start with a giant training script.

## 6. Decision Rules

Use these to keep the branch disciplined.

- If Stage 1 fails at basic model load:
  - fix environment only
  - do not touch scorer code
- If Stage 2 shows the stock output is unusable for GOP:
  - stop investing in stock wrapper quality
  - move to phoneme-retuned Citrinet
- If Stage 4 cannot converge to a sane phoneme CTC model:
  - publish the negative result as a compact-backbone feasibility finding
- Do not merge Citrinet utilities into shared `p003_compact` until a full scored
  result exists

## 7. Immediate Next Tasks

1. Export the full `train_clean_100` / `dev_clean` manifests.
2. Run the first GPU-backed `P2-B` fine-tune.
3. Inspect convergence and decoder behavior.
4. Only then decide whether the `44`-class tokenizer path needs refinement.
