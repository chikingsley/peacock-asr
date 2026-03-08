# Citrinet TODO

## Stage 1

- [x] Load `nvidia/stt_en_citrinet_256_ls` in an isolated NeMo env.
- [x] Record tokenizer type, vocab size, decoder class count, and output tensor
  shape on one sample.
- [x] Record effective frame rate / stride from one known utterance.

## Stage 2

- [x] Build a tiny NeMo manifest for a few LibriSpeech samples.
- [x] Test stock posterior extraction on that manifest.
- [x] Decide whether stock posterior outputs are useful for GOP-SF at all.

## Stage 3

- [x] Create and pin the 41-token ARPABET tokenizer assets.
- [x] Document exact token order and blank handling.

## Stage 4

- [x] Fine-tune Citrinet with the phoneme tokenizer on a tiny preflight run.
- [x] Stage the first real `train_clean_100` run command and output layout.
- [x] Export the full `train_clean_100` / `dev_clean` manifests for a real run.
- [ ] Run the first GPU-backed `train_clean_100` fine-tune.
- [ ] Export/evaluate through the normal `P003` scoring contract.
