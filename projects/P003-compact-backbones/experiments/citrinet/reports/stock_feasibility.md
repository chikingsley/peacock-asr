# Citrinet Stock Preflight

Date:

- 2026-03-07

Environment:

- isolated env:
  `projects/P003-compact-backbones/env/citrinet/.venv`
- Python:
  `3.10.19`
- NeMo:
  `2.7.0`

Command:

```bash
projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  projects/P003-compact-backbones/code/citrinet/scripts/inspect_stock_model.py \
  --audio projects/P001-gop-baselines/third_party/zipa/sample.wav \
  --output projects/P003-compact-backbones/experiments/citrinet/reports/stock_model_probe.json
```

Probe artifact:

- [stock_model_probe.json](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/reports/stock_model_probe.json)

Observed facts:

- model loads successfully via NeMo `ASRModel.from_pretrained`
- class:
  `EncDecCTCModelBPE`
- encoder:
  `ConvASREncoder`
- decoder:
  `ConvASRDecoder`
- tokenizer class:
  `AutoTokenizer`
- tokenizer vocab:
  `256`
- decoder classes:
  `256`
- sample rate:
  `16000`
- configured preprocessor window stride:
  `0.01`

Audio probe:

- sample:
  `projects/P001-gop-baselines/third_party/zipa/sample.wav`
- duration:
  `5.855s`
- alignment length:
  `74`
- alignment inner length:
  `257`
- transcript:
  `"[CLS] mister quilter is the apostle of the middle clas [UNK]s and we are glad to welcome his gospel [SEP]"`

Derived implication:

- effective frame step from the alignment length is about
  `5.855 / 74 = 0.079s`
- that is roughly `79ms` per step, which matches the expected severe time
  reduction problem for GOP-SF

Interpretation:

- Stage 1 preflight passes
- stock Citrinet is usable for inspection and wrapper prototyping
- the stock tokenizer/output space is visibly wrong for direct phoneme GOP use
- `P2-A` remains a wrapper/feasibility exercise only
- `P2-B` is still the real experiment

Next:

1. build a tiny NeMo manifest with the export script
2. probe stock output shapes on a few more utterances
3. decide whether `P2-A` gets a thin backend wrapper or whether we skip straight
   to phoneme retuning
