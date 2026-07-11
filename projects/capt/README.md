# capt — Two-Path Pronunciation Scoring

No-training, generalist pronunciation scoring built on a universal phone recognizer + canonical G2P + phonological-feature distance — aimed at languages with **no L2 learner database**.

One funnel, two target sources (see `docs/free_speaking_architecture_status.md` for the why):

```text
text -> G2P -> canonical IPA   vs   audio -> phone recognizer -> produced IPA   ->   PER / PFER
```

- **read-aloud** — target text is the known reference. No ASR. Clean ceiling of the funnel.
- **free-form** — target text is recovered by ElevenLabs Scribe v2 (`superwhisper-api`).

Recognizer: **ZIPA** (universal multilingual IPA phone recognizer, ONNX), run in-process via onnxruntime (vendored in `capt.recognize`). XLSR-eSpeak was compared and dropped — ZIPA won. G2P target: per-language routed backends (espeak-ng / Epitran / CharsiuG2P), with ZIPA-distilled FSTs for the no-G2P gap languages (`capt.g2p.models`).

## Package layout

```text
src/capt/
  pipeline.py            orchestrator / public API
  audio.py  asr.py       audio IO; ElevenLabs Scribe v2 lane (free-form target)
  g2p/                   text -> canonical IPA (routing.py, routing.json, text_normalization.py, models/)
  recognize/             audio -> produced IPA (zipa.py + vendored _vendor_zipa.py)
  score/                 align + score (alignment.py, features.py, phones.py)
  cli/                   entry points (eval, manifest, ablation, fetch)
```

## Setup (gmk-server)

```bash
cd ~/github/peacock-asr/projects/capt
~/.local/bin/uv sync
uv run capt-fetch-zipa          # downloads the ZIPA ONNX + tokens into artifacts/
```

The recognizer expects the model at `$ZIPA_ONNX` or `artifacts/zipa-large-crctc-ns-800k/model.onnx` (or `model.fp16.onnx`). Free-form (Scribe) needs `superwhisper-api` auth on the host.

## Run the two-path eval

```bash
# Build a multilingual FLEURS manifest (any FLEURS config codes).
uv run capt-manifest --out-dir runs/two_paths --per-language 20 \
  --languages en_us ru_ru fr_fr de_de es_419 it_it fa_ir hi_in tr_tr ja_jp

# Read-aloud only (local, no ASR):
uv run capt-eval --manifest runs/two_paths/manifest.jsonl --out-dir runs/two_paths/eval

# Add the free-form path (target = Scribe ASR; needs superwhisper auth + network):
uv run capt-eval --manifest runs/two_paths/manifest.jsonl --out-dir runs/two_paths/eval --free-form
```

Outputs `summary.csv`, `words.csv` (per-word target vs recognized phones — the G2P diagnostic), `results.jsonl`, and `report.md` (avg PER/PFER by language × mode × lane).

The per-language G2P routing table (`src/capt/g2p/routing.json`) is rebuilt with `capt-g2p-ablation`; trainable gap-language G2P lives in `experiments/g2p_train/`.
