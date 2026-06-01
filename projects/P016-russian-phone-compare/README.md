# P016 — Two-Path Pronunciation Scoring

No-training, generalist pronunciation scoring built on a universal phone recognizer + canonical
G2P + phonological-feature distance — aimed at languages with **no L2 learner database**.

One funnel, two target sources (see `docs/free_speaking_architecture_status.md` for the why):

```text
text -> G2P -> canonical IPA   vs   audio -> phone recognizer -> produced IPA   ->   PER / PFER
```

- **read-aloud** — target text is the known reference. No ASR. Clean ceiling of the funnel.
- **free-form** — target text is recovered by ElevenLabs Scribe v2 (`superwhisper-api`).

Recognizer: **ZIPA** (universal multilingual IPA phone recognizer, ONNX). XLSR-eSpeak was
compared and dropped — ZIPA won. G2P target: espeak-ng (universal), with a Russian MFA lane.

## Setup (gmk-server)

```bash
cd ~/github/peacock-asr/projects/P016-russian-phone-compare
~/.local/bin/uv sync --extra zipa
scripts/bootstrap_zipa.sh     # ZIPA repo + ONNX into artifacts/zipa-large-crctc-ns-800k/
scripts/bootstrap_mfa.sh      # optional: Russian MFA lane (.mfa/); espeak-ng fallback otherwise
```

ZIPA ONNX is expected at `$ZIPA_ONNX` or `./artifacts/zipa-large-crctc-ns-800k/model.onnx`
(or `model.fp16.onnx`). Free-form (Scribe) needs `superwhisper-api` auth on the host — off-Mac it
mirrors the Superwhisper cache via `SUPERWHISPER_MAC_HOST`/ssh and mints a realtime key.

## Run the two-path eval

```bash
# Build a multilingual FLEURS manifest (any FLEURS config codes).
uv run python scripts/build_fleurs_manifest.py --out-dir runs/two_paths --per-language 20 \
  --languages en_us ru_ru fr_fr de_de es_419 it_it fa_ir hi_in tr_tr ja_jp

# Read-aloud only (local, no ASR):
uv run python scripts/eval_two_paths.py --manifest runs/two_paths/manifest.jsonl \
  --out-dir runs/two_paths/eval

# Add the free-form path (target = Scribe ASR; needs superwhisper auth + network):
uv run python scripts/eval_two_paths.py --manifest runs/two_paths/manifest.jsonl \
  --out-dir runs/two_paths/eval --free-form
```

Outputs `summary.csv`, `words.csv` (per-word target vs recognized phones — the G2P diagnostic),
`results.jsonl`, and `report.md` (avg PER/PFER by language × mode × lane).
