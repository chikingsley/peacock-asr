# P016 Russian Phone Compare

This is the no-training prototype for the strict pronunciation pipeline:

1. audio -> Qwen3-ASR-1.7B -> word hypothesis
2. ASR text -> recognizer-specific target G2P phone sequences
3. audio -> ZIPA phone recognizer
4. audio -> facebook/wav2vec2-xlsr-53-espeak-cv-ft phone recognizer
5. canonical phones vs recognized phones -> PER and alignment

The displayed practice sentence is not used as the scoring reference. It is only there so the
tester has something to say. The scoring reference comes from the ASR hypothesis.

The two default recognizer lanes intentionally use different target G2P backends:

- ZIPA target phones use Russian MFA for Russian and `espeak-ng` for English.
- XLSR-eSpeak target phones use `espeak-ng --ipa=3`, matching the eSpeak-labeled recognizer.

The eval script also emits Russian diagnostic lanes for ZIPA+Charsiu and XLSR+MFA so target
backend changes can be checked against the same recognizer output.

## Remote Setup

On `gmk-server`:

```bash
cd ~/github/peacock-asr/projects/P016-russian-phone-compare
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv sync --extra zipa
```

MFA is optional at process startup but required for the Russian MFA lane. MFA's Kaldi/OpenFst
binaries come from conda-forge, so keep that environment separate from the `uv` Python env,
but still inside this project:

```bash
scripts/bootstrap_mfa.sh
```

That creates the removable local tree `.mfa/`:

- `.mfa/env/bin/mfa`
- `.mfa/root`
- `.mfa/bin/micromamba`

The app auto-detects `.mfa/env/bin/mfa`. If MFA is missing, it falls back to `espeak-ng` for
G2P and marks that in the output.

ZIPA ONNX is expected in one of these places:

```text
$ZIPA_ONNX
./artifacts/zipa-large-crctc-ns-800k/model.onnx
./artifacts/zipa-large-crctc-ns-800k/model.fp16.onnx
```

The helper script downloads the ZIPA repo and ONNX files:

```bash
scripts/bootstrap_zipa.sh
```

Prefetch the two Hugging Face checkpoints so the first microphone run does not block on model
downloads:

```bash
~/.local/bin/uv run hf download facebook/wav2vec2-xlsr-53-espeak-cv-ft
~/.local/bin/uv run hf download Qwen/Qwen3-ASR-1.7B
~/.local/bin/uv run hf download charsiu/g2p_multilingual_byT5_tiny_16_layers_100
```

## Run

```bash
~/.local/bin/uv run p016-app --server-name 0.0.0.0 --server-port 7860
```

Microphone capture must be opened from a browser-secure origin. Use the SSH tunnel and open the
localhost URL:

```bash
ssh -f -N -L 7860:127.0.0.1:7860 gmk-server
open http://127.0.0.1:7860
```

Opening `http://gmk-server:7860` can show the page, but the browser can block microphone devices
because that is plain HTTP on a non-localhost host.

CLI:

```bash
~/.local/bin/uv run p016-compare analyze path/to/audio.wav --language ru --json
```
