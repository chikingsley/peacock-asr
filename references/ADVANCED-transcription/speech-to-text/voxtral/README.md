# Voxtral Speech to Text

## Scripts

- **`chat-v-transcribe.py`** — Demo of Voxtral's transcription vs chat template differences
- **`modal-training/`** — Fine-tune Voxtral-Mini-3B with LoRA on Modal ([README](modal-training/README.md))
- **`modal-serving/`** — OpenAI-compatible vLLM serving on Modal ([README](modal-serving/README.md))
- **`eval_whisper_baseline.py`** — Standalone Whisper baseline WER evaluation on the same dataset

## Status

- [x] Model size investigation
- [x] Voxtral inference server (vLLM-based, see `modal-serving/`)
- [x] Fine-tuning with LoRA via transformers + PEFT (see `modal-training/`)
- [ ] Running on Mac? Review gguf / llama.cpp options
- [ ] Update timestamping notes below

Awaiting:
- [ ] Unsloth support: Pending [here](https://github.com/unslothai/unsloth/issues/3013)

Miscellaneous notes on Voxtral:
- Voxtral comes in a 3B and a 24B variant. Both outperform whisper, although both will be slower than running Whisper Turbo.
- Timestamping:
    - Voxtral today: segment timestamps only; word-level on the roadmap for the Mistral API, unclear if that will be release open source.
    - OpenAI gpt-4o-mini and Google's Gemini Flash ... See [here](https://chatgpt.com/g/g-p-6879fa0579948191bfc397c0d66524fa-videos/c/6879f22c-3370-8003-b6bb-1be4fa8040cc).
    - Whisper: already outputs word-level times via timestamp tokens → attention alignment (or external forced-alignment for even finer granularity):
        - Whisper natively includes segment timestamps. Voxtral does not (it comes from an LLM).
        - Whisper further uses attention maps to get word level timestamps (~20 ms accuracy).
        - WhisperX then uses a Connectionist Temporal Classification (CTC) model to even more accurately predict timestamps (~10 ms accuracy).
        - FWIW whisper-timestamped is a slightly better word timestamping version than whisper.
- As of Jul 21 2025, Voxtral can be fine-tuned with transformers but support is not there yet for unsloth - although it's coming soon.
- Voxtral is an encoder-decoder model so it's harder to port to gguf and llama.cpp, and it's different to whisper so hard to port to whisper.cpp .

## WER Results (Trelis/llm-lingo, 6 val samples)

| Model | Baseline WER | Fine-tuned WER | Improvement |
|---|---|---|---|
| Voxtral-Mini-3B | 30.6% | 14.6% | 16.0pp |
| Whisper-large-v3-turbo | 37.0% | 15.1% | 21.9pp |

Config: LoRA rank=32, alpha=32, RSLoRA, 3 epochs (Voxtral) / 2 epochs (Whisper).

## Transcription versus Chat Performance

To understand the differences between Voxtral's transcription and chat approaches, run the demo script:

```bash
uv venv
uv run chat-v-transcribe.py
```

This script demonstrates:
- **Template structures**: How each approach formats inputs differently
- **Token organization**: ACTUAL tokenized patterns from Voxtral processor
- **Transition points**: Context around key template boundaries (audio end, [/INST], etc.)