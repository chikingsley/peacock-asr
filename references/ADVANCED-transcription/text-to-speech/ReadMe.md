# Text to Speech

Folder contents:
- `kokoro` [BEST FOR SYNTHETIC DATA]: ~80M parameter TTS model allowing for local inference, and inference via a server. Allows multiple voices and accents. No fine-tuning support.
- `unsloth` [BEST FOR FINE-TUNING]: Scripts for fine-tuning and inferencing Orpheus and CSM-1B models, and any TTS supported by Unsloth.
    - Orpheus is best for fine-tuning and server/vllm deployment.
    - CSM is best for voice-cloning (one shot).
- `StyleTTS2` [GOOD FOR FINE-TUNING]: Scripts for fine-tuning StyleTTS2.
- `MeloTTS`: Scripts for inferencing MeloTTS with a fast UDP connection.
- `Trelis-orpheus` [DEPRECATED, see `unsloth`]: Scripts for inferencing, cloning AND fine-tuning Orpheus.
- `Trelis-csm` [DEPRECATED, see `unsloth`]: Scripts for inferencing and cloning CSM-1B.