# ADVANCED-transcription
Fine-tune transcription (e.g. Whisper) and text to speech models.

> To stay abreast of product updates, you can get on the Trelis Newsletter [here](https://trelis.substack.com).

Repo Contents:
- `audio-alignment`: Forced alignment tutorial (TorchAudio MMS_FA)
- `audio-llm`: Multi-modal Audio + Text LLMs (Qwen Audio)
- `speech-to-text`: Speech to Text
- `text-to-speech`: Text to Speech
- `speech-to-speech`: Speech to Speech
- `voice-detection`: Voice Detection, Turn Detection and Diarisation

To quickly get started with GPU fine-tuning, you may wish to make use of this [one-click-template [affiliate]](https://console.runpod.io/deploy?template=kfrosdmse5&ref=jmfkcdio) - do review the README.md as you'll need to set up a Github Personal Access token.

## Changelog:
7jan2025:
- Add audio-alignment tutorial with TorchAudio MMS_FA forced alignment
- Add modal whisper server support
- Add word confidence comparison of transcription APIs

3jan2025:
- Support kyutai stt fine-tuning

20Nov2025:
- Release Kokoro voice inference and server (text to speech) - good for synthetic data

19Nov2025:
- Release Whisper data preparation and fine-tuning with Unsloth

20Aug2025:
- Release Kyutai speech to text inference scripts.

17Jun2025:
- Release Orpheus TTS fine-tuning notebook with unsloth integration.
- Release Sesame CSM-1B TTS fine-tuning notebook with unsloth integration.
- Release vLLM server implementation for Orpheus TTS models.

03Apr2025:
- Release Trelis Qwen2.5-omni inference scripts + links to fine-tuning scripts (not tested)
- Moshi inference scripts for Mac M2/3/4 and GPU.

22Mar2025:
- Release Orpheus voice cloning and fine-tuning.

19Mar2025:
- Release Sesame CSM-1B voice cloning and context-aware scripts

14Mar2025:
- Release voice detection folder.
25Nov2024:
- Add audio+text LLM fine-tuning, Qwen Audio.

16Oct2024:
- Release updated Whisper Fine-tuning Notebook and Inference Options.

26Aug2024:
- Release speech to speech folder.

18Jul2024:
- Release text to speech folder.

01Jul2024:
- Reorganise repo into text to speech (forthcoming, not available just yet) and speech to text folders
