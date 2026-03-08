# P008 - CAPT Runtime

This workspace turns the existing pronunciation-scoring research into a runtime
stack for known-text pronunciation training.

Primary scope:

- `canonicalizer`: dictionary-first text-to-pronunciation with G2P fallback
- `aligner`: transcript and post-utterance timestamps for UI
- `scorer`: wrap `P001` and later `P002` scoring components
- `feedback`: assemble word- and phone-level issues into product payloads

Non-goals:

- training a new acoustic backbone
- replacing `P001` as the core scorer
- benchmarking every ASR/TTS model on day one

Start here:

- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `docs/CANONICALIZER.md`
- `docs/G2P_BAKEOFF.md`
