# Persian ASR Agent Notes

## Command Execution

- Use `uv run <project-console-script> ...` for repo commands.
- Use the project scripts defined in `pyproject.toml`, such as `persian-train-omni`, `persian-benchmark-suite`, and `persian-scribe-*`.
- Do not call `.venv/bin/python`, `.venv/bin/<tool>`, or other direct `.venv` paths.
- Do not use `uv run python` for project workflows; add or use a console script instead.
- Shared ASR tools live as peer projects under `peacock-asr/projects`; use uv-visible console scripts from the project environment rather than import-path hacks.

Official uv references:

- Project command execution: <https://docs.astral.sh/uv/concepts/projects/run/>
- CLI command behavior: <https://docs.astral.sh/uv/reference/cli/>
- Tools and `uvx`: <https://docs.astral.sh/uv/concepts/tools/>
- Tool usage guidance: <https://docs.astral.sh/uv/guides/tools/>
- Scripts and inline metadata: <https://docs.astral.sh/uv/guides/scripts/>
- Workspaces and `--package`: <https://docs.astral.sh/uv/concepts/projects/workspaces/>

## Scribe / ElevenLabs

- For Scribe work, use the shared `../superwhisper-api` project through the configured uv source dependency, for example `uv run superwhisper-audio ...`.
- Preserve raw ElevenLabs STT responses when endpoint confidence, word timings, logprobs, language probability, or entity metadata are part of the analysis.

Official ElevenLabs references:

- Speech-to-text API: <https://elevenlabs.io/docs/api-reference/speech-to-text/convert>
- Speech-to-text overview: <https://elevenlabs.io/docs/capabilities/speech-to-text>

## Required Local Skill

- For Persian ASR dataset, Scribe, benchmark, training, or GPU recovery work, use the local `persian-asr-operations` skill.

## Response Style Guard

- Avoid contrastive negation patterns in final answers. Replace negated contrast with the direct state or consequence.
- Treat literal negation tokens as stop-hook hazards in final answers. Use direct wording instead.
- Avoid filler adverbs such as "actually", "really", "simply", "basically", "essentially", and "quietly".
- Avoid stock capstone endings such as "That is the point" or "This is the answer".

## Normalization

- Use only `persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large.maybe_normalize` for Persian ASR dataset normalization.
- The source is the NVIDIA Persian FastConformer model-card normalizer pinned by repo revision and README SHA.
- Do not create or use ad hoc Persian text normalization code.

## Scribe Full-Run Gate

- Before any full Scribe audit/classification run, run a small normalized sample gate and show raw reference, normalized reference, raw Scribe, normalized Scribe, WER, CER, and category.
- Full runs require verified sample output and verified category schema.
- The audit prompt classifies transcript differences only; training-label policy belongs in export logic.

## Artifact Policy

- SQLite is the working audit/dataset store.
- JSONL is staging, transfer, or archival format.
- Do not create new `results/`, `labels/`, duplicate full-dataset folders, or overlay files without explicit user instruction.
- Preserve raw Scribe API responses when endpoint metadata matters.

## GPU Recovery

- Check `nvidia-smi`, train tmux/process state, latest checkpoint, and last metric before restart.
- If NVIDIA driver/module state is wedged, reboot host, then resume from the latest checkpoint.
