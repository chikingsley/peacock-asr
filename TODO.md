# TODO — peacock-asr

Active work only. Completed work belongs in `CHANGELOG.md`; live pipeline state comes from `packages/omni-curator/status.py`.

## Multi-VAD curation

- [ ] Add a durable production segmentation-run record (clip metadata already carries the exact engine/model/backend/postprocessor policy and effective hash).
- [ ] Extend explicit per-project routing with independently measured clean/read and noisy/conversational profiles for the remaining languages. Treat Dari -> Cobra as provisional until a Dari-specific pilot.
- [ ] Carry the promoted Farsi Silero policy through a bounded `labelq -> harvest -> merge -> verify` integration run before scale.
- [ ] Make direct CLI and factory worker/device defaults engine-aware, load project secrets before Cobra preflight, and rerun live parent-death/orphan-GPU cleanup tests.
- [ ] Add a repository-level third-party notice before redistributing VAD adapter code or the NVIDIA MarbleNet v2 weights; runtime dependencies have upstream licenses, but this repo has no consolidated release notice yet.

## Language curation

- [ ] Before production resegmentation, decide per language/channel whether archive-backed source gaps are restored to Tiny2T in bounded batches or read from Massive during pilots.
- [ ] Farsi broad-registry downloads: only `iran_international` and `avas_book_club` currently have local FLACs; download/enqueue the remaining 122 registered channels after the bounded VAD pilot.
- [ ] Tajik v4: finish download/enqueue, run the selected segmentation policy, label/harvest/merge/ verify/export, then train and evaluate on the frozen conversational and FLEURS tests. Deduplicate `tv_tajikistan` / `tvt_tojikiston`.
- [ ] Dari v0: refresh cookies, finish wired channels, stage Pimsleur Dari, run the curation chain, then compare cold-base versus Farsi-warm Omni training. Consider a Pashto gate for bilingual channels.
- [ ] Georgian v1: download/enqueue the expanded registry, run the curation chain, export, train, and rerun KenLM-fused evaluation.

## Data-quality audit

- [ ] Inspect the owned Trelis `ADVANCED-transcription` implementation on the Mac before copying thresholds or confidence formulas for edge-error and CTC alignment filtering.
- [ ] Review the completed 160-row Persian quality pilot across ASR-edge, WER, alignment-span, and margin combinations; then expand to 1,000 professional/human-transcribed plus 1,000 Scribe-v4 pseudo-labeled rows if the review confirms useful precision.
- [ ] Add optional VAD speech outside the aligned CTC span. Keep every signal audit-only until a human-reviewed threshold reaches at least 95% reject precision.
- [ ] Recover or rebuild an all-candidate Persian ledger if rejected-row experiments matter. The published v4 and WER35 Hub datasets contain kept training rows, while the old WER35 artifact omits its 92,462 rejected candidates.
- [ ] Run a size-matched training ablation: random unfiltered `N` versus cleaned `N`, identical recipe/steps, evaluated on untouched gold and conversational sets. Never confidence-filter dev or test data.

## Training and benchmarks

- [ ] English dictation 110M: finish the local `AutoArk-AI/ARK-ASR-3B` teacher benchmark, inventory owned recordings read-only, freeze the session-disjoint 200-segment review set, and run the bounded 25-hour 2K pilot described in `projects/english-asr/README.md`.
- [ ] Analyze the Tajik 110M TDT versus Omni v3 conversational predictions by source, duration, and error type. TDT wins WER (33.85 versus 37.65) but loses CER (14.89 versus 14.04); understand that tradeoff before any new Tajik TDT training.
- [ ] Harden `packages/asr-benchmark-core` after the current language work: make language scorers and Omni model cards pluggable, add suite/config registration, model/data checksums and run completion state, then standardize the cross-model contract around fixed clips and segmentation revision, raw and normalized WER/CER, empty-output rate, cold end-to-end latency, warm batched RTFx, safe maximum batch, peak VRAM, load time, timestamp support, exact model revision, and saved predictions.
- [ ] Persian benchmark: current Omni, newly accessible C1Tech Whisper Persian, Qwen3-ASR 0.6B/1.7B, Whisper large-v3-turbo, and Scribe v2 on the same canonical splits.
- [ ] Russian benchmark: GigaAM v3 CTC/RNNT, Parakeet v3, Qwen3-ASR, Whisper turbo, and Scribe on the same clips.
- [ ] Run the `omni-bench-llm` ceiling test on Tajik, Georgian, and Farsi.
- [ ] Farsi training provenance: restore or replace stale dataset/checkpoint/card paths before the next fine-tune. Keep the existing production model evaluation runnable in the meantime.

## Data lifecycle and storage

**Archive pass 2026-07-24 — project paused for token/compute budget.** Artifacts that had ZERO Hugging Face coverage were rescued first; see "Backup state" below before deleting anything local.

- [ ] Slow Scribe v2 relabel of the Farsi raw corpus for the new voice-lab requirements. The runner already exists: `/home/simon/github/language-learning/pimsleur-hub/transcript-generation-15-second-cooldown/` (Go). It submits one job at a time through the Peacockery Voice Lab batch API using `elevenlabs-scribe_v2`, holds an exclusive local file lock, and persists `next_eligible_at` so the 15-second cooldown survives across invocations; diarization and auto language detection on. Needs `PEACOCKERY_VOICE_API_KEY` or `PEACOCKERY_VOICE_LAB_API_KEY`. Source audio is the `iran_international` FLAC corpus, uploaded as individual `.flac` files (not tarred) specifically so clips can be relabeled incrementally without pulling the whole 257 GB. Partial relabeling is acceptable — the cooldown makes this inherently slow.
- [ ] Replace the retired ad hoc HF uploader with one maintained release workflow: explicit staging, append-only state, one upload method, remote sibling/checksum verification, and deletion only after proof.
- [ ] Export final YouTube datasets, publish/verify them on HF, then remove local release-safe clips and caches. Segmenting alone does not relieve disk pressure.
- [ ] Verify exact current HF repository names before deleting superseded Persian/Farsi model repos or the large local Farsi/Tajik exports.
- [ ] Publish or move Russian canonical audio off WorkersSSD before treating it as general scratch.
- [ ] Make `archive_needed()` cheaper than scanning every segmented source path on every factory tick.

### Backup state as of 2026-07-24

Verify any claim here against `HfApi().repo_info(repo, files_metadata=True)` — an earlier agent reported an upload complete that had never happened. Use `HF_HUB_DISABLE_XET=1` for large uploads (the Xet backend stalls silently on big files), and tar directories holding tens of thousands of small files (HF caps ~320 repo commits/hour).

Rescued because they had NO Hugging Face copy at all:

- `Peacockery/peacock-asr-owned-dictation-backup-2026-07-24` — **PRIVATE by design.** 945 MB: the 1,711 personal dictation recordings, the 200-row human `review.sqlite` (plus both pre-edit snapshots), the frozen `gold-v1` manifest, and session ledgers from `english-asr/data/owned-dictation/mac-import-20260716/`. Archive entry count verified 2992/2992. This is the only non-pseudo-label human supervision in the project. Keep it private — it is the user's own voice and dictation content, unlike the model/dataset repos.
- `Peacockery/english-asr-runs-backup-2026-07-24` — public. The 40-run english-asr experiment record (metrics, WER/CER, alpha sweeps, predictions), MOSS eval measurements, and data provenance ledgers; plus the 92 final `.nemo` exports (42 GB).
- `Peacockery/farsi-asr-iran-international-raw`: public and retrieval-verified on 2026-08-01 at commit `942b4dce8d58a6400db6b099d93ae16bd262b4ca`. The repository contains all 11,993 raw FLACs (`275,207,523,153` bytes): 9,990 at the root and 2,003 under `overflow/` to satisfy Hugging Face's 10,000-entry-per-directory limit. `REMOTE_PATHS.jsonl` records every filename, remote path, byte size, and SHA-256; its SHA-256 is `85dd32115c54510f52f42e3f39656b10db8da94ff8211e12776742d45d925e46`. A fresh download of `overflow/-1Xetdq5E_k.flac` matched the local file byte-for-byte at SHA-256 `e3f32179db7375d7a3233c0d92f73bc01f371c33dcdc8cc0dc5776019039dd3a`. The original `huggingface_hub 1.21.0` large-folder worker stalled after 9,990 commits because its queue-size branch retried Hugging Face's 320-commit/hour response immediately; the completion pass re-preuploaded the remaining 2,003 files in resumable batches and committed them once under the sharded layout. This raw source is separate from `farsi-asr-corpus-v4`, whose seven corpora are `thomcles_persian_farsi_speech`, `common_voice_25_0`, `asr_farsi_youtube`, `mana_tts`, `neyshekar`, `worldspeech`, and `fleurs`.

`Peacockery/MOSS-Transcribe-preview-2B-MLX` was completed and retrieval-verified on 2026-08-02 at commit `f453d019e9efb64f7ed275d67b55999a5d2985c7`. All 62 local artifact paths were present remotely; the three `weights.safetensors` files matched the Hub's stored SHA-256 values, and a fresh download of the 1.37 GB all-model 4-bit weight matched locally at SHA-256 `be0e6c838b10e34ae3b7341cee507d9b41cef8f77fcb1569dd377b49decce22e`.

- `Peacockery/english-asr-checkpoints-backup-2026-08-02`: public and fully verified on 2026-08-03 at commit `61bcd1102776f05162564c1d1b250046f01fc0cf`. All 621 expected run files (`214,876,783,010` bytes) matched the remote path and byte-size inventory; all 156 Lightning checkpoint SHA-256 values matched; and a fresh 1.38 GB checkpoint retrieval matched locally.
- `Peacockery/farsi-asr-runs-backup-2026-08-02`: public and fully verified on 2026-08-03 at commit `b5d057fb14bb58e3e12da508605e0e7668180858`. All 747 expected durable run and experiment files (`313,339,019,626` bytes) matched the remote path and byte-size inventory; all 190 Lightning checkpoint and 63 NeMo SHA-256 values matched; and a fresh NeMo retrieval matched locally. Regenerable `experiments/lm_decoding/logits/` files were intentionally excluded.

Corrected belief: there is no paid ElevenLabs/Scribe labeled data in this repo today. In `english-asr/data/curator.sqlite` the `scribe_wer` / `scribe_cer` / `scribe_status` columns are 100% NULL across all 1,337,984 rows; existing pseudo-labels come from local `AutoArk-AI/ARK-ASR-3B` inference. Paid Scribe labeling is planned work (see the relabel item above), not sunk cost.

Cleanup completed on 2026-08-02 after the remote checks above:

- Removed the re-downloadable `base_models/` content while retaining `parakeet/ctc.nemo` (418 MB) because its provenance remains unreconciled.
- Removed `moss-mlx-conversion/bundles/`, duplicated `coreml/build/`, and `artifacts/mlx/` after their respective CoreML and MLX Hub verification. Also removed the 43 GB retained `artifacts/coreml/` workbench after the user retired the unsuccessful MOSS effort.
- Removed the published English `data/datasets/` and re-downloadable `data/benchmarks/`, plus the 92 final `.nemo` exports covered by `Peacockery/english-asr-runs-backup-2026-07-24`.
- Removed the retrieval-verified 11,993-file Iran International source cache from Tiny2T.
- Removed the heavy English `training-cache/{hf-cache,hf,exams,canonical_audio}` trees from WorkersSSD while retaining 536 MB of manifests, logs, evaluations, and tools.

Final checkpoint cleanup completed on 2026-08-03 after every remote gate passed:

- Removed `english-asr/runs/`, `farsi-asr/runs/`, and the regenerable Farsi `experiments/lm_decoding/logits/` cache, plus their completed Hugging Face upload metadata. Peacock ASR fell from 599 GB to 14 GB; NVMe free space rose from 282 GB immediately before cleanup to 858 GB afterward.

## Documentation

- [ ] Rewrite `packages/omni-curator/README.md` around the queue-driven create pipeline; remove the deleted `vad_path` / chunks / stitching / polishing architecture.
- [ ] Update `CURATION_FACTORY.md`, `NEW_LANGUAGE.md`, the project standard, and the data-state handoff after the measured Farsi multi-VAD pilot. Archive the pre-resegmentation handoff only after its decisions are represented in current docs.
- [ ] Review remaining historical per-language docs and delete exact duplicates or archive stale runbooks without treating experiment records as current commands.

## Blocked on external input

- [ ] Native-speaker conversational ground truth remains the only true accuracy measure beyond the Scribe-label ceiling. It becomes actionable when Tajik/Persian reviewer access is available.

## Reference

- Current curation/factory plan: `packages/omni-curator/docs/CURATION_FACTORY.md`.
- Current pre-resegmentation handoff: `docs/data-state-audit-2026-06-28.md`.
