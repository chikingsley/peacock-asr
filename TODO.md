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

- [ ] Replace the retired ad hoc HF uploader with one maintained release workflow: explicit staging, append-only state, one upload method, remote sibling/checksum verification, and deletion only after proof.
- [ ] Export final YouTube datasets, publish/verify them on HF, then remove local release-safe clips and caches. Segmenting alone does not relieve disk pressure.
- [ ] Verify exact current HF repository names before deleting superseded Persian/Farsi model repos or the large local Farsi/Tajik exports.
- [ ] Publish or move Russian canonical audio off WorkersSSD before treating it as general scratch.
- [ ] Make `archive_needed()` cheaper than scanning every segmented source path on every factory tick.

## Documentation

- [ ] Rewrite `packages/omni-curator/README.md` around the queue-driven create pipeline; remove the deleted `vad_path` / chunks / stitching / polishing architecture.
- [ ] Update `CURATION_FACTORY.md`, `NEW_LANGUAGE.md`, the project standard, and the data-state handoff after the measured Farsi multi-VAD pilot. Archive the pre-resegmentation handoff only after its decisions are represented in current docs.
- [ ] Review remaining historical per-language docs and delete exact duplicates or archive stale runbooks without treating experiment records as current commands.

## Blocked on external input

- [ ] Native-speaker conversational ground truth remains the only true accuracy measure beyond the Scribe-label ceiling. It becomes actionable when Tajik/Persian reviewer access is available.

## Reference

- Current curation/factory plan: `packages/omni-curator/docs/CURATION_FACTORY.md`.
- Current pre-resegmentation handoff: `docs/data-state-audit-2026-06-28.md`.
