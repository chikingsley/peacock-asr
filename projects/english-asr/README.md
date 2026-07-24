# English ASR

This project targets a fast, distributable English Parakeet TDT+CTC 110M model. It is the first high-resource project using the canonical path from a typed source registry through `omni-curator`, immutable direct NeMo manifests, TDT fine-tuning, and held-out evaluation.

## Locked decisions

- The base is the exact official `nvidia/parakeet-tdt_ctc-110m` artifact pinned in `src/english_asr/parakeet/artifacts.json`.
- English fine-tuning preserves the base model's embedded 1024-piece case-and-punctuation tokenizer. It does not call NeMo `change_vocabulary()` and therefore does not reset pretrained decoder/joint vocabulary weights.
- Only upstream **train** material is ingested. A deterministic group-disjoint 5% internal dev split is carved from train; upstream validation/test sets remain untouched.
- There is no global Scribe WER cutoff. Human/reference labels stay authoritative. Scribe disagreement, edge error, and CTC alignment are additive audit columns until a reviewed, source-specific filter passes a matched two-seed training ablation.
- Hard exclusions are missing/unreadable audio, empty or descriptor-only text, confirmed wrong language, duplicates/leakage, nonpositive duration, and the 30-second model duration cap.
- Earnings21, Earnings22, Open ASR leaderboard test packages, and other reporting exams are never training data.

## Data waves

The exhaustive executable inventory is `src/english_asr/sources.py`; it includes legal status, label authority, base-replay overlap, and adapter state. The active first wave is intentionally bounded:

| Source                   | Purpose                   |      Bound | License          | State                                             |
| ------------------------ | ------------------------- | ---------: | ---------------- | ------------------------------------------------- |
| People's Speech microset | End-to-end pipeline proof |     ~1.3 h | CC-BY / CC-BY-SA | Ready                                             |
| GigaSpeech XS train      | Novel mixed-domain speech |       10 h | Apache-2.0       | 2K gate rejected                                  |
| AMI IHM train            | Novel meeting speech      |       25 h | CC-BY-4.0        | 2K gate rejected                                  |
| Common Voice 26 train    | Broad read-speech accents | full train | CC0-1.0          | 65.7 h gate won at 20K as a source-domain model   |
| People's Speech clean    | Mixed public speech       |       10 h | CC-BY / CC-BY-SA | Frozen 10 h source gate and replay material ready |

The source-isolated gates are measured in `EXPERIMENTS.md`: neither GigaSpeech XS nor AMI beat the unchanged base. Common Voice 26 at the original `3e-4` learning rate tied but did not beat the base; the same frozen 65.7-hour gate at `1e-4` advanced to 20K and produced a statistically supported CV26-domain win, reducing normalized WER from 7.0167% to 6.6356% at 700.7 warm-cache RTFx. It is not the general-English promotion: GigaSpeech, AMI, and People's Speech regressed. A fresh four-source replay run and a low-rate sequential recovery run both failed, establishing that unconstrained concatenation or replay does not solve the label-style/domain conflict. The replacement experiment uses word-preserving punctuation-and-capitalization restoration, NeMo's weighted `input_cfg` multiplexer at 25% per source, and L2-SP anchoring to the official base. Its 20K seed-0 and seed-1 runs established two reproducible local frontiers: a 50% candidate / 50% base interpolation for balanced retention and an 80% candidate / 20% base interpolation for aggregate accuracy. A later 10%-weighted LibriSpeech `train-clean-100` replay arm completed its 2K gate at 8.8700% four-domain macro WER, behind the original four-source 2K candidate's 8.7827%, and is rejected without a 20K run. Large weak/pseudo pools such as Granary and YODAS remain later-stage inputs because source licensing, duplication, and benchmark contamination must be preserved and audited.

The 110M base already reports training on LibriSpeech 960, Fisher, NSC Part 1, VCTK, VoxPopuli English, Europarl-ASR English, roughly 2,000 hours of MLS English, and Common Voice v7. Those are replay anchors, not novel-data claims.

## Dictation lane

The product lane now targets a separate `english-dictation-110m` model rather than requiring one 110M checkpoint to dominate meetings, distant microphones, dialect studies, read speech, and near-field dictation simultaneously. The general-English 50% and 80% frontiers remain frozen comparison artifacts; regressions on general suites are reported for a dictation candidate but do not automatically veto it.

`AutoArk-AI/ARK-ASR-3B` at revision `1e28271b79edc97635783bea65abc89195a09ed3` is the locked pseudo-label teacher. The local benchmark confirmed its broad-English accuracy and throughput, and the owned-recording spot check accepted its cleanup and punctuation behavior without recording any issue markers. Preserve the raw teacher generation, cleaned transcript, exact model revision, inference configuration, source/session identifier, segment boundaries, and label timestamp. Teacher output remains pseudo-label supervision except for the explicitly user-approved frozen product-output gold set.

The first owned-recording review is 200 segments sampled across at least 20 sessions, dates, durations, and recording conditions, totaling approximately 30–60 minutes rather than hundreds of hours. ARK transcribes every segment first. The reference surface is the ideal text the product should paste, not a verbatim acoustic transcript: omitting harmless fillers, false starts, repeated syllables, and stutters is acceptable when meaning is preserved. A keyboard-driven review UI records accepted output, content-word, formatting, bad-boundary, privacy, and uncertainty decisions. Content and formatting decisions require a corrected full output; clicking a word pauses and preselects it in the editor, while Shift-click extends the range. The timestamped marker localizes the model failure and the post-edit supplies the product-output gold reference. Split by original session before labeling or training so adjacent segments cannot cross train, development, and test.

The live import contains 1,688 usable sessions / 51.8583 hours. Frozen review v1 reserves 60 whole sessions and materializes 200 duration-prioritized Silero `conservative-v1` clips across all 60 sessions. Those clips total 22.5347 minutes with a 6.108-second median; only three are shorter than one second. The reviewer manually spot-checked 23 rows, found no issue markers, and explicitly accepted the remaining 177 rows as the same product-output surface. The immutable `gold-v1` export records that distinction instead of claiming full verbatim human transcription. The identities of all 60 held-out sessions remain outside training even though only 200 of their 1,974 candidate regions are in the scored target.

The bounded execution sequence is:

1. Lock the benchmarked ARK-3B revision and inference contract. **Complete.**
1. Inventory owned recordings without copying or modifying them, then create a session ledger containing duration, date, device/source, and privacy/exclusion state. **Complete.**
1. Freeze the 200-segment review sample. Keep it outside training permanently; use the accepted or corrected product-output references for WER and the full reviewed set for acceptance, error-category, boundary, and exclusion rates. **Complete.**
1. Benchmark the official 110M base on the frozen target and retain ARK-3B, Parakeet TDT v3, and the English-only Parakeet TDT v2 as teacher/deployment controls. **Complete.**
1. Build the first dictation training derivative from the remaining owned recordings, ARK pseudo-labels, the shared VAD/postprocessor contract, and session-disjoint splits. Start with at most 25 labeled hours and run the standard 2K gate. **Complete.**
1. Promote to two independent 20K runs only if dictation WER improves without punctuation collapse or boundary-loss growth, then evaluate the 50/50 seed average. General suites remain regression reports. **Complete.**
1. Expand next to approximately 250 matching hours, then to 2,000 or more hours only after the owned-domain pilot establishes a repeatable gain. Common Voice Spontaneous may be a low-weight auxiliary source; scripted, meeting, and broadcast corpora are not automatically mixed into the dictation lane.

The campaign winner is the 50/50 parameter average of the two independently trained 20K best-`val_loss` checkpoints. It scores 5.3911% normalized WER / 2.7814% CER with zero empty outputs on `gold-v1`, down from the unchanged official base's 10.8200% / 6.4941%. Both individual seeds independently reached 5.5419% WER. The promoted model is `runs/parakeet/english-dictation-owned-ark-25h-v1-20k-seed-average/english-dictation-owned-ark-25h-v1-20k-seed-average.nemo` with SHA-256 `71d36405b8c0b86fe8f722d49199e462abd4a1aa9e00f70566a05871fb540547`. Boundary agreement improved, punctuation density remained aligned with the reviewed product surface, and the warm fp32 pass reached 771.84x realtime at 2.25 GiB peak CUDA allocation. `EXPERIMENTS.md` records the complete manifests, seed results, hashes, and audit.

## Run the bounded sequence

The project data symlink points at `/mnt/tiny-2t/peacock-asr/english-asr/data`.

```bash
# 1. Prove ingest on 336 People's Speech rows.
uv run --project projects/english-asr english-curate ingest peoples-speech-microset

# 2. Export normalized, tokenizer-covered direct NeMo manifests. No audio copy and no WER gate.
uv run --project projects/english-asr english-curate export source-peoples-speech-microset-manifest-v0 --format nemo-manifest --source peoples-speech-microset --no-mixture-weights --max-duration 30

# 3. Run a short plumbing gate before a real ablation.
uv run --project projects/english-asr english-parakeet-train-tdt --prepare-only
uv run --project projects/english-asr english-parakeet-train-tdt --max-steps 200 --val-every 100 --warmup 20 --name english-smoke-200
```

For a source-isolated gate, train the direct source manifest for 2,000 steps with seed 0, select by `val_loss`, and compare the exported checkpoint in fp32 against the unchanged official 110M base on the exact same dev manifest. Only a source that beats the base advances to a longer run. Before general promotion, evaluate every candidate on the CV26, GigaSpeech, AMI, and People's Speech internal dev sets separately; a source win with material regressions elsewhere remains a domain checkpoint rather than the default English model.

## Canonical artifacts

- Working rows: `data/curator.sqlite`
- Immutable derivative exports: `data/datasets/<name>/` (direct manifests) or `data/datasets/<name>/version=0` (portable Parquet when explicitly requested)
- Hot training cache on this host: `/mnt/workerssd-2t/peacock-asr/english-asr/training-cache/`
- Training runs: `runs/parakeet/<run>`
- Promoted final model: `data/parakeet/final/`
- Reproducibility pointer: `src/english_asr/parakeet/artifacts.json`

Source references: [NVIDIA 110M model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m), [GigaSpeech](https://github.com/SpeechColab/GigaSpeech), [AMI and ICSI](https://groups.inf.ed.ac.uk/ami/), [People's Speech](https://mlcommons.org/datasets/peoples-speech/), and [Common Voice](https://commonvoice.mozilla.org/en/datasets).
