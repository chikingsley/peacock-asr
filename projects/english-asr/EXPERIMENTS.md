# English ASR experiments

## Baseline snapshot — 2026-07-14

The initial target is the official English Parakeet TDT+CTC 110M model with its embedded tokenizer and untouched weights. The current Open ASR Leaderboard English short-form snapshot reports approximately 7.323 mean original WER, 6.607 mean cleaned WER, and 6,119 RTFx for the 110M model. Scribe v2's approximately 4.647 cleaned WER is the accuracy-teacher reference, not a deployable weight source.

No fine-tune has been promoted as the default general-English model. The CV26 20K checkpoint is retained as a verified source-domain winner. A run earns an entry here only after its exact export, seed, base SHA, best `val_loss` checkpoint, fp32 held-out WER, and warm-cache RTFx are measured.

## Pipeline smoke — 2026-07-14

- **Source:** People's Speech `microset` at revision `f10597c5d3d3a63f8b6827701297c3afdf178272`.
- **Curator result:** 336 rows / 1.3162 hours; deterministic source-group split produced 323 train rows / 1.2631 hours and 13 dev rows / 0.0531 hours.
- **Export result:** 336 rows, no Scribe-WER gate, 30-second duration cap, zero embedded-tokenizer `<unk>` rows.
- **Materialization result:** all 336 rows became deterministic FLAC + NeMo JSONL records; zero empty or duration exclusions. Manifest hashes are pinned in `src/english_asr/parakeet/artifacts.json`.
- **Live model preflight:** official SHA-pinned `.nemo` restored on CPU as `EncDecHybridRNNTCTCBPEModel`, 114,624,647 parameters, embedded vocabulary 1024, `compute_eval_loss=True`, auxiliary CTC weight 0.3. No vocabulary replacement occurred.
- **GPU smoke result:** passed after the Cohere vLLM service was stopped. A two-step live proof trained and validated (`val_loss=16.3571`); the 200-step run trained and exported successfully; and a 50-step cross-epoch proof validated at step 50 (`val_loss=6.9419`, noisy BF16 `val_wer=0.0651`) and exported the best-validation `.nemo`.
- **Smoke fixes promoted to the shared path:** English pins Numba 0.65.1 because Numba 0.66.0 broke NeMo's TDT gradient kernel signature, and the shared trainer now sets `check_val_every_n_epoch=None` so integer validation intervals apply globally across short iterable-data epochs. The smoke therefore proved CUDA forward/backward, validation, best-checkpoint selection, checkpoint restore, and `.nemo` export; it was not an accuracy experiment.

## Source-isolated 2K gates — 2026-07-14

The source-isolated gate starts from the unchanged official 110M base, trains one source for 2,000 steps with seed 0, selects the lowest-`val_loss` checkpoint, and compares it in fp32 against the base on that source's untouched internal dev split. A source that does not beat the base stops here: no 20K extension, second seed, or combination run.

| Source / recipe      | Dev rows | Base WER | Candidate WER | Base CER | Candidate CER | Decision                                                                   |
| -------------------- | -------: | -------: | ------------: | -------: | ------------: | -------------------------------------------------------------------------- |
| GigaSpeech XS        |      276 |  2.7336% |      11.3831% |  1.1003% |       4.6261% | Reject; the already-strong base was over-specialized                       |
| AMI IHM 25 h         |    1,876 | 20.3805% |      26.3124% | 12.1454% |      16.6613% | Reject; transcript-style adaptation did not improve normalized recognition |
| CV26 65.7 h, LR 3e-4 |    1,777 |  7.0167% |       7.0469% |  2.5776% |       2.5776% | Reject; controlled tie, but no WER win                                     |
| CV26 65.7 h, LR 1e-4 |    1,777 |  7.0167% |       6.9441% |  2.5776% |       2.5379% | Promote to a fresh 20K run                                                 |

GigaSpeech's best checkpoint was step 1,750 (`val_loss=7.2042`); the candidate remained fast at 566.5 RTFx, so the rejection is accuracy-only. AMI's best checkpoint was the final step (`val_loss=6.5288`); it reduced empty hypotheses from 169 to 47 and made raw uppercase transcript matching look better, but normalized WER still regressed by 5.93 absolute points. The standard CV26 arm selected step 2,000 (`val_loss=2.0473`) and scored 7.0469% normalized WER at 847.8 RTFx. The one-knob `1e-4` CV26 arm also selected step 2,000 (`val_loss=1.9427`) and scored 6.9441% normalized WER, 2.5379% CER, zero empty hypotheses, and 804.6 RTFx. This is a 0.0726-point absolute / 1.03% relative WER improvement over the unchanged base without a throughput regression.

The data path used for these gates no longer embeds and re-extracts audio through Parquet. Omni Curator emits immutable direct NeMo manifests that reference canonical FLACs. The source manifests and approximately 1.9 GB of current canonical audio were staged under `/mnt/workerssd-2t/peacock-asr/english-asr/training-cache/` because the project data symlink is on a USB HDD; this restored normal GPU utilization while CV26 continued downloading on the HDD.

## Common Voice 26 preparation — 2026-07-14

The English MDC archive is 94,639,372,950 bytes. The downloader now preserves incomplete `.part` files, validates the server-declared total, calls `fsync`, atomically promotes only an exact-size archive, and writes a full SHA-256 sidecar. Omni Curator's CV26 adapter consumes the verified archive in one forward pass, reads upstream `train.tsv` only, excludes the frozen CV9 benchmark by clip ID or encoded-audio SHA-256, converts selected clips directly to resumable 16 kHz mono FLAC, and carves the internal dev set by stable client identity. It does not extract the 95 GB tarball or create six embedded-audio Parquet copies.

The first CV26 snapshot was invalid and was discarded before it could count as a training result: Python's CSV quote handling merged multiple physical TSV records whenever a prompt contained an unmatched literal double quote, producing 30,000-character labels and RNNT joint-memory failures. Common Voice TSV values are not CSV-quoted. The adapter now reads them with `csv.QUOTE_NONE`; a regression test covers unmatched and balanced prompt quotes. All invalid store rows, manifests, baselines, and incomplete runs were removed, while deterministic canonical FLACs were retained and reused. The rebuilt immutable gate contains 46,600 rows / 65.7268 hours (44,823 train and 1,777 dev), maximum prompt length 145 characters, zero tabs/newlines, zero missing audio, and zero embedded-tokenizer `<unk>` rows.

The matched 1,000-row quality pilot did not justify a cleaned-data arm. NeMo forced alignment placed all 1,000 references and found no missing/empty CTM; apparent low word coverage came from contractions, while low aligned-span ratios reflected benign clip padding. Base-ASR boundary mismatch was near zero (mean beginning/end error 0.028/0.073 characters). Filtering these rows would be arbitrary student-to-base agreement filtering, so no CTC or edge cutoff was applied. The next independent data candidate is a frozen People's Speech clean slice with 2,509 rows / 10.0012 hours (2,380 train and 129 dev), zero missing audio, and zero tokenizer `<unk>` rows.

## CV26 20K promotion and broad confirmation — 2026-07-15

The promoted CV26 schedule restarted from the unchanged official base on the exact frozen 65.7268-hour manifest with seed 0, LR `1e-4`, 2,000 warmup steps, 20,000 maximum steps, and validation every 2,000 steps. Step 16,000 was selected by `val_loss=1.9181`; the final step scored `1.9197`. The exported best model is `runs/parakeet/english-source-common-voice-26-gate-60h-v1-lr1e4-promote20k-seed0/english-source-common-voice-26-gate-60h-v1-lr1e4-promote20k-seed0_best-valloss.nemo`.

On the matched 1,777-row CV26 dev set, normalized WER improved from 7.0167% to 6.6356% and CER from 2.5776% to 2.4144%, with zero empty hypotheses. The candidate made 1,097 word errors versus the base's 1,160 over 16,532 reference words: 63 fewer errors, a 0.3811-point absolute / 5.43% relative WER improvement. A 20,000-replicate paired clip bootstrap gave a candidate-minus-base 95% interval of `[-0.6646, -0.0968]` percentage points; 99.49% of replicates favored the candidate. A warm-cache rerun reached 700.7 RTFx at 1.83 GiB peak allocation; a separate People's Speech pass reached 1,003.8 RTFx, confirming that the initial 207.7 RTFx measurement was contaminated by first-batch CUDA-graph compilation.

The broad fp32 matrix prevents a source-specific win from silently becoming the default model:

| Model / phase                        | CV26 WER | GigaSpeech WER |  AMI WER | People's WER | Macro WER | Decision                                                                 |
| ------------------------------------ | -------: | -------------: | -------: | -----------: | --------: | ------------------------------------------------------------------------ |
| Unchanged official base              |  7.0167% |        2.7336% | 20.3805% |      9.7420% |   9.9682% | General baseline                                                         |
| CV26 20K, LR 1e-4                    |  6.6356% |        2.8968% | 23.4880% |     11.7853% |  11.2014% | Keep as CV26-domain winner; reject as general promotion                  |
| Fresh four-source replay 2K, LR 1e-4 |  7.0288% |        2.8968% | 23.8147% |     10.6295% |  11.0924% | Reject; natural-frequency mixing from base degraded every domain         |
| CV winner + recovery replay 2K, 3e-5 | 24.9335% |       19.0942% | 33.5439% |     13.5810% |  22.7882% | Reject; sequential replay catastrophically erased recognition everywhere |

The fresh replay export is `data/datasets/replay-cv26-ami-giga-peoples-106h-v0`: 89,158 train rows / 106.2279 hours and 4,058 dev rows / 4.4831 hours, with zero duplicate sample IDs and zero missing audio. Its step-2,000 best checkpoint improved the mixed raw validation loss to 27.8256 but worsened every normalized domain WER. The sequential recovery export is `data/datasets/replay-ami-giga-peoples-43h-v0`: 44,335 train rows / 42.9417 hours and 2,281 dev rows / 2.0425 hours. Warm-starting the CV winner at LR `3e-5` drove validation loss from 25.8608 at step 500 to 15.2735 at step 2,000 while normalized WER collapsed on all four domains. This proves that a lower mixed raw-label loss is not a valid general-English promotion signal when the corpora use conflicting case, punctuation, disfluency, and transcript-style conventions.

The bounded campaign stops here rather than spending a 20K run on a rejected 2K recipe. The next viable general-model path is to make the training target and sampler explicit: harmonize case/punctuation style without changing lexical labels, sample by source rather than natural cut count, retain separate per-source validation metrics, and constrain updates toward the base or a teacher so replay cannot erase pretrained behavior. A new recipe must beat the unchanged base on the four-domain macro and avoid a material regression on any individual domain before it advances to 20K.

## Balanced lexical replay and base retention — 2026-07-15

The next general-model gate implements the requirements above without modifying any source manifest. `data/datasets/general-balanced-lexical-v1` contains hash-pinned derivative views with lowercase lexical targets, punctuation removed, lexical apostrophes preserved, and all other words unchanged. The four training manifests retain their original row counts and are sampled independently at 25% each through NeMo's weighted `input_cfg`; the selection manifest contains 516 rows, exactly 129 per source. A 12,000-cut sampler preflight produced 2,970 Common Voice, 3,000 AMI, 3,063 GigaSpeech, and 2,967 People's Speech cuts. People's Speech contributed only 2,380 unique IDs, proving that its stream repeated across the finite source boundary instead of terminating or silently changing the weights.

The first weighted run exposed two trainer-path defects before the recipe could be treated as canonical. NeMo's older weighted `manifest_filepath` list stopped yielding after a repeat boundary, and its `max_open_streams` shortcut sampled only two of four non-tarred sources. The shared trainer now emits the supported grouped `input_cfg` shape; the live sampler distribution above and a full uninterrupted 2K L2-SP run verify the corrected path. The earlier unconstrained 2K result remains diagnostic because it crossed this repair through checkpoint resumes.

L2-SP anchors every trainable parameter to its restored official-base value by adding `0.001 * (parameter - base_parameter)` to the unscaled gradient before clipping. The anchors are non-persistent buffers: they move with the model during training but do not enter checkpoints or `.nemo` artifacts. The matched 2K run selected step 2,000 by `val_loss=2.6783`. RMS parameter drift was 0.002486 versus 0.002856 for the unconstrained final, and the final half-squared L2 distance was 354.2765, an effective penalty of 0.3543.

The untouched fp32 four-domain matrix is:

| Model / retention point         | CV26 WER | GigaSpeech WER |  AMI WER | People's WER | Macro WER | Decision                                                                          |
| ------------------------------- | -------: | -------------: | -------: | -----------: | --------: | --------------------------------------------------------------------------------- |
| Unchanged official base         |  7.0167% |        2.7336% | 20.3805% |      9.7420% |   9.9682% | General baseline                                                                  |
| Weighted lexical, unconstrained |  7.6700% |        3.5088% | 15.4650% |      7.9257% |   8.6424% | Diagnostic macro win; reject direct promotion because CV26 and GigaSpeech regress |
| Weighted lexical, L2-SP `1e-3`  |  7.7970% |        3.3048% | 14.3832% |      8.1527% |   8.4094% | Best unblended macro; reject direct promotion because CV26 and GigaSpeech regress |
| Square-root-hours, L2-SP `1e-3` |  7.8393% |        3.3456% | 14.4268% |      8.3179% |   8.4824% | Reject; worse than equal source weighting on every domain                         |
| 33% L2-SP candidate / 67% base  |  6.9804% |        2.6928% | 18.9937% |      9.0196% |   9.4216% | Seed-0 general candidate; beats base on all four domains, pending confirmation    |

The square-root-hours run changed only sampling probabilities: AMI `0.255907`, Common Voice `0.418115`, GigaSpeech `0.163934`, and People's Speech `0.162044`. It selected step 2,000 at `val_loss=2.7531`; its RMS parameter drift was 0.002484 and its effective L2-SP penalty was 0.3536, both effectively matched to the equal-source run. The worse four-domain matrix therefore rejects this sampler rather than attributing the result to a different amount of model movement.

The interpolated candidate is `runs/parakeet/english-general-balanced-lexical-equal4-l2sp1e3-2k-seed0/interpolate-base-alpha0.33.nemo`, 459,233,280 bytes with SHA-256 `3463029ca1355d61d445ba753e9541f3fc7c391a80b01cdd3e5d9e40f6120356`. Its macro improvement is 0.5466 absolute points / 5.48% relative. Warm-cache RTFx was 864.4 on CV26, 484.4 on GigaSpeech, 439.1 on AMI, and 1,086.9 on People's Speech; the architecture and parameter count are unchanged. Weight interpolation is a cheap retention frontier, not proof that the recipe is release-ready.

The required punctuation/capitalization audit rejects release-seed promotion of this lexical recipe. Original training-label styles are mutually incompatible: AMI is 100% uppercase with terminal punctuation on only 0.6% of rows, Common Voice has terminal punctuation on 96.6% of rows, GigaSpeech is 100% uppercase with terminal punctuation on 62.1%, and People's Speech is 100% lowercase with no terminal punctuation. Lowercase lexical views eliminate that training conflict but remove all positive PnC supervision. On the punctuated CV26 development set, the official base produced terminal punctuation on 99.6% of hypotheses and commas on 24.7%; the raw L2-SP model produced 0% for both, while the 33% interpolation recovered only 39.2% terminal punctuation and 4.2% commas. Initial capitalization recovered to 98.1%, but the punctuation loss is unacceptable for dictation even though punctuation-neutral WER improves. The next seed-0 work must establish a word-preserving, consistently restored PnC label surface and score lexical WER and PnC quality separately before any second seed or longer promotion run.

## Consistent PnC restoration and general gate — 2026-07-15

A fixed 512-row CV26 development pilot compared NVIDIA's dedicated `punctuation_en_bert` lexical tagger with locally served Qwen 3.5 4B in non-thinking mode. NVIDIA preserved all 512 word sequences, reached punctuation micro-F1 `0.8836` and capitalization accuracy `0.9783`, and processed 1,060 rows/s. Qwen reached punctuation F1 `0.8725` and capitalization accuracy `0.9828`, but changed words in 16/512 rows and processed 15.3 rows/s after warmup. The deterministic NVIDIA tagger therefore won the label-production lane; the LLM was shut down before any corpus-scale pass.

`data/datasets/general-balanced-pnc-nemo-v1` restores a consistent native punctuation-and-capitalization surface over the exact 93,216 rows from the four-source lexical derivative while preserving every word sequence and the same 516 balanced validation IDs. The isolated NeMo 1.23 runner processed the pool in 116.36 seconds at 801.1 rows/s, with a 5.29 GiB CUDA peak. The immutable restoration output SHA-256 is `6a83f4bc1b08b81d2dda00db04db3b6010ead8ff8c1f07b6fbefef982eaa9e4c`.

The matched 2K gate retained equal 25% source weights, LR `1e-4`, 2,000-step warmup, seed 0, and L2-SP `1e-3`. It selected step 2,000 at `val_loss=4.1079`, consumed 35.5520 actual unpadded audio-hours, and ended with RMS parameter drift `0.002483`, effectively identical to the lexical comparator's `0.002486`. The direct best-checkpoint SHA-256 is `c94dae62230054c68cff347fa5ac55d00377b18a5498c4c4791bf077dfb6dfda`.

The fp32 retention frontier is:

| Model / retention point          | CV26 WER | GigaSpeech WER |  AMI WER | People's WER | Macro WER | Decision                                                     |
| -------------------------------- | -------: | -------------: | -------: | -----------: | --------: | ------------------------------------------------------------ |
| Unchanged official base          |  7.0167% |        2.7336% | 20.3805% |      9.7420% |   9.9682% | General baseline                                             |
| Lexical 33% candidate / 67% base |  6.9804% |        2.6928% | 18.9937% |      9.0196% |   9.4216% | Superseded: WER win, unacceptable punctuation loss           |
| PnC direct checkpoint            |  7.2466% |        3.1008% | 15.1528% |      8.2353% |   8.4339% | Reject direct promotion; CV26 and GigaSpeech regress         |
| PnC 33% candidate / 67% base     |  6.8776% |        2.4888% | 18.1950% |      9.2054% |   9.1917% | Clears all domains and restores native PnC                   |
| PnC 50% candidate / 50% base     |  6.8655% |        2.5296% | 17.1495% |      8.5862% |   8.7827% | Locked seed-0 frontier candidate; beats base on every domain |
| PnC 67% candidate / 33% base     |  6.8776% |        2.7744% | 16.0459% |      8.5243% |   8.5555% | Reject; crosses the no-regression boundary on GigaSpeech     |

PnC quality is scored on the fixed 953-row CV26 intersection where every compared hypothesis has the exact reference word sequence, so the formatting comparison is not biased by different lexical errors:

| Model                     | Capitalization accuracy | Punctuation micro-F1 | Exact PnC rows |
| ------------------------- | ----------------------: | -------------------: | -------------: |
| Unchanged official base   |                 98.323% |              90.595% |        76.600% |
| Lexical direct checkpoint |                 83.696% |               0.000% |         0.000% |
| Lexical 33% interpolation |                 97.850% |              43.295% |        23.610% |
| PnC direct checkpoint     |                 98.287% |              90.488% |        76.705% |
| PnC 33% interpolation     |                 98.348% |              90.850% |        77.335% |
| PnC 50% interpolation     |                 98.311% |              90.694% |        76.810% |

The first locked candidate was `runs/parakeet/english-general-balanced-pnc-nemo-equal4-l2sp1e3-2k-seed0/interpolate-base-alpha0.50.nemo`, 459,233,280 bytes with SHA-256 `48a3e31b73a5f9ef3e550eb8933564da6cc5a4065022621904b643400e66cb45`. It does not need a runtime punctuation postprocessor for normal dictation. The seed confirmation and interpolation sweep below supersede its 50% retention choice while preserving the same measured 35.5520-hour exposure budget.

## Seed confirmation and campaign matrix — 2026-07-15

The matched seed-1 replay completed all 2,000 steps with the same four PnC-restored sources, equal weights, LR `1e-4`, 2,000-step warmup, and L2-SP `1e-3`. Validation loss improved monotonically from `5.2262` at step 500 to `4.1237` at step 2,000. Its 50% base interpolation is `runs/parakeet/english-general-balanced-pnc-nemo-equal4-l2sp1e3-2k-seed1/interpolate-base-alpha0.50.nemo`, SHA-256 `26a69d022bb46f36e0361be79cb1909e627648f30b84ef08ce43ed10468eef22`.

| Model                            | CV26 WER | GigaSpeech WER |  AMI WER | People's WER | Macro WER | Decision                              |
| -------------------------------- | -------: | -------------: | -------: | -----------: | --------: | ------------------------------------- |
| Unchanged official base          |  7.0167% |        2.7336% | 20.3805% |      9.7420% |   9.9682% | General baseline                      |
| Seed 0, 50% candidate / 50% base |  6.8655% |        2.5296% | 17.1495% |      8.5862% |   8.7827% | First frontier candidate              |
| Seed 1, 50% candidate / 50% base |  6.8957% |        2.4072% | 17.1785% |      8.6687% |   8.7875% | Confirms the all-domain and macro win |

The two seeds differ by only 0.0048 macro-WER points. Both beat the unchanged base on all four frozen domains, so the bounded 2K recipe is reproducible enough to advance to external exams. The pinned external data matrix uses revision `b6bdcd0beb34f8975dc659796176d88f43aff502` of `hf-audio/open-asr-leaderboard` and materializes Common Voice, LibriSpeech clean/other, Earnings22, VoxPopuli, AMI, and GigaSpeech as immutable local manifests. Inference runs at batch 16 and writes reusable prediction ledgers. Those ledgers are rescored in an isolated PEP 723 environment with official leaderboard revision `b2dada0b970cb3eaa8dca8a755345234113fc84c` and compound-aware `kaldialign` revision `06ac40f03c3d368932adf8536965a088d54189b1`; this preserves the NeMo environment's older compatible aligner while reproducing the current official scoring path exactly. A partial batch-32 pass was invalidated by stale training-process VRAM and an unreported normalized-empty-reference edge case.

| External model          | Common Voice | Libri clean | Libri other | Earnings22 | VoxPopuli |      AMI | GigaSpeech | Macro WER | Pooled WER | Aggregate RTFx | Decision                                                      |
| ----------------------- | -----------: | ----------: | ----------: | ---------: | --------: | -------: | ---------: | --------: | ---------: | -------------: | ------------------------------------------------------------- |
| Unchanged official base |     10.7880% |     1.9998% |     4.7274% |   12.0892% |   6.6482% | 15.0147% |    8.9450% |   8.6018% |    9.2872% |          378.4 | Pinned baseline                                               |
| Seed 0, 50% candidate   |     10.7401% |     2.0074% |     4.7160% |   11.4669% |   6.6459% | 13.2743% |    8.6062% |   8.2081% |    8.8983% |          577.6 | Pass; six wins and one immaterial `+0.0075`-point clean shift |
| Seed 1, 50% candidate   |     10.6698% |     1.9753% |     4.7596% |   11.5716% |   6.6528% | 13.1111% |    8.6177% |   8.1940% |    8.8807% |          900.6 | Pass; five wins and two shifts no larger than `+0.0322`       |

The two external candidate macros differ by only 0.0141 points. Seed 0 improves the base macro by 0.3936 absolute / 4.58% relative, and seed 1 improves it by 0.4078 absolute / 4.74% relative. Aggregate RTFx increased across the fixed model order because the immutable audio and execution stack became progressively warmer; all candidates retain the exact architecture and parameter count, so these figures prove no throughput regression but are not evidence that the fine-tuned weights made inference intrinsically faster.

Common Voice Spontaneous 4 contributes 1,198 usable official train/dev clips / 2.83 hours after the 30-second model ceiling, descriptor-only rejection, and English normalization. Forty non-English labels normalized to empty and one label normalized to punctuation only; the exporter now rejects both cases. Source filters are also pushed into SQLite, reducing this source-specific export from a full 1.4 GB store scan taking minutes to an indexed export taking under one second.

The five-source PnC restoration processed 94,414 train/dev rows in 125.81 seconds at 750.5 rows/s with output SHA-256 `6af7d5f43b0b6409f200dbff89100e1dbe96d4989bafbd7286b7236f0caa60e2`. Before training on the spontaneous source, the fixed 111-row spontaneous dev gate scored 10.6764% WER for the unchanged base, 10.3504% for the existing seed-0 candidate, and 10.1059% for seed 1. The four-source recipe therefore already transfers positively to this unseen conversational set; new five-source arms must beat the seed-1 result without surrendering the frozen four-domain gains.

The equal-20% five-source arm completed at 2K with monotonically improving validation loss through step 2,000 (`5.4151`, `4.7361`, `4.4954`, `4.1652`), but its 50% interpolation failed the fp32 WER gate. It scored 6.9381% on CV26, 2.4888% on GigaSpeech, 18.0062% on AMI, 9.3498% on People's Speech, and 10.1874% on Common Voice Spontaneous. Its frozen four-domain macro was 9.1957%, versus 8.7875% for the four-source seed-1 frontier, and it also missed seed 1 on the spontaneous gate by 0.0815 point. Equal exposure therefore oversamples this 2.83-hour source and is rejected.

Reducing Common Voice Spontaneous to 10% exposure recovered most retention and produced a genuine specialized-domain gain, but it did not clear the general-model gate. The 50% interpolation scored 6.9683% on CV26, 2.6112% on GigaSpeech, 17.5561% on AMI, 8.6687% on People's Speech, and 9.6985% on Common Voice Spontaneous. Spontaneous WER improved by 0.4075 point over the four-source seed-1 frontier, while the frozen four-domain macro regressed by 0.1636 point to 8.9511%. This arm is retained as evidence that the corpus is useful for specialization, but it is rejected for the general-model promotion lane.

The People's Speech 100-hour replacement contains 25,414 unique rows / 100.0022 hours, zero missing audio, and zero embedded-tokenizer `<unk>` rows. It includes all 2,509 IDs from the earlier 10-hour slice and has no duplicate sample IDs, so its training arm replaces that slice instead of sampling the overlapping exports together. Its four-source lexical derivative contains 110,902 train rows, and the complete train/dev PnC restoration pool contains 116,121 rows with the original 129-per-source validation budget preserved.

PnC restoration of the 116,121-row People's Speech replacement pool completed in 207.98 seconds at 558.3 rows/s with output SHA-256 `6ce7f90b54b13a65810ae7cc69209ba5dd2e3edbea25f746ec1548517925ba1a`. Its 2K arm failed to replace the smaller source: the 50% interpolation scored 6.8050% on CV26, 2.8152% on GigaSpeech, 17.2584% on AMI, and 8.7719% on People's Speech for an 8.9126% macro. Moving its interpolation to 35% worsened the macro to 9.2242%; moving it to 65% improved the macro to 8.8002%, still behind the original four-source candidate. The larger source is therefore rejected rather than being promoted by size alone.

The bounded interpolation sweep changed the general-model frontier. Lower 35% candidate weight consistently over-retained the base, while higher candidate weight continued improving the four-domain macro through the raw best-validation checkpoint. Both seeds show the same direction.

| Candidate weight | Seed-0 macro WER | Seed-1 macro WER |
| ---------------: | ---------------: | ---------------: |
|              35% |          9.1569% |          9.1209% |
|              50% |          8.7827% |          8.7875% |
|              65% |          8.5709% |          8.5541% |
|              80% |          8.5335% |          8.4285% |
|             100% |          8.4327% |          8.2841% |

The raw seed-1 checkpoint improves the unchanged base macro by 1.6841 points and the former seed-1 50% frontier by 0.5034 point. It scores 7.0530% on CV26, 2.8560% on GigaSpeech, 15.2400% on AMI, and 7.9876% on People's Speech. Because it trades some CV26 retention for larger AMI and People's Speech gains, the exact external promotion matrix includes seed-0 raw, seed-1 80%, and seed-1 raw rather than declaring a winner from the internal average alone. The 10%-spontaneous arm at 65% remains a separate Pareto point: 8.7525% frozen macro and 9.5355% spontaneous WER.

The exact seven-exam matrix rejects both raw checkpoints and shows that the 80% seed-1 blend is only a technical macro winner, not a meaningful balanced improvement over 50%. Seed-1 80% lowers macro WER by 0.0069 point and pooled WER by 0.0109 point versus seed-1 50%, but it gives back 0.2229 point on Common Voice, 0.0358 on LibriSpeech clean, 0.1400 on LibriSpeech other, and 0.1271 on VoxPopuli. Its gain is concentrated in AMI. The 50% blend therefore remains the balanced deployment setting; the 20K gate reevaluates 50%, 80%, and raw because the longer run can move the retention optimum.

| External candidate | Common Voice | Libri clean | Libri other | Earnings22 | VoxPopuli |      AMI | GigaSpeech | Macro WER | Pooled WER | Decision                                      |
| ------------------ | -----------: | ----------: | ----------: | ---------: | --------: | -------: | ---------: | --------: | ---------: | --------------------------------------------- |
| Seed 1, 50%        |     10.6698% |     1.9753% |     4.7596% |   11.5716% |   6.6528% | 13.1111% |    8.6177% |   8.1940% |    8.8807% | Balanced frontier                             |
| Seed 0, raw        |     11.2542% |     2.1017% |     5.0113% |   11.4751% |   6.9455% | 12.8241% |    8.5939% |   8.3151% |    8.9811% | Reject; external retention loss               |
| Seed 1, 80%        |     10.8927% |     2.0112% |     4.8996% |   11.4669% |   6.7798% | 12.6811% |    8.5784% |   8.1871% |    8.8697% | Pareto point; negligible macro gain is narrow |
| Seed 1, raw        |     11.1826% |     2.0980% |     5.0226% |   11.6969% |   6.9568% | 12.7858% |    8.6605% |   8.3433% |    9.0092% | Reject; external retention loss               |

The seed-0 20K continuation completed the full schedule and selected step 15,500 by validation loss (`3.5917`). Validation continued improving well after apparent local plateaus at 8,000–9,000 and 12,000–13,000, so the full exposure budget was informative rather than redundant. On the frozen internal four-domain gate, 50%, 80%, and raw candidate weights scored 8.3209%, 7.7489%, and 7.5971% macro WER respectively, compared with 8.7827% for the 2K seed-0 50% candidate and 9.9682% for the unchanged base.

The pinned external matrix confirms a genuine 20K gain but moves the retention optimum. The 80% blend is the aggregate accuracy frontier at 8.0063% macro / 8.6470% pooled, improving the former seed-1 50% frontier by 0.1876 / 0.2337 point and the unchanged base by 0.5954 / 0.6403 point. The 50% blend scores 8.0502% macro / 8.7289% pooled and beats 80% on five of seven exams; 80% wins by concentrating a 0.7745-point gain on AMI plus 0.0744 on GigaSpeech. Both are promoted as distinct deployment frontiers pending seed-1 confirmation. The raw checkpoint is rejected: despite winning the internal macro, it regresses all seven official exams relative to at least one blend and scores 8.2589% macro / 8.8183% pooled.

| Seed-0 20K candidate | Common Voice | Libri clean | Libri other | Earnings22 | VoxPopuli |      AMI | GigaSpeech | Macro WER | Pooled WER | Decision                              |
| -------------------- | -----------: | ----------: | ----------: | ---------: | --------: | -------: | ---------: | --------: | ---------: | ------------------------------------- |
| 50%                  |     10.4310% |     2.0074% |     4.6138% |   11.7565% |   6.6845% | 12.2972% |    8.5609% |   8.0502% |    8.7289% | Balanced retention frontier           |
| 80%                  |     10.4603% |     2.0923% |     4.6574% |   11.9886% |   6.8365% | 11.5227% |    8.4865% |   8.0063% |    8.6470% | Aggregate accuracy frontier           |
| Raw                  |     10.6787% |     2.1395% |     4.7539% |   12.6931% |   7.3993% | 11.5610% |    8.5872% |   8.2589% |    8.8183% | Reject; broad external retention loss |

The matched seed-1 20K run reproduced the unusual late optimum: it also selected step 15,500, at validation loss `3.5941` versus seed 0's `3.5917`. Its internal 50%, 80%, and raw macros were 8.5524%, 7.7999%, and 7.6926%. The 80% blend is only 0.0510 point behind seed 0 internally, while the 50% blend is more seed-sensitive. The exact external matrix then confirmed both interpolation frontiers. Seed-1 50% scored 8.0686% macro / 8.7457% pooled, only 0.0184 / 0.0168 point from seed 0. Seed-1 80% scored 7.9924% macro / 8.6662% pooled, only 0.0139 / 0.0192 point from seed 0. Raw was not rerun externally because raw checkpoints had already failed retention at 2K on both seeds and at 20K on seed 0.

| Seed-1 20K candidate | Common Voice | Libri clean | Libri other | Earnings22 | VoxPopuli |      AMI | GigaSpeech | Macro WER | Pooled WER | Decision                             |
| -------------------- | -----------: | ----------: | ----------: | ---------: | --------: | -------: | ---------: | --------: | ---------: | ------------------------------------ |
| 50%                  |     10.4546% |     2.0206% |     4.6309% |   11.7236% |   6.6414% | 12.4515% |    8.5576% |   8.0686% |    8.7457% | Confirms balanced retention frontier |
| 80%                  |     10.5287% |     2.0489% |     4.6782% |   11.7914% |   6.7866% | 11.5970% |    8.5163% |   7.9924% |    8.6662% | Confirms aggregate accuracy frontier |

The short-form release decision does not choose a seed after inspecting the official test matrix. Seed 0 was the predeclared promotion checkpoint, so its 80% interpolation remains the accuracy candidate and its 50% interpolation remains the retention-oriented alternative; seed 1 supplies replication evidence. The 80% candidate improves the unchanged base by 0.5954 macro point / 0.6403 pooled point and retains the same 114M-parameter architecture and decoding path.

The original path-based Lhotse shutdown repairs did not reach Lightning's detached live sampler, as two exported runs and an initial disposable smoke demonstrated. The final repair finds the owning `DynamicBucketer` through Lhotse's producer-thread closure, marks its source exhausted, joins the non-daemon thread, and leaves the joined thread reference for Lhotse's generator finalizer. The focused core suite covers both loader graphs and the detached-thread case, and a weighted four-source one-step live smoke completed training, validation, both NeMo exports, interpreter shutdown, and log inspection without a traceback or ignored finalizer exception.

The official long-form inventory is pinned separately at revision `d6797370d3189c618e722721ab5b6c9be78c022c` of `hf-audio/asr-leaderboard-longform`: Earnings21 has 44 test recordings, Earnings22 has 125, and TED-LIUM has 11. Their declared download size is approximately 17.7 GB, so materialization remains behind the short-form promotion gate; CORAAL is a separate conversational robustness exam rather than a config in the current official long-form repository.

| ID  | Experiment or preparation                                                    | Status      | Advance condition                                                                                |
| --- | ---------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| E01 | Matched four-source PnC seed 1 at 2K                                         | Complete    | Reproduce the seed-0 all-domain gain                                                             |
| E02 | Seven-exam external matrix for unchanged base                                | Complete    | Establish one pinned local baseline                                                              |
| E03 | Seven-exam external matrix for seed-0 50% interpolation                      | Complete    | Improve external macro without a material domain collapse                                        |
| E04 | Seven-exam external matrix for seed-1 50% interpolation                      | Complete    | Confirm the external direction across seeds                                                      |
| E05 | Base and both existing candidates on Common Voice Spontaneous dev            | Complete    | Establish the novel-domain baseline before training on it                                        |
| E06 | Five-source PnC, equal 20% weights, seed 0 at 2K                             | Complete    | Validation improved, but equal exposure oversampled the 2.83-hour source                         |
| E07 | E06 at 50% interpolation on the four frozen domains plus spontaneous dev     | Complete    | Rejected: 9.1957% frozen macro and 10.1874% spontaneous both missed seed 1                       |
| E08 | Five-source PnC, spontaneous at 10% and other sources at 22.5%, seed 0 at 2K | Complete    | Lower exposure improved spontaneous speech without equal-weight collapse                         |
| E09 | E08 at 50% interpolation on the same five-domain gate                        | Complete    | Rejected for general use: 9.6985% spontaneous, but 8.9511% frozen macro                          |
| E10 | People's Speech clean 100-hour immutable replacement export                  | Complete    | Produce a tokenizer-covered, train-only replacement for the 10-hour slice                        |
| E11 | GigaSpeech S 100-hour immutable replacement export                           | Complete    | Immutable replacement completed                                                                  |
| E12 | One-at-a-time 100-hour replacement 2K gates                                  | Complete    | People's Speech and GigaSpeech replacements both failed the original four-source frontier        |
| E13 | Stronger L2-SP `2e-3` at 2K on the best data arm                             | Conditional | Run only if the unblended candidate improves but the interpolation frontier is retention-limited |
| E14 | Best surviving recipe at 20K, seed 0                                         | Complete    | 50% and 80% both pass exact external promotion; raw is rejected                                  |
| E15 | Best 20K recipe, seed 1                                                      | Complete    | Both 50% and 80% frontiers reproduce within 0.02 external macro/pooled point                     |
| E16 | Frozen official long-form matrix plus separate CORAAL confirmation           | Pending     | Short-form candidate survived; materialize and run the pinned long-form exams                    |
| E17 | 35% and 65% interpolation bracket across both seeds and new data arms        | Complete    | Higher candidate weight improved both original seeds                                             |
| E18 | 80% and raw-checkpoint continuation across both original seeds               | Complete    | Raw seed 1 established the 8.2841% internal frontier                                             |
| E19 | Exact seven-exam external matrix for seed-0 raw, seed-1 80%, and seed-1 raw  | Complete    | Keep 50% for balance; 80% gains only 0.0069 macro point and raw checkpoints regress              |
| E20 | LibriSpeech `train-clean-100` replay at 10% exposure, seed 0 at 2K           | Complete    | Rejected at 8.8700% macro versus the original four-source 2K candidate's 8.7827%; no 20K run     |

The matrix is deliberately conditional: failed 2K arms do not consume 20K runs, and larger corpora replace their smaller source rather than appearing beside an overlapping subset. This keeps every gain attributable to one data or retention change.

## Dictation campaign handoff — 2026-07-16

The next campaign is a product-specific `english-dictation-110m` lane, not another attempt to make the general checkpoint universal. The imported owned-data ledger contains 1,688 usable sessions / 51.8583 hours: 1,640 MacWhisper sessions and 48 native TimberVox sessions after excluding TimberVox's imported MacWhisper copies. The canonical source snapshot and ledger are under `/mnt/tiny-2t/peacock-asr/english-asr/data/owned-dictation/mac-import-20260716/`.

The frozen review v1 reserves 60 whole sessions and samples 200 Silero `conservative-v1` segments across all 60. The final duration-aware sample totals 22.5347 minutes with a 6.108-second median; only three clips are shorter than one second. All 1,974 emitted candidates and the complete held-out session ledger are retained beside the 200 cut FLACs, so sampling can be audited without changing the held-out identities.

The matched local LibriSpeech-clean n200 check retains all three comparison models. ARK-ASR-3B scored 1.35% WER / 0.69% CER at 88.62x realtime; Parakeet TDT v2 scored 1.32% / 0.49% at 249.58x; Parakeet TDT v3 scored 1.26% / 0.45% at 239.63x. ARK is locked as the pseudo-label teacher because the official broad English matrix is stronger; v3 and v2 are fast deployment/control baselines.

The owned-recording gate closed on 2026-07-17. The reviewer manually accepted 23 of 200 ARK rows with no issue markers, then explicitly accepted the remaining 177 as the same ideal pasted-dictation surface. `review/frozen-v1/gold-v1/decision.json` records that this is bounded user approval rather than full verbatim human transcription; the 200-row manifest SHA-256 is `e228d7c0ccc342a8746b634ce934514c11d7ecc8042604d3cd1474f5aee6abe9`. The unchanged official 110M base scored 10.8200% normalized WER / 6.4941% CER, zero empty outputs, and 37.43x realtime in the fp32 batch-16 pass. The resumable 25-hour derivative now excludes all 60 held-out sessions, uses the promoted Silero `conservative-v1` contract, splits development data by whole session, and preserves every ARK raw response before the 2K product gate.

## Dictation campaign result — 2026-07-18

The first owned-data campaign completed end to end. Serialized inference with locked `AutoArk-AI/ARK-ASR-3B` revision `1e28271b79edc97635783bea65abc89195a09ed3` produced 36,639 successful raw responses with zero unresolved failures. The immutable derivative contains 34,831 training rows / 23.8196 hours and 1,808 development rows / 1.2613 hours, split by 1,162 whole sessions after excluding all 60 held-out sessions. The training-manifest SHA-256 is `44c6faf99352e4724aee98a05af3df00b8eb45b8cd4c3abcc778dcf0836c8d0e`; the development-manifest SHA-256 is `7f03847f61c7e3ca32bd4b6f18f142c3047f627034f6984c1ed1c730f9c0505f`.

Every row below is the best `val_loss` checkpoint evaluated in fp32 with batch size 16 on the same frozen 200-row `gold-v1` manifest. Both fresh 20K seeds reached exactly 5.5419% normalized WER, proving that the gain reproduces. A 50/50 parameter average of those two checkpoints improved the primary metric again to 5.3911%.

| Candidate        | Normalized WER | Normalized CER |  Raw WER |  Raw CER | Empty |
| ---------------- | -------------: | -------------: | -------: | -------: | ----: |
| Official base    |       10.8200% |        6.4941% | 29.6113% | 11.0061% |     0 |
| 2K seed 0        |        6.6352% |        3.5158% | 14.7957% |  5.2242% |     0 |
| 2K seed 1        |        6.0320% |        3.2163% | 13.5264% |  4.7354% |     0 |
| 20K seed 0       |        5.5419% |        2.8389% | 12.2570% |  4.2465% |     0 |
| 20K seed 1       |        5.5419% |        2.7978% | 11.9397% |  4.1616% |     0 |
| 20K seed average |        5.3911% |        2.7814% | 12.1777% |  4.2141% |     0 |

The promoted artifact is `runs/parakeet/english-dictation-owned-ark-25h-v1-20k-seed-average/english-dictation-owned-ark-25h-v1-20k-seed-average.nemo`, 459,233,280 bytes with SHA-256 `71d36405b8c0b86fe8f722d49199e462abd4a1aa9e00f70566a05871fb540547`. Its normalized WER is 5.4288 points / 50.17% relatively below the unchanged base. The exact predictions and summary have SHA-256 values `32e6e621bae56fa4a464da0d4a207d4fca33f9a92a7dfaa56461c7d66897601a` and `0b3c5974cd7321181d88b58e1207a08cbc3a01603f4fcc94311a0cb461cb9084`.

The output audit passes the bounded product gate. The seed average has zero empty hypotheses, 91.0% first-word agreement and 96.5% last-word agreement versus 79.5% and 89.5% for the base, and a 1.0139 hypothesis/reference word-count ratio versus 0.9424 for the base. It emits terminal punctuation on 97.0% of rows and 7.99 punctuation marks per 100 words, close to the product references' 93.0% and 7.23; this is not punctuation collapse. Initial capitalization is 35.5% versus the references' 42.0%, so capitalization remains a measured weakness rather than a release blocker. The warm fp32 pass reached 771.84x realtime at the same 2.25 GiB peak CUDA allocation; timing comparisons use caution because the original base pass included cold-start/CUDA-graph cost.

## Promotion matrix

| Gate                | Required comparison                                                  | Promotion rule                                                                                      |
| ------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Pipeline smoke      | 336-row People's Speech microset                                     | Ingest, normalize, coverage, materialize, train, and eval complete without manual file surgery      |
| First data ablation | Official base vs two fine-tune seeds on the same untouched exams     | Mean WER improves without a material RTFx regression; no single important domain regresses silently |
| Quality filter      | Unfiltered vs source-specific filtered equal-hour exports, two seeds | Filter must win the matched training ablation; audit agreement alone is insufficient                |
| Model release       | Best `val_loss` checkpoint evaluated in fp32                         | Exact dataset revision, manifest hashes, base SHA, WER/CER, empty hypotheses, and RTFx are recorded |

Leaderboard snapshot: [english_short_latest.csv](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/main/english_short_latest.csv).
