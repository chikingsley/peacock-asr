# CHANGELOG — peacock-asr

Historical record of completed work, terse. Active work is in `TODO.md`; live pipeline state is `STATUS.md`.

## Tajik

- v2 trained + eval: best step_19500, FLEURS test WER 17.17 (base 19.74, v0 17.34). Recorded in EXPERIMENTS.md.
- Conversational test set + v3 — the data lever proven. Held-out 157 whole videos (frozen manifest, leakage-safe carve). Conversational held-out (1,625 clips): v0 49.89 → v3 37.65 WER (−12.2 pts / −24.5% rel from 1,070h); v3 (37.65) ≈ v2-contaminated (37.40) so it's real generalization. Shipping model `omni_ctc_300m_v2_tajik_v3_step_20000`. KenLM fusion proven (−16% rel, α=0.5/β=0).
- Export v2: WER ≤ 0.35 + descriptor-junk filter + language gate; 0 unk.
- Gate-fix recovery (`6259a355`): audited the 38,133 drops (~95% genuinely non-Tajik); added function-word vocabulary tiebreak to `keep_for_language`. +1,563 Tajik recovered, 0 regressions, 0 Russian re-admitted. Applies to v4.
- Re-ingested legacy HF datasets (`887854fe`): `hf-muhtasham` (1,300 rows, forced train) + `hf-commonvoice22` (282 gold rows, splits preserved); all 1,582 Scribe-scored, 0 failures. (CV25_tg never existed — Mozilla stopped at v17; Tajik comes from `fsicoli/common_voice_22_0`.)
- Language-learning channels wired (`887854fe`): `learning_tajik_achilovs` + three "with Chris" channels.
- Stale/done backlog items closed: export→train/eval (done by v3 cards + template); tajik `train.py` converged to preset; zero-span videos non-issue (queue.sqlite PK de-dups); 269 unscored rows are non-speech markers verify skips.
- Tidy: deleted 43 MB `ws_1.c4af3cd1` stub + empty root `src/`. Gotcha: editing a training config changes fairseq2's `ws_<hash>` (re-launch starts fresh; hand-move checkpoints to resume).

## Verify / scoring

- Script-aware verify scoring (Sonnet transliteration won the bake-off; hypothesis-only prompt) + `rescore` CLI.
- Full verify + 150k-row rescore — every scoreable row has an honest score.
- Dev split: FLEURS dev/test are the benchmarks (gates train-only by design — `Selection.gated_splits`).
- Codex review of breaker/renewal work — all 7 findings fixed (`c5b4cedf`).

## HuggingFace

- Policy set 2026-06-10: HF (Peacockery org, public) is the archive, local is working state. Shipped versions get plain-prose cards; superseded versions recorded + deleted locally. Naming: models `<family>-<size>-<language>`, datasets `<language>-asr-corpus-vN` + `<language>-asr-<scope>`.
- Fleurs mirror dupes deleted from HF (`fleurs-parquet`, `google-fleurs`).
- `hf-upload` skill bans the broken batch pattern; per-shard committer used for large sets.

## Templates / architecture (one pipeline, one structure)

- Deleted the fused path + chunks/align (`a4d2f067`): ~850 lines removed (`create/run.py`, `pipeline.py`, `align.py`, `fuse/stitch.py`, `fuse/polish.py`, `segmenters/chunks.py`, package `cli.py` + `omni-curator` entry point, tajik `cmd_label`); `cut_audio` moved to `create/audio.py`.
- `create/` reorg (`6c70295b`): flat, one module per stage in pipeline order (`youtube → queue → vad → segment → transcribe → fuse → labelq`).
- Curate-side language template (`8823c481`, Codex xhigh reviewed): 12-command CLI in `omni_curator/project.py`, parameterized by frozen `CuratorProject`; tajik + georgian `curate.py` ~45 lines of config. Ingest sources are a registry (`IngestFn`); coverage gate injected; fail-fast validation. Recipe `docs/NEW_LANGUAGE.md`.
- Model-side language template (`18248691`): `omni_finetune_core/project.py` owns train + eval via `FinetuneProject`; pinned typed `TrainingPreset`s (tajik v3 field-equivalent to YAML, `configs/` deleted) + georgian `--regime` path; `fragment_cache_dir` in typed config; eval ported with injected normalizer; 7 core tests.
- Georgian model side on the template (`georgian-train --regime gpu_max`); 145.3 h v0 export existed.
- persian-asr migration phase 1 (`38b9779f`): `src/persian_asr/` template package — `persian-curate`, `persian-train-v2` (scribe-v4-rewarm pinned, 0 field diffs), `persian-eval-v2`; production checkpoint registered (step_7000, dev WER 11.15). Legacy cards imported, zero removals.
- Killed `Any` types (ProcessFn/Mapping/SuperwhisperClient throughout).

## Curator package (this session)

- Package reorg (`audit/` `data/` `scribe/`); dead-code/shim inventory (codex: zero stale refs).
- Live Scribe concurrency control + cross-job balancer (`scribe/concurrency.py`, `scribe/balance.py`).
- GPU/hybrid VAD + codex fixes (`vad.py` NVML `resolve_devices`; `segment.py` thread caps + worker-exit checks).
- Dev tooling: ruff, ty, vulture, ai-slop-detector, dslop.
- Verify hardened for the SuperWhisper async-API migration (rides silent-clip 200s).

## Factory prereqs (design: `factory_plan.md`)

- P1 — video claim-tokens + lease guard (`queue.py`).
- P2 — `merge` preserves `scribe_wer`/`meta` (`store.insert_if_absent`).
- P3 — verify unscoreable sentinel (`scribe_status`).
- P4 — abortable download (`--disk-guard`, mid-channel).
- Source-audio archiver: `create/archive.py` + `cmd_archive` (built, tested, running).

## Status snapshots (history)

- 2026-06-13: Tajik v3 ships (FLEURS 16.9 / conversational 37.6), v4 scale run wired (80 channels) + downloading. Farsi (ex-persian) atomic rename done; production `omni_ctc_300m_v2_farsi_v4_step_41000`, parked. Dari project scaffolded (`50817454`, `fas_Arab`, 27 channels, Farsi warm-start card). Georgian v0 trained (pooled 20.7 WER) + KenLM (24.7→18.9 FLEURS). GPU down — hardware bus-drop under load (Xid 79) ×2; needed host reboot + power cap (~175W) + temp logging before any training/eval.
