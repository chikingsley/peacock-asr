# Russian Evidence Dossier

This note separates three different evidence tiers that are easy to blur
together:

1. the exact released Russian FastConformer recipe
2. broader multilingual Parakeet / Canary / Granary / NeMo ASR Set 3.0
   references that include Russian but do not expose an exact Russian manifest
3. the wider Russian corpus landscape for later reconstruction or scale-up work

The goal is not to pretend the public trail is complete. The goal is to make it
obvious which Russian claims are exact, which are only multilingual mentions,
and where each claim can be checked.

## Evidence Tiers

| Tier | Meaning | Use it for | Do not use it for |
|---|---|---|---|
| exact_recipe | The released Russian asset itself names the corpus | seed reconstruction | extrapolating hidden filters or hidden subsets |
| multilingual_reference | Released multilingual assets or papers mention pooled data that includes Russian | understanding Parakeet / Canary era data composition | claiming an exact Russian manifest |
| corpus_landscape | Official dataset pages, marketplace pages, and mirrors | finding Russian sources for extension or replacement | claiming NVIDIA definitely used them |

## Exact Released Russian FastConformer Evidence

### Canonical released asset

| Asset | URL | What it gives |
|---|---|---|
| Russian FastConformer hybrid model card | https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc | Canonical public source for the released Russian recipe and its approximate corpus hours |
| Russian FastConformer hybrid README | https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc/blob/main/README.md | Alternate path to the same model-card content |
| NGC model page | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_ru_fastconformer_hybrid_large_pc | Product page for the same released checkpoint |
| Related bilingual checkpoint | https://huggingface.co/nvidia/stt_kk_ru_fastconformer_hybrid_large | Corroborating Russian corpus names in a different NVIDIA release, but not the exact Russian-only recipe |

### Exact Russian seed stack

These are the only Russian datasets that should currently be treated as exact
named ingredients of the released Russian FastConformer recipe.

| Dataset | Hours used in released Russian recipe | Access class | Why this is exact | Official source trail | Notes |
|---|---:|---|---|---|---|
| Golos | `1200h` | needs_audit | Named in the released Russian FastConformer recipe | https://developers.sber.ru/portal/products/golos ; https://www.openslr.org/114/ ; https://huggingface.co/datasets/SberDevices/Golos | Largest Russian seed component; public mirrors exist, so dedup across mirrors matters |
| SOVA | `310h` | needs_audit | Named in the released Russian FastConformer recipe | https://sova.ai/dataset/ | Only obvious public path from the seed recipe toward much larger Russian scale, but headline hours are mixed-language |
| Dusha | `200h` | needs_audit | Named in the released Russian FastConformer recipe | https://developers.sber.ru/portal/products/dusha | Emotion-oriented corpus with transcripts; useful for broader speaking style but licensing still needs audit |
| Russian LibriSpeech (RuLS) | `92.5h` | public_open | Named in the released Russian FastConformer recipe | https://www.openslr.org/96/ | Cleanest small open anchor in the Russian seed stack |
| Common Voice 12 (ru) | `36.7h` | public_open | Named in the released Russian FastConformer recipe | https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 | Smallest seed component; easiest exact-match public dataset |

### Dataset-level source notes for the exact seed

#### Golos

- Official public entry point: `Sber Developers`.
- Known public mirrors: `OpenSLR 114`, official Hugging Face dataset card, and
  public GitHub mirrors.
- Important nuance: `Golos` is the same corpus family across those mirrors, so
  they are not additive.
- Public descriptions expose `crowd` and `farfield` subsets; the public trail is
  still not enough to recover the exact post-filter manifest used in the
  released checkpoint.

#### SOVA

- Official public entry point: `sova.ai/dataset`.
- The official release headline is much larger than the `310h` NVIDIA used.
- The public SOVA page describes a mixed-language collection, so the Russian-only
  usable total is still unresolved.
- Public subset names such as `RuAudiobooksDevices` and `RuDevices` appear in
  mirrors and secondary references, but they should be treated as slices of
  `SOVA`, not independent corpora.

#### Dusha

- Official public entry point: `Sber Developers`.
- Publicly framed as a speech-emotion dataset with transcripts and labels.
- That matters because it broadens speaking style beyond pure read speech, but it
  also means ASR-focused filtering choices inside the released checkpoint are not
  publicly recoverable.

#### Russian LibriSpeech (RuLS)

- Official public entry point: `OpenSLR 96`.
- Public-domain audiobook lineage makes this the cleanest legal anchor in the
  released Russian stack.
- Domain is narrow: read audiobooks rather than conversational or spontaneous
  speech.

#### Common Voice 12 (ru)

- Official public entry point for the exact seed snapshot: `Common Voice 12.0`
  on Hugging Face / Mozilla release line.
- Important nuance: later Common Voice releases are part of the same corpus
  family and should not be blindly summed with `v12` as net-new hours.

## Russian Datasets Mentioned In FastConformer vs Parakeet / Canary References

This is the core "what is mentioned where" map.

| Dataset / corpus family | Exact Russian FastConformer recipe | Multilingual Parakeet / Canary reference | Interpretation |
|---|---|---|---|
| Golos | yes | yes, via NeMo ASR Set 3.0 discussion | strongest bridge between the Russian seed recipe and the newer multilingual stack |
| SOVA | yes | no direct public Parakeet / Canary mention recovered | treat as Russian FastConformer-only evidence for now |
| Dusha | yes | no direct public Parakeet / Canary mention recovered | treat as Russian FastConformer-only evidence for now |
| Russian LibriSpeech (RuLS) | yes | no direct public Parakeet / Canary mention recovered | Russian-only seed evidence, not a known multilingual ingredient |
| Common Voice | yes, `Common Voice 12 (ru)` | yes, but only at multilingual corpus-family level | exact for Russian seed, broad-only for Parakeet / Canary |
| FLEURS | no, not seed | yes in multilingual data / eval discussion | safest current Russian classification is eval-only |
| MLS | no, not seed | yes in multilingual data / eval discussion | broad multilingual mention, not a recovered Russian training ingredient |
| VoxPopuli | no, not seed | yes in Granary / multilingual composition discussion | important for multilingual composition, not recovered as a Russian seed corpus |
| Granary | no | yes | multilingual training pool, not an exact Russian manifest |
| NeMo ASR Set 3.0 | no | yes | multilingual high-quality pool that includes Russian material, but not publicly broken down into a Russian manifest |
| CoVoST2 | no | yes, evaluation coverage only | evaluation-only evidence |

## Released Multilingual Parakeet / Canary Evidence Relevant To Russian

### Public multilingual assets

| Asset | URL | Russian-relevant signal | What it does not give |
|---|---|---|---|
| Parakeet TDT 0.6B v3 model card | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | Russian is supported among 25 European languages; card ties training to `Granary` and `NeMo ASR Set 3.0` | no Russian-only corpus list |
| Canary 1B v2 model card | https://huggingface.co/nvidia/canary-1b-v2 | Same multilingual training era and data-family framing | no Russian-only corpus list |
| NVIDIA multilingual speech blog | https://blogs.nvidia.com/blog/speech-ai-dataset-models | Connects `Granary`, `Canary-1b-v2`, and `Parakeet-tdt-0.6b-v3` as one released stack | no Russian-only data breakdown |
| Canary / Parakeet paper | https://huggingface.co/papers/2509.14128 | Best public numeric breakdown for Russian inside the multilingual training stack | still no Russian manifest or file list |

### What the paper gives for Russian

The repo-local paper copy is the strongest public evidence for Russian inside the
released multilingual Parakeet / Canary era:

- `Granary` is the main multilingual source, built from `MOSEL`, `YTC`, and
  `YODAS`.
- `NeMo ASR Set 3.0` adds `227k` hours of human-labeled data and explicitly
  names `FLEURS`, `Common Voice`, `MLS`, and language-specific datasets such as
  `Golos` for Russian.
- `Parakeet-TDT-0.6B-v3` is trained on the ASR subset of the same overall
  training pool.
- The appendix exposes a Russian numeric decomposition inside the multilingual
  stack:
  - ASR: `20460.39h` from `Granary` + `1716.46h` from `NeMo`
  - X->En: `19595.31h` from `Granary` + `1263.28h` from `NeMo`
  - En->X: `8511.02h` from `NeMo` + `20262.18h` supplementary
- Russian appears on `FLEURS` and `CoVoST2` evaluation coverage, but not on the
  `MLS` coverage table in that paper.

Practical reading:

- the multilingual Parakeet / Canary story for Russian is real and well-supported
- the exact Russian corpus composition inside that multilingual story is still
  not public enough to reconstruct exactly
- `Golos` is the only explicit Russian-specific dataset bridge currently visible
  in the public multilingual references

## Russian Corpus Landscape For Reconstruction And Scale-Up

### Strong curated set

These are strong enough to keep in the curated Russian note now.

| Dataset | Access class | Role | Source | Why it matters |
|---|---|---|---|---|
| Golos | needs_audit | nvidia_seed_recipe | https://developers.sber.ru/portal/products/golos | indispensable seed corpus |
| SOVA | needs_audit | nvidia_seed_recipe / public_scale_candidate | https://sova.ai/dataset/ | only obvious public path to much larger Russian scale |
| Dusha | needs_audit | nvidia_seed_recipe | https://developers.sber.ru/portal/products/dusha | gives a less read-speech-heavy ingredient than RuLS |
| RuLS | public_open | nvidia_seed_recipe | https://www.openslr.org/96/ | clean legal anchor |
| Common Voice 12 (ru) | public_open | nvidia_seed_recipe | https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 | exact released public seed component |
| Common Voice Scripted Speech 24.0 (ru) | public_open | public_scale_candidate | https://datacollective.mozillafoundation.org/datasets/cmj8u3prj00o9nxxbg5pbn88l | cleanest net-new public extension after the older CV12 slice |
| FLEURS (ru_ru) | public_open | eval_only | https://huggingface.co/datasets/google/fleurs | standard multilingual Russian benchmark |
| Multilingual TEDx Russian (OpenSLR 100) | needs_audit | eval_only | https://www.openslr.org/100 | good domain-shift benchmark, but training use is license-sensitive |
| Russian Speech Data by Mobile Phone (ELRA-S0443) | licensed | licensed_option | https://catalog.elra.info/en-us/repository/browse/ELRA-S0443/ | large paid Russian mobile-speech option |
| Russian SpeechDat(E) Database | commercial | commercial_option | https://datasets.appen.com/product/russian_speechdat_e_database/ | clear telephony-style paid option |
| Russian Real-world Casual Conversation and Monologue | commercial | commercial_option | https://www.nexdata.ai/datasets/speechrecog/1271 | valuable paid spontaneous-speech option |

### Watchlist-only items

These are real leads, but not clean enough to count as stable public Russian
training hours in the main ledger yet.

| Dataset | Source | Why it is watchlist-only |
|---|---|---|
| Open STT | https://github.com/snakers4/open_stt | huge umbrella corpus, but provenance, overlap, and label quality need a deeper audit |
| M-AILABS Russian | https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/ | small read-speech supplement with custom licensing and likely upstream audiobook overlap |
| RUSLAN | https://ruslan-corpus.github.io/ | single-speaker TTS-style resource, weak fit for broad ASR reconstruction |
| Common Voice Spontaneous Speech 2.0 (ru) | https://datacollective.mozillafoundation.org/datasets/cmj8u48ey004xnxzpphv4udzz | real public source, but too tiny to change the Russian scaling picture |

## Overlap And Counting Risks

- `Golos` on Sber Developers, OpenSLR, GitHub, and Hugging Face is one corpus
  family, not multiple additive corpora.
- `Common Voice 12` and `Common Voice 24` are version snapshots of the same
  scripted corpus line; pick one deduped Common Voice lane when counting hours.
- `RuLS` and `M-AILABS Russian` both inherit audiobook-style upstream sources, so
  overlap is plausible even when the manifests differ.
- `SOVA` subset names should be treated as parts of `SOVA`, not new corpora.
- `Open STT` is an umbrella build from books, radio, web, and call-style audio,
  so overlap and transcript quality must be audited before hour accounting.
- some commercial Russian mobile / telephony products appear across multiple
  resellers; do not assume two listings are distinct corpora without checking
  lineage.

## What Is Still Not Recoverable From Public References

1. the exact released Russian FastConformer training manifest
2. the exact filtering and normalization rules applied to `Golos`, `SOVA`, and
   `Dusha`
3. the Russian-only usable hour total inside the full `SOVA` release
4. the exact Russian corpus list inside released `Parakeet-TDT-0.6B-v3`
5. the exact Russian corpus list inside released `Canary-1B-v2`
6. any public file list that ties Russian `Granary` hours back to specific raw
   source corpora

## Bottom Line

- The strict exact public Russian recipe is still:
  `Golos + SOVA + Dusha + RuLS + Common Voice 12 (ru)`.
- The multilingual Parakeet / Canary era clearly includes Russian and clearly
  uses `Granary + NeMo ASR Set 3.0`, but the Russian corpus-level manifest is
  not public.
- `Golos` is the strongest visible bridge between the Russian FastConformer
  recipe and the later multilingual stack.
- For a future reconstruction pass, the main blocker is provenance cleanup, not
  the absence of plausible Russian data sources.
