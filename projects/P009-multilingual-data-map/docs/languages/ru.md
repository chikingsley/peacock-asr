# Russian

## Target

Map the data path for reproducing and then extending the released Russian FastConformer line.

Released seed model: `nvidia/stt_ru_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

~`1840h` total per the model card.

| Dataset | Hours used | Est. available (RU) | Access | Source | Notes |
|---|---:|---:|---|---|---|
| Golos | `1200h` | `~1240h` | public_open | <https://developers.sber.ru/portal/products/golos> | Custom BY-SA license (reworked CC BY-SA 4.0 under Russian law). Attribution + ShareAlike. NVIDIA used it; we follow the same approach. Same corpus across OpenSLR 114, HF, and GitHub mirrors — not additive. |
| SOVA | `310h` | `~17,850h` | public_open | <https://github.com/sovaai/sova-dataset> | CC-BY 4.0. Russian subsets: RuAudiobooksDevices (298h), RuDevices (101h), RuYoutube (17,451h, ASR-annotated). NVIDIA used only 310h of this pool. |
| Dusha | `200h` | `~350h` | public_open | <https://github.com/salute-developers/golos/tree/master/dusha> | Speech-emotion corpus with transcripts. Same repo as Golos, same BY-SA license. HF mirror: `xbgoose/dusha`. NVIDIA used it; we follow the same approach. |
| Russian LibriSpeech (RuLS) | `92.5h` | `~98h` | public_open | <https://www.openslr.org/96/> | Public-domain audiobook lineage; cleanest legal anchor in the seed stack. |
| Common Voice 12 (ru) | `36.7h` | — | public_open | <https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0> | Exact released public seed snapshot. |

## Extension Datasets

| Dataset | Est. hours | Access | Role | Source | Notes |
|---|---:|---|---|---|---|
| Open STT | `~6,770h` | public_open | scale_candidate | <https://github.com/snakers4/open_stt> | Downloaded to `data/ru_open_stt/` (117GB). Subsets: radio_2 (1,439h), private_buriy_audiobooks (1,511h), youtube1120 (1,104h), youtube700 (701h), phone_calls (812h), youtube1120_hq (291h), stories (116h), other (52h). Excludes 754h synthetic TTS addresses and ~17h val sets. Overlap with RuLS audiobooks and SOVA YouTube probable; quality varies by subset. |
| Common Voice Scripted 24.0 (ru) | `290.23h` | public_open | scale_candidate | <https://datacollective.mozillafoundation.org/datasets/cmj8u3prj00o9nxxbg5pbn88l> | Best clean extension of the CV branch beyond the smaller CV12 slice. |
| Granary (ru) | see note | public_open | multilingual_pool | <https://huggingface.co/datasets/nvidia/Granary> | Russian is one of 25 supported languages (VoxPopuli, YODAS, YouTube-Commons). CC-BY-3.0 (YODAS) + CC-BY-4.0 (MOSEL). Manifests only — audio downloaded separately from source corpora. |
| Multilingual TEDx Russian (OpenSLR 100) | `61.12h` | needs_audit | eval_only | <https://www.openslr.org/100> | Good domain-shift benchmark; noncommercial / no-derivatives license blocks training use. |
| FLEURS (ru) | `10–12h` | public_open | eval_only | <https://huggingface.co/datasets/google/fleurs> | Standard multilingual benchmark; better for evaluation than scale. |
| M-AILABS Russian | `46.78h` | needs_audit | replacement_candidate | <https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/> | Small read-speech supplement; custom license and likely RuLS audiobook overlap. |
| RUSLAN | `31h` | needs_audit | replacement_candidate | <https://ruslan-corpus.github.io/> | Single-speaker TTS-style resource; weak fit for broad ASR reconstruction. |
| Common Voice Spontaneous 2.0 (ru) | `2.52h` | public_open | eval_only | <https://datacollective.mozillafoundation.org/datasets/cmj8u48ey004xnxzpphv4udzz> | Too small to move the scaling picture. |
| Russian Speech Data by Mobile Phone (ELRA-S0443) | — | licensed | licensed_option | <https://catalog.elra.info/en-us/repository/browse/ELRA-S0443/> | Large paid Russian mobile-speech option. |
| Russian SpeechDat(E) Database | — | commercial | commercial_option | <https://datasets.appen.com/product/russian_speechdat_e_database/> | Telephony-style paid option. |
| Russian Real-world Casual Conversation | — | commercial | commercial_option | <https://www.nexdata.ai/datasets/speechrecog/1271> | Paid spontaneous-speech option for conversational Russian. |

## Parakeet / Canary Era

The seed recipe above is exact. The newer multilingual models include Russian but do not expose a Russian-only manifest.

| Asset | Source | Russian signal | What it does not give |
|---|---|---|---|
| Russian FastConformer hybrid | `nvidia/stt_ru_fastconformer_hybrid_large_pc` | Exact five-dataset recipe | Manifest, filters, or subset file lists |
| Bilingual FastConformer (kk+ru) | `nvidia/stt_kk_ru_fastconformer_hybrid_large` | Corroborates Golos / SOVA / Dusha reuse | Canonical Russian-only recipe |
| Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3` | Russian among 25 languages; trained on Granary + NeMo ASR Set 3.0 | Russian-only corpus manifest |
| Canary 1B v2 | `nvidia/canary-1b-v2` | Same multilingual training era | Russian-only corpus manifest |
| Canary / Parakeet paper | `2509.14128` | Russian ASR hours: `20,460h` (Granary) + `1,716h` (NeMo) | Exact Russian raw-corpus list inside those totals |

Golos is the only explicit Russian-specific dataset named in both the seed recipe and the NeMo ASR Set 3.0 multilingual discussion.

## Overlap and Counting Risks

- Golos mirrors (Sber Developers, OpenSLR 114, GitHub, HF) are one corpus family — not additive.
- SOVA subsets (RuAudiobooksDevices, RuDevices, RuYoutube) are slices of SOVA, not independent corpora.
- CV 12 and CV 24 are version snapshots of the same scripted line; pick one deduplicated lane when counting hours.
- RuLS, M-AILABS Russian, and Open STT `private_buriy_audiobooks` all draw from audiobook upstream sources; overlap is probable.
- Open STT TTS addresses subset (754h) is synthetic — exclude from ASR training counts.

## What Is Not Recoverable

1. Exact released Russian FastConformer training manifest
2. Exact filtering and normalization rules applied to Golos, SOVA, and Dusha
3. Exact Russian corpus list inside released Parakeet-TDT-0.6B-v3 and Canary-1B-v2
4. File-level mapping from Russian Granary hours back to specific raw source corpora

## Reproduction View

- All five seed datasets are now confirmed public and accessible
- A public-only reproduction close to the released `115M` recipe looks realistic; main unknown is which SOVA and Golos subsets NVIDIA filtered to
- Scaling well beyond the seed recipe is already possible via Open STT (~6,000h usable) and SOVA (~17,850h RU) — bottleneck is quality filtering, not raw hours
- The Russian FastConformer story is much more explicit than the Russian Parakeet / Canary story
