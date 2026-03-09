# Russian

## Target

Map the data path for reproducing and then extending the released Russian
FastConformer line.

Deep source trace:

- [`ru_dossier.md`](./ru_dossier.md)

Released seed model:

- `nvidia/stt_ru_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The Russian FastConformer hybrid card lists about `1840h` total:

| Dataset | Hours listed on card | Estimated total public hours | Access class | Source | Notes |
|---|---:|---:|---|---|---|
| Golos | `1200h` | `1240h` | needs_audit | https://developers.sber.ru/portal/products/golos | Official Sber/OpenSLR-distributed Russian ASR corpus; public download is clear, but the governing terms still need audit. |
| SOVA | `310h` |  | needs_audit | https://sova.ai/dataset/ | Official SOVA page says the dataset is free and CC-BY-4.0, but the headline hours cover a mixed Russian/English collection rather than a clean Russian-only tally. |
| Dusha | `200h` | `350h` | needs_audit | https://developers.sber.ru/portal/products/dusha | Official Sber speech-emotion corpus with transcripts and public access, but the reusable training terms still need audit. |
| Russian LibriSpeech (RuLS) | `92.5h` | `98h` | public_open | https://www.openslr.org/96/ | OpenSLR corpus based on LibriVox public-domain audiobooks; this is the cleanest small seed component. |
| Common Voice 12 (ru) | `36.7h` |  | public_open | https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 | Mozilla public corpus branch used in the released recipe. |

Current confidence:

- this five-dataset stack is the exact public Russian FastConformer evidence
- public source trails exist for all five ingredients
- the main unresolved issue is not "what are the names," but the post-filter
  provenance and licensing details for the Russian-specific corpora

## Additional Russian Datasets To Track

| Dataset | Estimated total public hours | Access class | Role | Source | Notes |
|---|---:|---|---|---|---|
| Common Voice Scripted Speech 24.0 (ru) | `290.23h` | public_open | public_scale_candidate | https://datacollective.mozillafoundation.org/datasets/cmj8u3prj00o9nxxbg5pbn88l | Best clean extension of the Common Voice branch beyond the smaller CV12 slice NVIDIA used. |
| FLEURS (ru) | `10-12h` | public_open | eval_only | https://huggingface.co/datasets/google/fleurs | Strong multilingual benchmark-style Russian set; better for evaluation than scale. |
| Multilingual TEDx Russian (OpenSLR 100) | `61.12h` | needs_audit | eval_only | https://www.openslr.org/100 | Useful domain-shift benchmark, but the noncommercial/no-derivatives license needs review before training use. |
| Open STT |  | needs_audit | public_scale_candidate | https://github.com/snakers4/open_stt | Large umbrella corpus candidate, but provenance, overlap, and transcript quality need a deeper audit before counting it as clean public Russian ASR hours. |
| M-AILABS Russian | `46.78h` | needs_audit | public_replacement_candidate | https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/ | Small read-speech supplement with likely upstream audiobook overlap and custom licensing. |
| RUSLAN | `31h` | needs_audit | public_replacement_candidate | https://ruslan-corpus.github.io/ | Single-speaker TTS-style Russian resource; interesting, but weak fit for broad ASR reconstruction. |
| Common Voice Spontaneous Speech 2.0 (ru) | `2.52h` | public_open | eval_only | https://datacollective.mozillafoundation.org/datasets/cmj8u48ey004xnxzpphv4udzz | Tiny spontaneous-speech stress-test set, not a meaningful scale source. |
| Russian Speech Data by Mobile Phone (ELRA-S0443) |  | licensed | licensed_option | https://catalog.elra.info/en-us/repository/browse/ELRA-S0443/ | Large paid Russian mobile-speech option if the public path stalls. |
| Russian SpeechDat(E) Database |  | commercial | commercial_option | https://datasets.appen.com/product/russian_speechdat_e_database/ | Telephony-style paid option for closer channel coverage. |
| Russian Real-world Casual Conversation and Monologue |  | commercial | commercial_option | https://www.nexdata.ai/datasets/speechrecog/1271 | Paid spontaneous-speech option if conversational Russian is needed. |

## Compared To Parakeet / Canary Era References

The Russian FastConformer recipe above is exact. The newer multilingual
Parakeet / Canary references are broader:

| Asset family | Public source | Russian data signal | What it gives | What it does not give |
|---|---|---|---|---|
| Russian FastConformer hybrid | `nvidia/stt_ru_fastconformer_hybrid_large_pc` | exact five-dataset Russian recipe | exact named seed stack and recipe-hour split | released manifest, filters, or subset file lists |
| Related bilingual FastConformer | `nvidia/stt_kk_ru_fastconformer_hybrid_large` | corroborates repeated Russian corpus names | confirms `Golos` / `SOVA` / `Dusha` recur in NVIDIA Russian FastConformer releases | canonical Russian-only recipe |
| Parakeet / Canary model cards | `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-v2` | Russian is part of 25-language support and training era | proves Russian is inside the multilingual release family | Russian-only corpus manifest |
| Canary / Parakeet paper | `2509.14128` | strongest public Russian multilingual evidence | shows `Granary + NeMo ASR Set 3.0`, and Russian hour totals inside the multilingual stack | exact Russian raw-corpus list inside released multilingual checkpoints |

Most important multilingual takeaways:

- `Golos` is the clearest explicit Russian bridge into the multilingual
  `NeMo ASR Set 3.0` discussion
- `Parakeet-TDT-0.6B-v3` is trained on the ASR subset of the same overall
  `Granary + NeMo ASR Set 3.0` pool
- public multilingual materials do not expose a Russian-only manifest comparable
  to the released Russian FastConformer recipe
- the repo-local paper gives Russian multilingual hour totals, but not the exact
  Russian corpus list inside those totals

## Immediate Research Tasks

- audit the governing terms for `Golos`, `SOVA`, and `Dusha`
- determine the Russian-only usable hours inside the full `SOVA` release
- estimate public-only reproducibility of the full `1840h` recipe after license
  cleanup
- identify whether `Open STT` or other Russian umbrella corpora add real net-new
  hours after overlap and license filtering
- identify the best high-quality public Russian corpora beyond the NVIDIA seed
  stack

## Reproduction View

Russian is a strong first target because the released model already exists and
the recipe looks mostly public/openly reachable. The main blocker is not raw
hours; it is provenance cleanup around the Russian-specific corpora.

Near-term judgment:

- a public-only reproduction close to the released `115M` recipe looks realistic
- a strict "clean-open" reproduction is not fully proven until `Golos`,
  `Dusha`, and `SOVA` are audited
- scaling much beyond the seed recipe likely requires overlap control and
  stronger quality filtering, not just more hours
- the Russian FastConformer story is much more explicit than the Russian
  Parakeet / Canary story
- Russian remains one of the best candidates for a serious non-English branch
  once the seed licenses are clarified

## Open Questions

- Which Russian datasets in the card are truly open vs registration-gated?
- How many Russian-only usable hours are actually available in `SOVA` after
  removing the English portion?
- What exactly are the Russian corpus components inside released multilingual
  `Parakeet` / `Canary` checkpoints beyond the broad `Granary + NeMo ASR Set 3.0`
  statement?
- What additional Russian ASR corpora are good enough to push beyond the
  released `115M` recipe?
- Is there enough audited public Russian speech to justify a serious XL
  (`~600M`) run without licensed acquisitions?
