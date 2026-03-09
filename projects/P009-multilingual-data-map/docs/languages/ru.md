# Russian

## Target

Map the data path for reproducing and then extending the released Russian
FastConformer line.

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

## Additional Russian Datasets To Track

| Dataset | Estimated total public hours | Access class | Role | Source | Notes |
|---|---:|---|---|---|---|
| Common Voice Scripted Speech 24.0 (ru) | `290.23h` | public_open | public_scale_candidate | https://datacollective.mozillafoundation.org/datasets/cmj8u3prj00o9nxxbg5pbn88l | Best clean extension of the Common Voice branch beyond the smaller CV12 slice NVIDIA used. |
| FLEURS (ru) | `10-12h` | public_open | eval_only | https://huggingface.co/datasets/google/fleurs | Strong multilingual benchmark-style Russian set; better for evaluation than scale. |
| Multilingual TEDx Russian (OpenSLR 100) | `61.12h` | needs_audit | eval_only | https://www.openslr.org/100 | Useful domain-shift benchmark, but the noncommercial/no-derivatives license needs review before training use. |

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
- Russian remains one of the best candidates for a serious non-English branch
  once the seed licenses are clarified

## Open Questions

- Which Russian datasets in the card are truly open vs registration-gated?
- How many Russian-only usable hours are actually available in `SOVA` after
  removing the English portion?
- What additional Russian ASR corpora are good enough to push beyond the
  released `115M` recipe?
- Is there enough audited public Russian speech to justify a serious XL
  (`~600M`) run without licensed acquisitions?
