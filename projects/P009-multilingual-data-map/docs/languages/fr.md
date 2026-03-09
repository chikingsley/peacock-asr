# French

## Target

Map the French FastConformer recipe and identify whether it is the best Western
Europe public-data branch after Spanish.

Released seed model:

- `nvidia/stt_fr_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The French FastConformer hybrid card lists about `1800h` total:

| Dataset | Hours listed on card | Estimated total public hours | Access class | Source | Notes |
|---|---:|---:|---|---|---|
| Common Voice 12 (fr) | `710h` | `710h` | public_open | https://commonvoice.mozilla.org/fr/datasets | Clean public seed corpus; NVIDIA explicitly cites the French v12 slice at about `710h`. |
| MLS French | `925h` | `1333h` | public_open | https://www.openslr.org/94/ | Strong public audiobook seed; the public French pool appears materially larger than the NVIDIA-used subset. |
| VoxPopuli French | `165h` | `211h` | public_open | https://huggingface.co/datasets/facebook/voxpopuli | Public parliamentary speech; the official dataset appears larger than the slice NVIDIA used. |

## Additional French Datasets To Track

| Dataset | Estimated total public hours | Access class | Role | Source | Notes |
|---|---:|---|---|---|---|
| Audiocite.net (OpenSLR 139) | `6682h` | needs_audit | public_scale_candidate | https://www.openslr.org/139/ | Huge French audiobook pool and the best obvious scale-up candidate, but file-level licenses are mixed and need filtering. |
| African Accented French (OpenSLR 57) | `22h` | public_open | public_replacement_candidate | https://www.openslr.org/57/ | Small but valuable accent-diversity add-on under Apache 2.0; good robustness supplement, not a core scale source. |
| TCOF (Traitement de Corpus Oraux en Francais) | `124h` | needs_audit | public_replacement_candidate | https://www.cnrtl.fr/corpus/tcof/ | Useful spontaneous French speech, but the noncommercial license needs policy review before training use. |
| FLEURS French | `12h` | public_open | eval_only | https://huggingface.co/datasets/google/fleurs | Public standardized multilingual benchmark; strong eval set, not a serious scaling source. |
| ESTER / EPAC French broadcast news |  | licensed | licensed_option | https://catalogue.elra.info/en-us/repository/browse/ELRA-S0305/ | Classic French broadcast-news lane distributed through ELRA rather than public-open. |
| ETAPE |  | licensed | licensed_option | http://www.elda.org/projects/archived-projects/etape/ | French TV and radio benchmark with more spontaneous speech than read/audiobook sets. |
| Appen French SpeechDat II FDB-5000 |  | commercial | commercial_option | https://datasets.appen.com/product/french_speechdat_ii_fdb-5000/ | Clear commercial telephony-scale French option if cleaner phone/domain coverage is needed. |
| Appen French conversational telephony (`FRF_ASR001`) |  | commercial | commercial_option | https://datasets.appen.com/product/frf_asr001/ | Smaller conversational-phone add-on that helps domain balance more than raw scale. |

## Immediate Research Tasks

- identify which `Audiocite.net` subsets are actually usable after license
  filtering
- identify additional open French corpora beyond the NVIDIA recipe
- identify purchasable French speech sources
- estimate whether French has enough clean public speech to justify scaling
  beyond the released `115M` model

## Reproduction View

French looks like a strong public-only lane with relatively large hours. It is
probably the cleanest "high-hours public Western Europe" FastConformer branch.

Current judgment:

- French is the cleanest high-hours public-first branch in the seed set
- the released seed recipe is already strong enough for a serious public-only
  reproduction
- the biggest extra public hours beyond the seed skew toward read speech and
  mixed-license sources rather than clean conversational data
- the best expansion framing is likely tiered: clean public reproduction first,
  audited public-plus next, then licensed/commercial domain balancing

## Open Questions

- Is French actually easier to scale publicly than Spanish once Fisher is
  removed from the Spanish recipe?
- How much of `Audiocite.net` is usable after filtering out restrictive subsets?
- Which French licensed or commercial corpora are worth buying for domain
  balance after the public seed recipe is exhausted?
