# Spanish

## Target

Map the Spanish FastConformer recipe and separate the public-only path from the
licensed path.

Released seed model:

- `nvidia/stt_es_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The Spanish FastConformer hybrid card lists about `1424h` total:

| Dataset | Hours listed on card | Estimated total public hours | Access class | Source | Notes |
|---|---:|---:|---|---|---|
| Fisher Spanish | `141h` |  | licensed | https://catalog.ldc.upenn.edu/LDC2010S01 | LDC conversational telephone corpus; this is the clearest non-public ingredient in the released recipe. |
| Common Voice 12 (es) | `395h` | `395h` | public_open | https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 | Open Mozilla crowd corpus; later Common Voice releases may be materially larger. |
| MLS Spanish | `780h` | `938h` | public_open | https://huggingface.co/datasets/facebook/multilingual_librispeech | Public audiobook/read-speech corpus; the full Spanish branch appears larger than the NVIDIA-used subset. |
| VoxPopuli Spanish | `108h` | `166h` | public_open | https://huggingface.co/datasets/facebook/voxpopuli | Public European Parliament speech; the official dataset is larger than the slice NVIDIA used. |

## Additional Spanish Datasets To Track

| Dataset | Estimated total public hours | Access class | Role | Source | Notes |
|---|---:|---|---|---|---|
| Multilingual TEDx Spanish (mTEDx / OpenSLR 100) |  | needs_audit | public_replacement_candidate | https://www.openslr.org/100/ | Useful prepared-speech supplement, but the noncommercial/no-derivatives license needs review before training use. |
| Crowdsourced Argentinian Spanish Speech Data Set (OpenSLR 61) |  | public_open | public_replacement_candidate | https://www.openslr.org/61/ | Useful accent-diversity supplement with no clean public hour total on the official page. |
| Crowdsourced Venezuelan Spanish Speech Data Set (OpenSLR 75) |  | public_open | public_replacement_candidate | https://www.openslr.org/75/ | Useful Latin American accent supplement, though the official page surfaces archive sizes more clearly than hours. |
| MediaSpeech Spanish (OpenSLR 108) | `10h` | public_open | eval_only | https://www.openslr.org/108/ | Small public media-speech benchmark; better for robustness checks than bulk recipe replacement. |
| FLEURS `es_419` | `11.2h` | public_open | eval_only | https://huggingface.co/datasets/google/fleurs | Strong multilingual benchmark/eval corpus for Spanish, but too small to matter for training scale. |
| CALLHOME Spanish Speech |  | licensed | licensed_option | https://catalog.ldc.upenn.edu/LDC96S35 | Licensed conversational telephone Spanish; useful if closer telephony-domain evaluation or adaptation is needed. |
| Multi-Language Conversational Telephone Speech 2011 -- Spanish |  | licensed | licensed_option | https://catalog.ldc.upenn.edu/LDC2018S12 | Licensed telephone-speech supplement that is closer to Fisher's domain than open read-speech alternatives. |
| Appen Spanish (Spain) conversational smartphone |  | commercial | commercial_option | https://datasets.appen.com/product/esp_asr003/ | Buyable conversational Spanish smartphone corpus; more spontaneous than read prompts but still not identical to Fisher telephony. |
| Spanish Speech Data by Mobile Phone - 435 Hours (ELRA-S0447) |  | commercial | commercial_option | https://catalog.elra.info/en-us/repository/browse/ELRA-S0447/ | Large commercial Spanish mobile-speech option that is useful for scale, not literal recipe matching. |

## Immediate Research Tasks

- confirm exact licensing and transcript package path for Fisher Spanish
- compute a public-only reproduction plan without Fisher
- verify whether `mTEDx` can be used for training under the current project
  policy
- quantify how much extra public speech is recoverable by using full public MLS
  and VoxPopuli rather than the smaller NVIDIA subsets
- identify purchasable Spanish speech sources if a closer reproduction is needed

## Reproduction View

Spanish already has a clean split between open and licensed data. That makes it
a good language for two parallel plans:

- a public-only reproduction
- a closer licensed reproduction

Current judgment:

- the released recipe is not public-only because of Fisher
- the public branch is already substantial at about `1283h` from `CV12 + MLS +
  VoxPopuli`
- the fastest public-only path is to use the full public totals of MLS and
  VoxPopuli before searching for new corpus families
- if conversational telephone performance matters, a licensed or commercial
  telephony supplement still looks like the best path

## Open Questions

- What is the best public replacement for Fisher Spanish in this recipe?
- Is `mTEDx` usable for training or only for evaluation under the current
  licensing policy?
- Which Spanish commercial corpora are realistic to buy at lab scale?
