# Italian

## Target

Map the smallest released FastConformer language recipe that still looks
realistic to reproduce quickly.

Released seed model:

- `nvidia/stt_it_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The Italian FastConformer hybrid card lists about `487h` total:

| Dataset | Hours listed on card | Estimated total public hours | Access class | Source | Notes |
|---|---:|---:|---|---|---|
| Common Voice 12 (it) | `220h` |  | public_open | https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 | Core NVIDIA seed corpus; public and easy to obtain, but the clean Italian-only CV12 total still needs confirmation. |
| MLS Italian | `214h` | `279.43h` | public_open | https://huggingface.co/datasets/facebook/multilingual_librispeech | Strong public read-speech seed; the full public Italian branch appears larger than the NVIDIA-used subset. |
| VoxPopuli Italian | `53h` | `91h` | public_open | https://huggingface.co/datasets/facebook/voxpopuli | Public parliament-speech seed; the official dataset appears larger than the slice NVIDIA used. |

Tokenizer note from the card:

- SentencePiece BPE
- vocab size `512`

## Additional Italian Datasets To Track

| Dataset | Estimated total public hours | Access class | Role | Source | Notes |
|---|---:|---|---|---|---|
| FLEURS Italian (`it_it`) | `10-12h` | public_open | eval_only | https://huggingface.co/datasets/google/fleurs | Best obvious open Italian eval set to add immediately, but too small to matter for training scale. |
| ASR-ItaCSC | `10.43h` | needs_audit | public_scale_candidate | https://magichub.com/datasets/italian-conversational-speech-corpus/ | Useful conversational Italian speech, but the gated download and restrictive license need review before training use. |
| KIParla |  | needs_audit | eval_only | https://kiparla.it/en/il-corpus/ | Important spoken-Italian corpus with aligned transcripts, but current audio access is not a frictionless public-training path. |
| EVALITA 2011 ASR (Italian Parliament) |  | needs_audit | eval_only | https://www.evalita.it/campaigns/evalita-2011/tasks/asr/ | Official Italian ASR benchmark with useful task data, but current reuse/access packaging needs audit before training use. |
| Italian Speech Data by Mobile Phone - 1,441 Hours (ELRA-S0450) |  | licensed | licensed_option | https://catalog.elra.info/en-us/repository/browse/ELRA-S0450/ | Best obvious large licensed Italian scale-up option if the public-only lane proves too small. |
| Italian (Italy) scripted microphone (`ITA_ASR001`) |  | commercial | commercial_option | https://datasets.appen.com/product/ita_asr001/ | Off-the-shelf commercial add-on that is small but cleanly packaged. |
| 499 Hours Italian spontaneous dialogue telephony (Nexdata) |  | commercial | commercial_option | https://www.nexdata.ai/datasets/speechrecog/1232 | Commercial conversational and telephony option if more spontaneous Italian is needed. |

## Immediate Research Tasks

- confirm the Italian-only public total for `Common Voice 12`
- verify whether `ASR-ItaCSC` and `KIParla` are trainable under the current
  project policy
- identify whether `487h` is enough for a serious reproduction baseline
- estimate how much purchasable Italian speech exists if we want to scale past
  the NVIDIA public recipe

## Reproduction View

Italian is the easiest public-only reproduction candidate in the seed set. It
is also the cleanest language for testing the full manifest/tokenizer/training
pipeline end-to-end before attempting a bigger language.

Current judgment:

- Italian remains the best first public-only technical reproduction
- the main weakness is scale beyond the seed recipe, not seed reproducibility
- French is likely the better next step for a larger public-only expansion
- if Italian needs materially more hours, the best additions may be licensed or
  commercial rather than purely public

## Open Questions

- Is Italian better as the first technical reproduction even if Russian is the
  more interesting language?
- What open Italian data exists beyond the three-card recipe that is both large
  enough and clean enough to train on?
- Is there a viable public Italian extension path, or should Italian be treated
  as a small clean reproduction target only?
