# Russian

## Target

Map the data path for reproducing and then extending the released Russian
FastConformer line.

Released seed model:

- `nvidia/stt_ru_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The Russian FastConformer hybrid card lists about `1840h` total:

| Dataset | Hours listed on card | Access class | Notes |
|---|---:|---|---|
| Golos | `1200h` | needs_audit | likely major seed corpus in this recipe |
| SOVA | `310h` | needs_audit | verify official hosting and license |
| Dusha | `200h` | needs_audit | verify official hosting and license |
| Russian LibriSpeech (RuLS) | `92.5h` | needs_audit | likely open, still audit |
| Common Voice 12 (ru) | `36.7h` | public_open | Mozilla public corpus |

## Immediate Research Tasks

- verify official source for Golos
- verify official source for SOVA
- verify official source for Dusha
- verify official source for RuLS
- estimate public-only reproducibility of the full `1840h` recipe
- identify additional open Russian corpora beyond the NVIDIA seed stack

## Reproduction View

Russian is a strong first target because the released model already exists and
the recipe looks mostly public/openly reachable. The main risk is license and
cleanup ambiguity around the Russian-specific corpora.

## Open Questions

- Which Russian datasets in the card are truly open vs registration-gated?
- What additional Russian ASR corpora are good enough to push beyond the
  released `115M` recipe?
- Is there enough public Russian speech to justify a serious XL (`~600M`) run?
