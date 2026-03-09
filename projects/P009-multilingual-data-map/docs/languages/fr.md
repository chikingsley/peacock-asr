# French

## Target

Map the French FastConformer recipe and identify whether it is the best Western
Europe public-data branch after Spanish.

Released seed model:

- `nvidia/stt_fr_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The French FastConformer hybrid card lists about `1800h` total:

| Dataset | Hours listed on card | Access class | Notes |
|---|---:|---|---|
| Common Voice 12 (fr) | `710h` | public_open | public Mozilla corpus |
| MLS French | `925h` | public_open | public multilingual LibriSpeech branch |
| VoxPopuli French | `165h` | public_open | public European Parliament speech |

## Immediate Research Tasks

- identify additional open French corpora beyond the NVIDIA recipe
- identify purchasable French speech sources
- estimate whether French has enough public speech to justify scaling beyond the
  released `115M` model

## Reproduction View

French looks like a strong public-only lane with relatively large hours. It is
probably the cleanest "high-hours public Western Europe" FastConformer branch.

## Open Questions

- Is French actually easier to scale publicly than Spanish once Fisher is
  removed from the Spanish recipe?
