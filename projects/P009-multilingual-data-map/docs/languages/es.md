# Spanish

## Target

Map the Spanish FastConformer recipe and separate the public-only path from the
licensed path.

Released seed model:

- `nvidia/stt_es_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The Spanish FastConformer hybrid card lists about `1424h` total:

| Dataset | Hours listed on card | Access class | Notes |
|---|---:|---|---|
| Fisher Spanish | `141h` | licensed | likely LDC-style restricted corpus |
| Common Voice 12 (es) | `395h` | public_open | Mozilla public corpus |
| MLS Spanish | `780h` | public_open | public multilingual LibriSpeech branch |
| VoxPopuli Spanish | `108h` | public_open | public European Parliament speech |

## Immediate Research Tasks

- confirm exact licensing path for Fisher Spanish
- compute a public-only reproduction plan without Fisher
- identify other public Spanish corpora that can replace Fisher
- identify purchasable Spanish speech sources if a closer reproduction is needed

## Reproduction View

Spanish already has a clean split between open and licensed data. That makes it
a good language for two parallel plans:

- a public-only reproduction
- a closer licensed reproduction

## Open Questions

- What is the best public replacement for Fisher Spanish in this recipe?
- Which Spanish commercial corpora are realistic to buy at lab scale?
