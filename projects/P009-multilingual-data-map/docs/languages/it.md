# Italian

## Target

Map the smallest released FastConformer language recipe that still looks
realistic to reproduce quickly.

Released seed model:

- `nvidia/stt_it_fastconformer_hybrid_large_pc`

## NVIDIA Seed Recipe

The Italian FastConformer hybrid card lists about `487h` total:

| Dataset | Hours listed on card | Access class | Notes |
|---|---:|---|---|
| Common Voice 12 (it) | `220h` | public_open | public Mozilla corpus |
| MLS Italian | `214h` | public_open | public multilingual LibriSpeech branch |
| VoxPopuli Italian | `53h` | public_open | public European Parliament speech |

Tokenizer note from the card:

- SentencePiece BPE
- vocab size `512`

## Immediate Research Tasks

- verify whether there are obvious additional open Italian corpora
- identify whether `487h` is enough for a serious reproduction baseline
- estimate how much purchasable Italian speech exists if we want to scale past
  the NVIDIA public recipe

## Reproduction View

Italian is the easiest public-only reproduction candidate in the seed set. It
is also the cleanest language for testing the full manifest/tokenizer/training
pipeline end-to-end before attempting a bigger language.

## Open Questions

- Is Italian better as the first technical reproduction even if Russian is the
  more interesting language?
- What open Italian data exists beyond the three-card recipe?
