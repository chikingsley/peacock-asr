# Language Matrix

This is the top-level summary for the seeded FastConformer languages.

## Current Seed Set

| Language | NVIDIA released model | Model family | Hours listed on card | Seed recipe type | Immediate note |
|---|---|---|---:|---|---|
| Russian | `nvidia/stt_ru_fastconformer_hybrid_large_pc` | FastConformer hybrid RNNT+CTC | `1840h` | mostly public/open, license audit still needed | best first non-English branch |
| Spanish | `nvidia/stt_es_fastconformer_hybrid_large_pc` | FastConformer hybrid RNNT+CTC | `1424h` | mixed public + licensed | includes Fisher, so not public-only |
| Italian | `nvidia/stt_it_fastconformer_hybrid_large_pc` | FastConformer hybrid RNNT+CTC | `487h` | public/open | smallest realistic reproduction target |
| French | `nvidia/stt_fr_fastconformer_hybrid_large_pc` | FastConformer hybrid RNNT+CTC | `1800h` | public/open | strong Western Europe public-data lane |

## Working Assumptions

- The hours above are the hours NVIDIA says were used in the released recipe,
  not necessarily the full public size of each dataset.
- "public/open" means the dataset appears to be publicly obtainable; exact
  download mechanics, license terms, and cleanup filters still need audit.
- "licensed" means paid or access-restricted data is already in the public
  NVIDIA recipe, so a public-only reproduction must change that recipe.

## Current Priority

1. Russian
2. Spanish
3. Italian
4. French
