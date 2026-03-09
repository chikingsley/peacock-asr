# Language Matrix

This is the top-level summary for the seeded FastConformer languages and the
most plausible next step for each one.

## Current Seed Set

| Language | NVIDIA released model | Hours listed on card | Seed recipe type | Public-only reproduction status | Best scale path |
|---|---|---:|---|---|---|
| Russian | `nvidia/stt_ru_fastconformer_hybrid_large_pc` | `1840h` | mostly public/open, seed licenses still need audit | plausible now, but blocked by `Golos` / `Dusha` / `SOVA` provenance cleanup | audit seed stack, then expand with newer Common Voice and other audited Russian public corpora |
| Spanish | `nvidia/stt_es_fastconformer_hybrid_large_pc` | `1424h` | mixed public + licensed | viable as a public-only branch, but not as a literal recipe match because of Fisher | public lane: full MLS + full VoxPopuli + selected open supplements; closest match lane: add Fisher or other licensed telephony data |
| Italian | `nvidia/stt_it_fastconformer_hybrid_large_pc` | `487h` | public/open | best first technical public-only reproduction | reproduce seed recipe first, then add small audited supplements or commercial speech if more hours are needed |
| French | `nvidia/stt_fr_fastconformer_hybrid_large_pc` | `1800h` | public/open | cleanest high-hours public-first branch | reproduce `CV12 + MLS + VoxPopuli`, then add audited public French corpora or ELRA/Appen sources for domain balance |

## Working Assumptions

- The hours above are the hours NVIDIA says were used in the released recipe,
  not necessarily the full public size of each dataset.
- `public_open` means the dataset appears to be publicly obtainable; exact
  license obligations, filtering rules, and release constraints may still need
  audit.
- `licensed` means paid or access-restricted data is part of the recipe or is a
  likely extension source.
- The language notes and draft TSVs should carry the operational details; this
  matrix is only a decision summary.

## Suggested Execution Order

1. Italian for first end-to-end public technical reproduction
2. Russian for first high-value non-English audit and seed reconstruction
3. French for first large public-only scale-up lane
4. Spanish in two branches: public-only and closer licensed reproduction
