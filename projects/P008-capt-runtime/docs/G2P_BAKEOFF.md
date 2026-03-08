# G2P Bake-Off

## Decision

Yes: do a bake-off.

But do not do a blind "everything against everything" benchmark.

The bake-off should answer one concrete product question:

- for each launch language, what canonicalizer stack gives acceptable coverage,
  acceptable output quality, and acceptable operational simplicity?

## Candidate Families

### Family A: dictionary-first

- `CMUdict`
- MFA dictionaries
- `gruut` lexicons

### Family B: lightweight packaged fallback

- `g2p-en`
- `gruut` G2P

### Family C: neural fallback

- NeMo `ByT5`
- NeMo `G2P-Conformer`
- `CharsiuG2P`

## What We Actually Compare

### English

Compare:

- `CMUdict` hit rate on target corpus
- `g2p-en`
- `gruut-en`
- NeMo `ByT5`
- NeMo `G2P-Conformer`
- `CharsiuG2P`

### Spanish

Compare:

- MFA Spanish dictionary hit rate
- `gruut-es`
- NeMo `ByT5`
- NeMo `G2P-Conformer`
- `CharsiuG2P`

### Italian

Compare:

- `gruut-it`
- NeMo `ByT5`
- NeMo `G2P-Conformer`
- `CharsiuG2P`

### French

Compare:

- MFA French dictionary hit rate
- `gruut-fr`
- NeMo `ByT5`
- NeMo `G2P-Conformer`
- `CharsiuG2P`

### Russian

Compare:

- MFA Russian dictionary hit rate
- `gruut-ru`
- NeMo `ByT5`
- NeMo `G2P-Conformer`
- `CharsiuG2P`

## Evaluation Sets

Per language:

- `lesson_corpus_in_vocab.tsv`
  - high-frequency lesson words
- `oov_words.tsv`
  - names, brands, slang, inflections, contractions, loanwords
- `hard_cases.tsv`
  - stress-sensitive or context-sensitive words

Special emphasis:

- English: heteronyms and stress
- French: spelling-to-pronunciation irregularity
- Russian: stress / reduction-sensitive cases

## Metrics

Each candidate gets scored on:

- dictionary coverage on lesson corpus
- phone error rate on hand-checked target list
- stress accuracy where applicable
- inventory compatibility with internal IPA
- latency
- model size / deployment complexity
- license / operational friction

## Acceptance Criteria

Minimum acceptance per language:

- dictionary coverage on lesson corpus: `>= 95%`
- fallback phone error rate on curated OOV set: `<= 10%`
- no phones outside our internal normalization map
- operational setup must be scriptable and cacheable

If a candidate fails these gates, it is not launch material.

## Why This Is Not A Pure Accuracy Contest

The best candidate is not necessarily the most accurate model on paper.

For runtime product work, we care about:

- stable outputs
- simple overrides
- license clarity
- fast local caching
- compatibility with the chosen scorer and aligner

So a slightly weaker but much more stable dictionary-first stack may beat a
neural model in practice.

## Output Artifact

The bake-off should produce one table per language with:

- primary dictionary
- best fallback
- known weak spots
- import/caching strategy
- launch recommendation
