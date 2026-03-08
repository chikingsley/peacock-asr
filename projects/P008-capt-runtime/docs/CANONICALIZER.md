# Canonicalizer

## Purpose

The canonicalizer is the text-to-pronunciation layer. It should answer:

- what words are in the lesson sentence?
- what is the canonical phone sequence for each word?
- where did that pronunciation come from?
- which words required fallback prediction?

The canonicalizer should be dictionary-first, not model-first.

## SQLite Lexicon Store

Yes: the dictionaries can just be imported into a local lexicon store.

That is the preferred shape for `P008`.

### Proposed tables

#### `lexicon_entries`

- `id INTEGER PRIMARY KEY`
- `language TEXT NOT NULL`
- `region TEXT`
- `word TEXT NOT NULL`
- `normalized_word TEXT NOT NULL`
- `phones_ipa TEXT NOT NULL`
- `stress TEXT`
- `source TEXT NOT NULL`
- `source_priority INTEGER NOT NULL`
- `source_format TEXT NOT NULL`
- `pos TEXT`
- `variant TEXT`
- `is_g2p INTEGER NOT NULL DEFAULT 0`
- `metadata_json TEXT`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

#### `lexicon_overrides`

- `id INTEGER PRIMARY KEY`
- `language TEXT NOT NULL`
- `region TEXT`
- `word TEXT NOT NULL`
- `phones_ipa TEXT NOT NULL`
- `stress TEXT`
- `reason TEXT`
- `created_at TEXT NOT NULL`

#### `g2p_cache`

- `id INTEGER PRIMARY KEY`
- `language TEXT NOT NULL`
- `word TEXT NOT NULL`
- `normalized_word TEXT NOT NULL`
- `phones_ipa TEXT NOT NULL`
- `stress TEXT`
- `provider TEXT NOT NULL`
- `provider_version TEXT`
- `confidence REAL`
- `metadata_json TEXT`
- `created_at TEXT NOT NULL`

### Lookup precedence

1. manual override
2. curated product lexicon
3. imported dictionary entry
4. cached G2P result
5. live G2P call

The key design rule is:

- G2P does less work because the lexicon does most of the work

## What Counts As OOV

`OOV` means `out-of-vocabulary`: a word that is not found in the dictionary.

Typical OOV cases:

- person names
- brand names
- slang
- contractions not present in the source lexicon
- borrowed words
- inflected variants
- typos or learner-generated variants

## Language Plan

### English

Primary dictionary:

- `CMUdict`

Secondary dictionary candidate:

- MFA `english_us_arpa`

Fallback G2P candidates:

- `g2p-en`
- `gruut-en`
- NeMo `ByT5` / `G2P-Conformer`
- `CharsiuG2P`

Why English is special:

- best open lexicon story
- best pedagogical stress information
- easiest compatibility with the current `P001` assumptions

### Spanish

Primary dictionary:

- MFA `spanish_mfa`
- MFA `spanish_latin_america_mfa`
- region-specific choice needed at import time

Fallback candidates:

- `gruut-es`
- NeMo `ByT5` / `G2P-Conformer`
- `CharsiuG2P`

Design note:

- Spanish orthography is relatively phonemic, so fallback pressure should be lower

### Italian

Primary pragmatic choice:

- `gruut-it`

Secondary source:

- MFA `italian_cv` dictionary resources where useful

Fallback candidates:

- `gruut-it`
- NeMo `ByT5` / `G2P-Conformer`
- `CharsiuG2P`

Design note:

- Italian is easy enough orthographically that a pragmatic G2P-first launch is acceptable

### French

Primary dictionary:

- MFA `french_mfa`
- MFA `french_prosodylab`

Fallback candidates:

- `gruut-fr`
- NeMo `ByT5` / `G2P-Conformer`
- `CharsiuG2P`

Design note:

- French orthography is less transparent than Spanish/Italian, so dictionary quality matters

### Russian

Primary dictionary:

- MFA `russian_mfa`

Fallback candidates:

- `gruut-ru`
- NeMo `ByT5` / `G2P-Conformer`
- `CharsiuG2P`

Design note:

- Russian stress and reduction make dictionary quality more important than naive rule systems

## Why `CharsiuG2P` Is In Scope

`CharsiuG2P` belongs in the bake-off as a real G2P candidate.

Important distinction:

- `Charsiu` (alignment / recognition family) and `CharsiuG2P` are not the same thing
- if we test `CharsiuG2P`, we test it as part of the canonicalizer, not the aligner

## Why We Still Need Dictionaries

Even with modern neural G2P:

- dictionaries are faster
- dictionaries are deterministic
- dictionaries are easier to override
- dictionaries are better for high-frequency words and pedagogically curated content
- dictionaries reduce G2P calls and make caching trivial

The canonicalizer should behave like a lexicon system with model assistance, not
like a pure neural service.

## Candidate Use Cases

### `g2p-en`

Use when:

- language is English
- we want an OOV fallback that stays close to the current ARPABET-centric stack

Do not use when:

- building the multilingual default path

### `gruut`

Use when:

- we want one pragmatic open-source fallback path across launch languages
- we care more about simple packaging and broad language support than about
  squeezing out the very last bit of model quality

This is the current best pragmatic launch candidate for Spanish, Italian,
French, and Russian fallback.

### NeMo `ByT5`

Use when:

- we want a neural multilingual fallback family
- we are willing to carry a heavier model dependency

Best role:

- later fallback experiment, not launch default

### NeMo `G2P-Conformer`

Use when:

- we want a neural per-language fallback path
- we are willing to manage language-specific models

Best role:

- later bake-off candidate for languages where the simpler fallback is weak

### `CharsiuG2P`

Use when:

- we want a modern neural G2P candidate in the bake-off
- we want to test whether it meaningfully beats the lighter launch fallback

Best role:

- explicit candidate in the bake-off, not assumed default

## Current Candidate Sources

- `CMUdict`: `https://github.com/cmusphinx/cmudict`
- `g2p-en`: `https://github.com/Kyubyong/g2p`
- `gruut`: `https://github.com/rhasspy/gruut`
- `CharsiuG2P`: `https://github.com/lingjzhu/CharsiuG2P`
- MFA model index: `https://mfa-models.readthedocs.io/en/latest/`
- NVIDIA NeMo G2P docs:
  `https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html`
