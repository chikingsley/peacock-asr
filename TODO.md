# TODO

## Research Infrastructure

- [ ] Build paper management system (Zotero replacement)
  - Postgres backend with pgvector embeddings + tsvector full-text search
  - Store papers as markdown with structured metadata (authors, year, venue, abstract)
  - Citation graph / relationship tracking between papers
  - Quick search across all stored papers
  - BibTeX import/export

- [ ] Organize research citations for the project
  - Collect all papers we reference into the system
  - Build citation list in proper format
  - Map the citation graph (who cites who, shared authors)

## Research Paper Shell

- [ ] Draft research paper outline
  - Contribution: first evaluation of ZIPA for GOP-based pronunciation scoring
  - Literature review: GOP methods, universal phone recognizers, pronunciation assessment
  - Experimental setup: three backends on SpeechOcean762
  - Results: PCC comparison table (original, xlsr-espeak, zipa)
  - Analysis: ZIPA character-level vocab incompatibility (32/39 phones)
  - Discussion: implications for multilingual pronunciation assessment

## Key Papers to Process

- [x] ZIPA (2505.23170) — saved to docs/papers/
- [x] POWSM (2510.24992) — saved to docs/papers/
- [x] PRiSM (2601.14046) — saved to docs/papers/
- [x] CTC-based-GOP (2507.16838v3) — saved to docs/papers/
- [ ] Allosaurus (ICASSP 2020) — universal phone recognizer, predecessor to ZIPA
- [ ] CLAP-IPA (NAACL 2024) — Jian Zhu's contrastive speech-phoneme embeddings
- [ ] CharsiuG2P — multilingual G2P, used to build IPAPack++
- [ ] Enhancing GOP with Phonological Knowledge (2506.02080)
- [ ] Original GOP paper (Witt & Young 2000)
- [ ] GOPT Transformer paper — the model we're building toward
- [ ] SpeechOcean762 dataset paper
- [ ] wav2vec2-xlsr-53 paper (Conneau et al.)

## Next Steps (After Benchmark Run 2)

- [ ] Collect ZIPA run 2 results (32/39 phones, ER/G fixes)
- [ ] Compare all three backends side-by-side
- [ ] Document findings in DECISIONS.md
- [ ] Plan next phase: fine-tuning a phoneme-level CTC head on ZIPA
  - This would resolve the 7 unmappable diphthongs/affricates
  - Small training job (only the final layer, ~39 output classes)
  - Need to decide: train on SpeechOcean762 or LibriSpeech?
