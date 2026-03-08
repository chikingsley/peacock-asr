# P001 Paper Workspace

Working title:
- **CTC Phoneme Posteriors for Segmentation-Free Pronunciation Assessment**

Purpose:
- Convert P001 from engineering narrative into a paper-grade manuscript.
- Keep every claim traceable to local evidence.

Source of truth:
- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Historical narrative: `../../../docs/research/archived/05_PHONEME_HEADS.md`

Draft files:
- `manuscript.md` (primary writing file)
- `PAPER_CLOSE_CHECKLIST.md` (W&B-first closeout checklist)

Citation convention:
- Use Pandoc/Quarto citekeys, e.g. `[@cao2026segmentation_free_gop]`.
- All citekeys are in `./refs.bib`.

Process:
1. Freeze Methods and dataset/eval protocol.
2. Lock experiment table schema and report all runs in one format.
3. Write Results only from reproducible logs/artifacts.
4. Run evidence audit before finalizing claims.

Operational pattern:
- Prepare/posterior caching and `k2` topology prewarm happen before the real
  eval run when using the `k2` scalar backend.
- Prewarm command:

```bash
uv run --project projects/P001-gop-baselines python -m p001_gop.cli \
  prewarm-k2 \
  --backend xlsr-espeak \
  --split both
```
