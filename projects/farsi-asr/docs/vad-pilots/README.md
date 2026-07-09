# Farsi VAD pilots

`farsi-clean-noisy-32.jsonl` is the frozen selector for the first shared-policy comparison.

- Clean: 16 `avas_book_club` audiobook files, four lexical IDs from each duration stratum
  (1-3, 3-8, 8-20, and 20-45 minutes).
- Noisy: 16 `iran_international` news files with historical Scribe-positive spans, greedily
  duration-matched to the clean files with lexical tie-breaking.
- All sources were verified as readable 16 kHz FLACs before selection.
- Clean duration: 11,550.848 seconds. Noisy duration: 11,554.698 seconds. Difference: 0.03%.

The selector is configuration and stays in Git. Generated clips and reports stay out of Git under
an explicit scratch root. On the originating host, the checksum/device-corrected interval report is
`/mnt/workerssd-2t/peacock-asr/pilots/farsi-vad-2026-07-09-v2`; retained clips, the historical
anchor analysis, and the failed Scribe sample are in the sibling directory without the `-v2`
suffix. The two runs produced byte-for-byte identical raw and emitted interval arrays for all 96
source-engine pairs.

The historical Scribe intervals are pseudo-reference speech-positive anchors from the retired
segmentation policy. They are useful for detecting gross coverage loss, but they are biased toward
the old MarbleNet boundaries and are not human VAD truth.
