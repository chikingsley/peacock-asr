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

## Blinded MarbleNet versus Silero review

The 2026-07-09 review is prepared under
`/mnt/workerssd-2t/peacock-asr/reviews/farsi-vad-marble-silero-2026-07-09`. It contains 160 unique
disagreement regions: 40 each for MarbleNet-only clean, MarbleNet-only noisy, Silero-only clean,
and Silero-only noisy. All 32 pilot sources are represented. Each clip has one second of context
around two tones that mark the disputed span; duration strata within every cell prevent the result
from being dominated by tiny boundary slivers.

The browser is intentionally blinded. Key `1` means usable speech between the tones, `2` means
non-speech, `3` means speech with a clipped boundary, and `4` means unsure. Space replays, `0` skips,
and Left Arrow revisits the previous item. Votes persist in `review.sqlite`; the generated JSON/CSV
exports join those votes back to the hidden engine direction for analysis. Cobra is not in this
review because the common-profile pilot already excluded it and supplied effectively no noisy
Cobra-only disagreement pool.
