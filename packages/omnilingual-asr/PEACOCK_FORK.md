# Peacock fork provenance

This directory contains Peacock's vendored Omnilingual ASR package. Peacock keeps it in-repository because the ASR projects require metadata that differs from the upstream release while sharing one editable package source.

- Upstream: <https://github.com/facebookresearch/omnilingual-asr>
- Upstream base: `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`
- Imported Peacock fork: <https://github.com/chikingsley/omnilingual-asr> at `7d289be12006679d3d322d5208393f143884946f`
- Peacock metadata commit: `1d88b03774c20ff3647010519fb97fdf131666c8`

The Peacock metadata constrains the package to CPython 3.12, raises the `fairseq2` requirement to `>=0.8.1`, and identifies the build as `0.2.0+peacock.1`. The upstream BSD-style license remains in [`LICENSE`](LICENSE).

Refreshes should start from the recorded upstream repository, retain the Peacock metadata intentionally, and update the commits above in the same change.
