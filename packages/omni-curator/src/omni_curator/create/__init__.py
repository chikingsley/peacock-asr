"""create: generate labels for raw, untranscribed audio (YouTube, podcasts, shows).

For sources whose audio has NO transcript, this synthesizes the labels through a split,
queue-driven pipeline (the counterpart is :mod:`omni_curator.ingest`, which pulls datasets that
ALREADY have transcripts):

- :mod:`omni_curator.create.segment` — VAD-segment a video into clips (CPU producer).
- :mod:`omni_curator.create.labelq` — Scribe-ensemble + compile-down each clip (I/O consumer).
- :mod:`omni_curator.create.queue` — the SQLite work queue decoupling the two.

These are driven directly from a per-language project CLI (e.g. ``tajik-curate
enqueue|segment|labelq|harvest``); there is nothing to re-export here.
"""
