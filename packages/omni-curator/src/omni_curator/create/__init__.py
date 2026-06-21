"""create: generate labels for raw, untranscribed audio (YouTube, podcasts, shows).

For sources whose audio has NO transcript, this synthesizes the labels through a split,
queue-driven pipeline (the counterpart is :mod:`omni_curator.ingest`, which pulls datasets that
ALREADY have transcripts). One module per stage, in pipeline order:

- :mod:`~omni_curator.create.youtube`    — source: download channel audio (16 kHz FLAC).
- :mod:`~omni_curator.create.queue`      — the SQLite work queue decoupling the stages.
- :mod:`~omni_curator.create.vad`        — segment: the NeMo frame-VAD model.
- :mod:`~omni_curator.create.segment`    — segment: claim a video -> VAD -> cut clips (CPU).
- :mod:`~omni_curator.create.transcribe` — label: the Scribe ensemble (+ key renewal).
- :mod:`~omni_curator.create.fuse`       — label: LLM consensus fuse / transliteration.
- :mod:`~omni_curator.create.labelq`     — label: the dispatcher + HTTP worker pool (I/O).

Harvest (folding labeled queue clips into per-channel ``CuratorStore``s) lives in the
per-language project CLI (e.g. ``tajik-curate enqueue|segment|labelq|harvest``), which drives
these stages directly; there is nothing to re-export here.
"""
