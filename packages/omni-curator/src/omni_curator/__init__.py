"""omni-curator: build ASR fine-tuning datasets from any source.

Two ways data comes in: :mod:`omni_curator.create` generates labels for raw, untranscribed audio
(YouTube etc.) through a split, queue-driven pipeline; :mod:`omni_curator.ingest` pulls datasets
that already have transcripts (FLEURS, Common Voice). Both yield
:class:`~omni_curator.sample.Sample`s, which flow through :mod:`~omni_curator.process`
(16 kHz / normalize / tokenizer-coverage) into :mod:`~omni_curator.store` (SQLite), scored by
:mod:`~omni_curator.benchmark` (WER/CER) and materialized by :mod:`~omni_curator.export`.
"""
