"""Mirror Mozilla Data Collective Common Voice archives to a Hugging Face parquet dataset.

One pipeline: download MDC archives, convert each to appendable HF parquet shards, upload the
shards, verify them on the Hub, then delete the local archive. See :mod:`cv26.pipeline`.
"""
