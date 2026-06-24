"""In-house, type-safe core for fine-tuning Meta's Omnilingual ASR (Omni CTC) models.

The heavy lifting (model architectures, dataset readers, and the fairseq2 recipe
framework) stays in Meta's code: ``fairseq2`` plus Peacock's editable fork of
``omnilingual-asr``. What lives here is *our* harness around them: typed (Pydantic v2)
training configs, asset-card registration, omni-parquet IO, tokenizer auditing,
evaluation/metrics, and a single owned copy of the wav2vec2-ASR recipe glue
(``omni_finetune_core.recipe``) that is otherwise only available in Meta's GitHub repo.
"""

from __future__ import annotations

__version__ = "0.1.0"
