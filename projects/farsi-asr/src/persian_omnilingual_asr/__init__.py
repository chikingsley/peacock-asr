"""Persian Omnilingual ASR: fine-tuning, asset registration, and evaluation.

This package is the Omni-CTC training/eval layer for the Persian ASR project, built
on the shared :mod:`omni_finetune_core` package (the same pattern as
``tajik_omnilingual_asr``). It replaces the bespoke ``finetune_omni`` training path
that vendored Meta's recipe and shelled into it via ``runpy``.

Data preparation still lives in the sibling ``persian_asr_dataset`` package; this
package only consumes the parquet/manifests it produces.
"""

from __future__ import annotations
