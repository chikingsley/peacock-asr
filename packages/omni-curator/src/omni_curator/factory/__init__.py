"""The omni-curator factory (#9): a polling supervisor that auto-drives the curate pipeline.

v0 scope: the supervisor core + the single-writer ``flock`` + the ``enqueue`` and ``segment``
stages, with a global 1-segment cap. See ``docs/factory_plan.md``. labelq/verify/harvest/merge/
archive, the Scribe balancer, and overflow backpressure are v1.
"""

from __future__ import annotations
