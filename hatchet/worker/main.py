"""Local Hatchet worker entrypoint."""

from __future__ import annotations

from hatchet.client import hatchet
from hatchet.workflows.p001_xlsr53 import p001_xlsr53_workflow
from hatchet.workflows.p003 import p003_compact_backbones_workflow


def create_worker():
    """Create a worker compatible with current Hatchet Python APIs."""
    try:
        return hatchet.worker(
            "peacock-asr-local",
            workflows=[p003_compact_backbones_workflow, p001_xlsr53_workflow],
            slots=4,
        )
    except TypeError:
        worker = hatchet.worker("peacock-asr-local", max_runs=4)
        worker.register_workflow(p003_compact_backbones_workflow)
        worker.register_workflow(p001_xlsr53_workflow)
        return worker


def main() -> None:
    worker = create_worker()
    worker.start()


if __name__ == "__main__":
    main()
