"""Label stage: the I/O consumer half of the split create pipeline.

One process. A single dispatcher owns the queue DB; a thread pool of ``workers`` does only the
network work (Scribe ensemble -> compile-down), so SQLite stays single-writer while Scribe runs at
the target concurrency (~200-250, the free API is I/O-bound). Each thread keeps its own
``SuperwhisperClient`` + Scribe functions (thread-local — the clients aren't assumed thread-safe).

Loop: reclaim expired leases -> batch-claim pending clips under a fresh ``claim_token`` -> label in
the pool -> write results back guarded by that token (a reclaimed clip's late result can't land).
Idles when the queue is empty so it keeps draining as the segmenter feeds it. See
``docs/PIPELINE_SPLIT.md``.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from omni_curator.create.queue import QueueStore
from omni_curator.create.transcribe import DEFAULT_LANGS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from omni_curator.create.queue import QClip

_local = threading.local()


def _scribe_fns(langs: tuple[str, ...]) -> dict[str, Any]:
    fns = getattr(_local, "scribe_fns", None)
    if fns is None:
        from omni_curator.create.transcribe import default_key, make_scribe_fns

        fns = _local.scribe_fns = make_scribe_fns(default_key(), langs)
    return fns


def _client() -> Any:
    client = getattr(_local, "client", None)
    if client is None:
        from superwhisper_api.text.client import SuperwhisperClient

        client = _local.client = SuperwhisperClient()
    return client


def _label_clip(
    clip: QClip, *, langs: tuple[str, ...], runs: int, instruction: str | None
) -> tuple[str, str, str]:
    """Label one clip (Scribe ensemble -> compile-down). Returns ``(clip_id, label, variants)``."""
    from pathlib import Path

    from omni_curator.create.fuse import compile_down
    from omni_curator.create.transcribe import transcribe_clip

    variants = transcribe_clip(Path(clip.clip_path), _scribe_fns(langs), runs=runs)
    label = (
        compile_down(
            variants, language=clip.language, script=clip.script,
            client=_client(), instruction=instruction,
        )
        if variants
        else ""
    )
    return clip.clip_id, label, json.dumps(variants, ensure_ascii=False)


def run_labeler(
    queue_path: Path,
    *,
    workers: int = 200,
    batch: int | None = None,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    runs: int = 1,
    instruction: str | None = None,
    lease_s: float = 900.0,
    poll_s: float = 5.0,
    idle_rounds: int = 3,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Drain the clip queue with ``workers`` concurrent Scribe calls. Returns clips labeled.

    Stops after ``idle_rounds`` consecutive empty polls (the segmenter has drained); raise it / set
    it huge to run as a long-lived service alongside a still-feeding segmenter.
    """
    queue = QueueStore(queue_path)
    batch = batch or workers * 2
    labeled = 0
    empty = 0
    worker = _make_worker(langs, runs, instruction)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                queue.reclaim_stale_clips(lease_s)
                token = uuid4().hex
                clips = queue.claim_clips(batch, token, lease_s=lease_s)
                if not clips:
                    empty += 1
                    if empty >= idle_rounds:
                        break
                    time.sleep(poll_s)
                    continue
                empty = 0
                results = list(pool.map(worker, clips))
                written = queue.complete_clips(token, results)
                labeled += written
                if on_progress:
                    on_progress(labeled)
    finally:
        queue.close()
    return labeled


def _make_worker(
    langs: tuple[str, ...], runs: int, instruction: str | None
) -> Callable[[QClip], tuple[str, str, str]]:
    def worker(clip: QClip) -> tuple[str, str, str]:
        return _label_clip(clip, langs=langs, runs=runs, instruction=instruction)

    return worker
