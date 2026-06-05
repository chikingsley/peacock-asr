"""Label stage: the I/O consumer half of the split create pipeline.

One process. A single dispatcher owns the queue DB; a thread pool of ``workers`` does only the
network work (Scribe ensemble -> compile-down), so SQLite stays single-writer while Scribe runs at
the target concurrency (~200-250, the free API is I/O-bound). Each thread keeps its own
``SuperwhisperClient`` + Scribe functions (thread-local — the clients aren't assumed thread-safe).

Loop: reclaim expired leases -> batch-claim pending clips under a fresh ``claim_token`` -> label in
the pool -> write results back guarded by that token (a reclaimed clip's late result can't land).
Idles when the queue is empty so it keeps draining as the segmenter feeds it.

Failure policy (the run must be impossible to turn into a request spray): a per-clip error
requeues the clip (attempts-capped, via ``fail_clips``); an auth failure (dead key) renews the
key through the Superwhisper proxy and rebuilds every thread's Scribe fns; consecutive renewals
with no successful batch in between, or a long streak of failed clips, abort the run. See
``docs/PIPELINE_SPLIT.md``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from omni_curator.create.queue import QueueStore
from omni_curator.create.transcribe import DEFAULT_LANGS, ScribeError, renew_scribe_key

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from omni_curator.create.queue import QClip

_local = threading.local()
_fns_generation = 0  # bumped on key renewal so every pool thread rebuilds its Scribe fns


def _bump_fns_generation() -> None:
    global _fns_generation  # noqa: PLW0603 — the renewal signal read by all pool threads
    _fns_generation += 1


def _scribe_fns(langs: tuple[str, ...]) -> dict[str, Any]:
    if getattr(_local, "fns_generation", None) != _fns_generation:
        from omni_curator.create.transcribe import default_key, make_scribe_fns

        _local.scribe_fns = make_scribe_fns(default_key(), langs)
        _local.fns_generation = _fns_generation
    return _local.scribe_fns


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


class _Breaker:
    """Failure accounting for the dispatcher — aborts before failures become a request spray.

    Tracks consecutive failed clips (reset by any success) and back-to-back key renewals
    (reset by any successful batch). ``record`` returns ``True`` when the caller should renew
    the key; it raises ``RuntimeError`` when the run must stop.
    """

    def __init__(self, threshold: int, max_renewals: int) -> None:
        self.threshold = threshold
        self.max_renewals = max_renewals
        self.consecutive = 0
        self.renewals = 0
        self.errors: Counter[str] = Counter()

    def record(
        self, done: int, errs: list[tuple[QClip, Exception]], labeled: int
    ) -> bool:
        """Account one batch outcome; ``True`` -> renew the key now."""
        if done:
            self.consecutive = 0
            self.renewals = 0
        if not errs:
            return False
        self.consecutive += len(errs)
        for _, exc in errs:
            self.errors[f"{type(exc).__name__}: {exc}"[:160]] += 1
        if any(isinstance(exc, ScribeError) and exc.auth for _, exc in errs):
            if self.renewals >= self.max_renewals:
                self._abort(f"key renewal exhausted ({self.renewals}x)", labeled)
            self.renewals += 1
            self.consecutive = 0
            return True
        if self.consecutive >= self.threshold:
            self._abort(f"{self.consecutive} consecutive failures", labeled)
        return False

    def _abort(self, reason: str, labeled: int) -> None:
        msg = (
            f"labelq aborted: {reason}; labeled {labeled}; "
            f"top failures: {self.errors.most_common(3)}"
        )
        raise RuntimeError(msg)


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
    breaker_threshold: int = 50,
    max_renewals: int = 3,
    on_progress: Callable[[int], None] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> int:
    """Drain the clip queue with ``workers`` concurrent Scribe calls. Returns clips labeled.

    Stops after ``idle_rounds`` consecutive empty polls (the segmenter has drained); raise it / set
    it huge to run as a long-lived service alongside a still-feeding segmenter.

    Failed clips are requeued via ``fail_clips`` (attempts-capped). An auth failure renews the key
    and rebuilds the pool's Scribe fns; ``max_renewals`` back-to-back renewals with no successful
    batch between them, or ``breaker_threshold`` consecutive failed clips, abort the run. The queue
    loses nothing on abort — failed clips are already requeued, claimed ones get lease-reclaimed.
    ``on_event`` receives operator messages (renewals, the final failure summary).
    """
    queue = QueueStore(queue_path)
    batch = batch or workers * 2
    labeled = 0
    empty = 0
    breaker = _Breaker(breaker_threshold, max_renewals)
    notify = on_event or (lambda _msg: None)
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
                outcomes = list(pool.map(worker, clips))
                done = [result for _, result, exc in outcomes if exc is None and result]
                errs = [(clip, exc) for clip, _, exc in outcomes if exc is not None]
                labeled += queue.complete_clips(token, done)
                for msg, clip_ids in _group_errors(errs).items():
                    queue.fail_clips(token, clip_ids, msg)
                if breaker.record(len(done), errs, labeled):
                    notify(f"auth failure — renewing key ({breaker.renewals})")
                    renew_scribe_key()
                    _bump_fns_generation()
                if on_progress:
                    on_progress(labeled)
    finally:
        queue.close()
    if breaker.errors:
        notify(
            f"labelq failures ({sum(breaker.errors.values())}): {breaker.errors.most_common(5)}"
        )
    return labeled


def _group_errors(errs: list[tuple[QClip, Exception]]) -> dict[str, list[str]]:
    """Group failed clip ids by error message (one ``fail_clips`` write per distinct error)."""
    by_msg: dict[str, list[str]] = {}
    for clip, exc in errs:
        by_msg.setdefault(f"{type(exc).__name__}: {exc}", []).append(clip.clip_id)
    return by_msg


def _make_worker(
    langs: tuple[str, ...], runs: int, instruction: str | None
) -> Callable[[QClip], tuple[QClip, tuple[str, str, str] | None, Exception | None]]:
    """Wrap ``_label_clip`` so the pool returns outcomes, never raises across the pool boundary."""

    def worker(clip: QClip) -> tuple[QClip, tuple[str, str, str] | None, Exception | None]:
        try:
            return clip, _label_clip(clip, langs=langs, runs=runs, instruction=instruction), None
        except Exception as exc:  # noqa: BLE001 — dispatcher classifies and requeues
            return clip, None, exc

    return worker
