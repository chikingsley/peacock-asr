"""Label stage: the I/O consumer half of the split create pipeline.

One process. A single dispatcher owns the queue DB; a thread pool of ``workers`` does only the
network work (Scribe ensemble -> compile-down), so SQLite stays single-writer while Scribe runs at
the target concurrency (~200-250, the service is I/O-bound). Each thread keeps its own
``SuperwhisperClient`` + Scribe functions (thread-local — the clients aren't assumed thread-safe).

Loop: reclaim expired leases -> batch-claim pending clips under a fresh ``claim_token`` -> label in
the pool -> write results back guarded by that token (a reclaimed clip's late result can't land).
Idles when the queue is empty so it keeps draining as the segmenter feeds it.

Failure policy: the deployed service owns ASR key rotation, so there is no in-process key
renewal here. A per-clip error requeues the clip (attempts-capped, via ``fail_clips``); a long
streak of consecutive failed clips aborts the run so a service outage can't become a request
spray. See ``docs/PIPELINE_SPLIT.md``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

from omni_curator.create.queue import QueueStore
from omni_curator.create.transcribe import DEFAULT_LANGS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from omni_curator.create.queue import QClip
    from omni_curator.create.transcribe import ScribeFn
    from omni_curator.swservice import SuperwhisperClient


class _ThreadState(threading.local):
    """Per-thread Scribe fns + text client (the clients aren't assumed thread-safe)."""

    def __init__(self) -> None:
        self.scribe_fns: dict[str, ScribeFn] | None = None
        self.client: SuperwhisperClient | None = None


_state = _ThreadState()


def _scribe_fns(langs: tuple[str, ...]) -> dict[str, ScribeFn]:
    if _state.scribe_fns is None:
        from omni_curator.create.transcribe import make_scribe_fns

        _state.scribe_fns = make_scribe_fns(langs)
    return _state.scribe_fns


def _client() -> SuperwhisperClient:
    if _state.client is None:
        from omni_curator.swservice import SuperwhisperClient

        _state.client = SuperwhisperClient()
    return _state.client


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

    Tracks consecutive failed clips (reset by any success). ``record`` raises ``RuntimeError``
    when a long failure streak means the run must stop (e.g. the service is down). The service
    owns ASR key rotation, so there is no key-renewal path here.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.consecutive = 0
        self.errors: Counter[str] = Counter()

    def record(
        self, done: int, errs: list[tuple[QClip, Exception]], labeled: int
    ) -> None:
        """Account one batch outcome; abort the run on a long consecutive-failure streak."""
        if done:
            self.consecutive = 0
        self.consecutive += len(errs)
        for _, exc in errs:
            self.errors[f"{type(exc).__name__}: {exc}"[:160]] += 1
        if self.consecutive >= self.threshold:
            self._abort(f"{self.consecutive} consecutive failures", labeled)

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
    on_progress: Callable[[int], None] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> int:
    """Drain the clip queue with ``workers`` concurrent Scribe calls. Returns clips labeled.

    Stops after ``idle_rounds`` consecutive empty polls (the segmenter has drained); raise it / set
    it huge to run as a long-lived service alongside a still-feeding segmenter.

    Failure policy: clip failures requeue via ``fail_clips`` (attempts-capped). The deployed
    service owns ASR key rotation, so there is no in-process renewal — a persistent service
    outage simply trips the ``breaker_threshold`` consecutive-failure abort, bounding the spray.
    ``on_event`` receives operator messages (the final failure summary).
    """
    queue = QueueStore(queue_path)
    batch = batch or workers * 2
    labeled = 0
    empty = 0
    breaker = _Breaker(breaker_threshold)
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
                # Only this many more consecutive failures will trip the breaker; stop the batch
                # there instead of draining (and charging an attempt to) all `batch` clips.
                budget = max(1, breaker_threshold - breaker.consecutive)
                outcome = _process_batch(
                    pool, worker, clips, window=workers, fail_budget=budget
                )
                labeled += queue.complete_clips(token, outcome.done)
                for msg, clip_ids in _group_errors(outcome.failures).items():
                    queue.fail_clips(token, clip_ids, msg)
                breaker.record(len(outcome.done), outcome.failures, labeled)
                if on_progress:
                    on_progress(labeled)
    finally:
        queue.close()
    if breaker.errors:
        notify(
            f"labelq failures ({sum(breaker.errors.values())}): {breaker.errors.most_common(5)}"
        )
    return labeled


class _BatchOutcome(NamedTuple):
    """One claimed batch, fully accounted: every clip is in exactly one bucket."""

    done: list[tuple[str, str, str]]  # labeled (clip_id, label, variants) -> complete_clips
    failures: list[tuple[QClip, Exception]]  # failures -> fail_clips (attempt charged)


def _process_batch(
    pool: ThreadPoolExecutor,
    worker: Callable[[QClip], tuple[QClip, tuple[str, str, str] | None, Exception | None]],
    clips: list[QClip],
    *,
    window: int,
    fail_budget: int,
) -> _BatchOutcome:
    """Stream a claimed batch through the pool with a bounded submission window.

    Stops submitting new clips once ``fail_budget`` consecutive failures accrue (the run breaker is
    about to trip) — a uniform outage then charges an attempt to only the in-flight clips, not the
    whole batch. Clips never submitted stay claimed and are reclaimed after their lease, unburnt.
    """
    todo: deque[QClip] = deque(clips)
    futures: dict[Future[tuple[QClip, tuple[str, str, str] | None, Exception | None]], QClip] = {}
    done: list[tuple[str, str, str]] = []
    failures: list[tuple[QClip, Exception]] = []
    consecutive = 0  # consecutive failures within this batch; a success resets it
    blind = 0  # clips submitted before the first success (service still unproven)
    while todo or futures:
        while todo and len(futures) < window and consecutive < fail_budget:
            # Before any success, cap TOTAL submissions at the failure budget: a uniform outage
            # then charges ~fail_budget clips, not the whole batch. One success ramps to the full
            # window for throughput.
            if not done and blind >= fail_budget:
                break
            clip = todo.popleft()
            futures[pool.submit(worker, clip)] = clip
            if not done:
                blind += 1
        if not futures:
            break
        finished, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in finished:
            clip = futures.pop(future)
            _, result, exc = future.result()
            if exc is None and result is not None:
                done.append(result)
                consecutive = 0
            elif exc is not None:
                failures.append((clip, exc))
                consecutive += 1
        if consecutive >= fail_budget:
            todo.clear()  # breaker about to trip: stop claiming more, just drain in-flight
    return _BatchOutcome(done, failures)


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
