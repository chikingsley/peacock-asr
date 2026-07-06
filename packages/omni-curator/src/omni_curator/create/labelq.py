"""Label stage: the I/O consumer half of the split create pipeline.

One process. A single dispatcher owns the queue DB; a thread pool of ``workers`` does only the
network work (Scribe ensemble -> consensus), so SQLite stays single-writer while Scribe runs at
the target concurrency (~200-250, the service is I/O-bound). Each thread keeps its own
``SuperwhisperClient`` + Scribe functions (thread-local — the clients aren't assumed thread-safe).

Loop: reclaim expired leases -> batch-claim pending clips under a fresh ``claim_token`` -> label in
the pool -> write results back guarded by that token (a reclaimed clip's late result can't land).
Idles when the queue is empty so it keeps draining as the segmenter feeds it.

Failure policy: the deployed service owns ASR key rotation, so there is no in-process key
renewal here. A per-clip error requeues the clip (attempts-capped, via ``fail_clips``); a long
streak of consecutive failed clips aborts the run so a service outage can't become a request
spray. See ``docs/CURATION_FACTORY.md`` for the current stage contracts.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

import httpx

from omni_curator.create.queue import QueueStore
from omni_curator.create.transcribe import DEFAULT_LANGS
from omni_curator.scribe.concurrency import read_window

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from omni_curator.create.queue import QClip
    from omni_curator.create.transcribe import ScribeFn
    from omni_curator.scribe.swservice import SuperwhisperClient


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
        from omni_curator.scribe.swservice import SuperwhisperClient

        _state.client = SuperwhisperClient()
    return _state.client


def _label_clip(
    clip: QClip, *, langs: tuple[str, ...], runs: int, instruction: str | None
) -> tuple[str, str, str]:
    """Label one clip (Scribe ensemble -> consensus). Returns ``(clip_id, label, variants)``."""
    from pathlib import Path

    from omni_curator.create.fuse import consensus_fuse
    from omni_curator.create.transcribe import transcribe_clip

    variants = transcribe_clip(Path(clip.clip_path), _scribe_fns(langs), runs=runs)
    label = (
        consensus_fuse(
            variants, language=clip.language, script=clip.script,
            client=_client(), instruction=instruction,
        )
        if variants
        else ""
    )
    return clip.clip_id, label, json.dumps(variants, ensure_ascii=False)


class _Breaker:
    """Failure accounting for the dispatcher — rides through blips, aborts a persistent outage.

    Tracks consecutive failed clips (reset by any success). When the streak hits ``threshold`` (a
    service blip — intermittent 500/502), ``record`` backs off (exponential, capped) and retries
    instead of aborting, so the run rides through. But if the streak keeps tripping the threshold
    across ``max_pauses`` consecutive pauses with NO success in between, the service is genuinely
    down and ``record`` aborts (raises) rather than spinning forever. Any success resets the streak,
    the backoff, and the pause count. The service owns ASR key rotation.
    """

    def __init__(
        self,
        threshold: int,
        *,
        notify: Callable[[str], None] | None = None,
        max_pauses: int = 5,
        base_backoff: float = 15.0,
    ) -> None:
        self.threshold = threshold
        self.consecutive = 0
        self.errors: Counter[str] = Counter()
        self._notify = notify or (lambda _m: None)
        self._backoff = 0.0
        self._max_pauses = max_pauses
        self._base_backoff = base_backoff
        self._pauses = 0

    def record(
        self, done: int, errs: list[tuple[QClip, Exception]], labeled: int
    ) -> bool:
        """Account one batch; back off on a streak (returns True), abort a persistent outage."""
        if done:
            self.consecutive = 0
            self._backoff = 0.0
            self._pauses = 0  # progress resumed -> the outage passed
        self.consecutive += len(errs)
        for _, exc in errs:
            self.errors[f"{type(exc).__name__}: {exc}"[:160]] += 1
        if self.consecutive >= self.threshold:
            self._pause(labeled)
            return True
        return False

    def _pause(self, labeled: int) -> None:
        """Transient outage: back off and retry; abort once it persists across ``max_pauses``."""
        if self._pauses >= self._max_pauses:
            self._abort(labeled)
        self._pauses += 1
        self._backoff = min(self._backoff * 2 if self._backoff else self._base_backoff, 120.0)
        self._notify(
            f"labelq: {self.consecutive} consecutive failures "
            f"(pause {self._pauses}/{self._max_pauses}, labeled {labeled}); "
            f"backing off {self._backoff:.0f}s then retrying -- top: {self.errors.most_common(3)}"
        )
        time.sleep(self._backoff)
        self.consecutive = 0  # reset; failed clips already requeued for retry

    def _abort(self, labeled: int) -> None:
        msg = (
            f"labelq aborted: service down across {self._pauses} backoff pauses "
            f"({self.consecutive} consecutive failures); labeled {labeled}; "
            f"top failures: {self.errors.most_common(3)}"
        )
        raise RuntimeError(msg)


def _commit_outcome(queue: QueueStore, token: str, outcome: _BatchOutcome) -> int:
    """Persist one batch outcome under ``token``; return the number completed.

    Completes labeled clips; releases (un-charges) clips claimed-but-never-attempted; and splits
    failures by error type — transient service/transport errors are released (retried on recovery,
    no attempt burned), clip-specific faults (missing/corrupt audio) are failed so a genuinely bad
    clip is still retired at the attempt cap. Call this BEFORE the breaker may abort the run.
    """
    done = queue.complete_clips(token, outcome.done)
    if outcome.unprocessed:
        queue.release_clips(token, [c.clip_id for c in outcome.unprocessed])
    retry_ids, real_failures = _split_failures(outcome.failures)
    if retry_ids:
        queue.release_clips(token, retry_ids)
    for msg, clip_ids in _group_errors(real_failures).items():
        queue.fail_clips(token, clip_ids, msg)
    return done


def run_labeler(
    queue_path: Path,
    *,
    workers: int = 200,
    batch: int | None = None,
    window_file: Path | None = None,
    pool_max: int | None = None,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    runs: int = 1,
    instruction: str | None = None,
    lease_s: float = 900.0,
    poll_s: float = 5.0,
    idle_rounds: int = 3,
    breaker_threshold: int = 50,
    max_pauses: int = 5,
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
    pool_max = max(pool_max or workers, workers)  # pool ceiling; the window throttles below it
    labeled = 0
    empty = 0
    notify = on_event or (lambda _msg: None)
    breaker = _Breaker(breaker_threshold, notify=notify, max_pauses=max_pauses)
    worker = _make_worker(langs, runs, instruction)
    try:
        with ThreadPoolExecutor(max_workers=pool_max) as pool:
            while True:
                # Re-read the live window each round (GNU-parallel --jobs style): edit the file
                # and concurrency retunes on the next batch. `batch` (if set) pins the claim size.
                window = read_window(window_file, default=workers, cap=pool_max)
                claim_n = batch or window * 2
                queue.reclaim_stale_clips(lease_s)
                token = uuid4().hex
                clips = queue.claim_clips(claim_n, token, lease_s=lease_s)
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
                    pool, worker, clips,
                    window_file=window_file, default_window=workers, pool_max=pool_max,
                    fail_budget=budget,
                )
                # Commit the batch BEFORE breaker.record (which may raise to abort the run).
                labeled += _commit_outcome(queue, token, outcome)
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
    failures: list[tuple[QClip, Exception]]  # failures -> fail_clips, or release on an outage round
    unprocessed: list[QClip]  # claimed but never submitted (breaker tripped) -> release (un-charge)


def _process_batch(
    pool: ThreadPoolExecutor,
    worker: Callable[[QClip], tuple[QClip, tuple[str, str, str] | None, Exception | None]],
    clips: list[QClip],
    *,
    window_file: Path | None,
    default_window: int,
    pool_max: int,
    fail_budget: int,
) -> _BatchOutcome:
    """Stream a claimed batch through the pool, honoring the live window each submit.

    Re-reads the window file every iteration (GNU-parallel --jobs style), so lowering it mid-batch
    stops new submissions until in-flight drains back under target. Stops submitting once
    ``fail_budget`` consecutive failures accrue (the run breaker is about to trip): a uniform outage
    then charges an attempt to only the in-flight clips, and the leftover claimed clips are returned
    as ``unprocessed`` so the caller can release them (un-charge) rather than burn an attempt.
    """
    todo: deque[QClip] = deque(clips)
    futures: dict[Future[tuple[QClip, tuple[str, str, str] | None, Exception | None]], QClip] = {}
    done: list[tuple[str, str, str]] = []
    failures: list[tuple[QClip, Exception]] = []
    unprocessed: list[QClip] = []
    consecutive = 0  # consecutive failures within this batch; a success resets it
    blind = 0  # clips submitted before the first success (service still unproven)
    while todo or futures:
        window = read_window(window_file, default=default_window, cap=pool_max)
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
        if consecutive >= fail_budget and todo:
            unprocessed.extend(todo)  # breaker about to trip: stop claiming; release the leftovers
            todo.clear()
    return _BatchOutcome(done, failures, unprocessed)


def _group_errors(errs: list[tuple[QClip, Exception]]) -> dict[str, list[str]]:
    """Group failed clip ids by error message (one ``fail_clips`` write per distinct error)."""
    by_msg: dict[str, list[str]] = {}
    for clip, exc in errs:
        by_msg.setdefault(f"{type(exc).__name__}: {exc}", []).append(clip.clip_id)
    return by_msg


#: Substrings marking a failure as a transient service/transport problem (HTTP 5xx, gateway,
#: timeout, rate-limit, auth, mid-flight disconnects) rather than a fault in the clip itself.
_RETRYABLE_MARKERS = (
    "500", "502", "503", "504", "bad gateway", "service unavailable", "internal server error",
    "gateway timeout", "timeout", "timed out", "connection", "temporarily", "429", "rate limit",
    "401", "unauthorized", "403", "forbidden", "disconnect", "server disconnected",
    "remote end closed", "connection reset", "broken pipe", "eof occurred",
)
#: Exception types that always mean a transport problem (the clip is fine, retry later).
_TRANSIENT_EXC = (ConnectionError, TimeoutError, httpx.TransportError, httpx.TimeoutException)
_SERVER_ERROR_MIN = 500  # HTTP status >= this is server-side -> retryable
_RETRYABLE_CLIENT_CODES = (401, 403, 408, 429)  # auth + request-timeout + rate-limit -> retryable


def _is_retryable(exc: Exception) -> bool:
    """True if a clip failure is a service/transport problem -> release for a later retry (no
    attempt charged); False if it is clip-specific (missing/corrupt audio) -> fail so a genuinely
    bad clip is retired at the attempt cap. Unknown errors default to NOT retryable, so a bad clip
    can never loop forever being released — only known-transient errors get the free retry.
    """
    if isinstance(exc, _TRANSIENT_EXC):
        return True  # transport problem (connect/read/disconnect/protocol) -> the clip is fine
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= _SERVER_ERROR_MIN or code in _RETRYABLE_CLIENT_CODES
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return False  # the audio file/path is the problem, not the service
    return any(marker in str(exc).lower() for marker in _RETRYABLE_MARKERS)


def _split_failures(
    failures: list[tuple[QClip, Exception]],
) -> tuple[list[str], list[tuple[QClip, Exception]]]:
    """Partition failures into (retryable clip ids -> release, clip-specific failures -> fail)."""
    retry_ids: list[str] = []
    real: list[tuple[QClip, Exception]] = []
    for clip, exc in failures:
        (retry_ids.append(clip.clip_id) if _is_retryable(exc) else real.append((clip, exc)))
    return retry_ids, real


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
