"""Store-level Scribe verification: score every clip's label against a fresh Scribe transcription.

The store is the master pool, so verification is a *uniform store-level step*: EVERY clip (ingested
or created) gets ONE Scribe pass and one full-jiwer score against its stored label. The two headline
rates land in the ``scribe_wer``/``scribe_cer`` columns (fast SQL filtering on export); the whole
jiwer breakdown (word/char S/D/I/H) and the Scribe hypothesis go in ``meta["scribe"]``. Export then
filters on those scores (:class:`omni_curator.export.Selection`'s ``max_scribe_wer``/
``max_scribe_cer``) — the Persian-style "scribe curation", made standard.

This is the *checking* pass, NOT the label-generating ensemble: a single Scribe pass (one reference)
is enough to ask "does the existing label match what Scribe hears?". ``scribe_language`` is normally
``None`` -> ``"auto"`` (Scribe detects / code-switches); forcing a language is possible but the
curator's FLORES codes (``kat_Geor``) are not Scribe ISO codes, so ``auto`` is the safe default.

Scribe is I/O-bound and free, so the run is parallelized across a large thread pool. Per-clip
failures (missing audio, a transient Scribe error) are counted and skipped — but failures are
never allowed to pile up: an auth failure (dead key) triggers an in-run key renewal, and a
circuit breaker aborts the whole run on consecutive failures. Spraying thousands of requests
at a dead key would burn the key source. ``verify_store`` is idempotent: it only scores rows
with ``scribe_wer IS NULL`` unless ``force``, so an aborted run resumes where it stopped.
"""

from __future__ import annotations

import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omni_curator.benchmark import score_pair
from omni_curator.create.transcribe import (
    ScribeError,
    default_key,
    make_scribe_fns,
    raise_for_scribe_error,
    renew_scribe_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore


@dataclass
class VerifyStats:
    """What a verification run produced: counts + the WER/CER distribution over newly-scored clips.

    ``scored`` is how many clips got a fresh score this run; ``skipped`` is how many already had one
    (idempotency — zero on a re-run means nothing new was scored); ``failed`` is per-clip Scribe /
    audio errors that were skipped (with ``(sample_id, message)`` recorded in ``failures``). The
    ``wer``/``cer`` summaries (mean/median/p90) are over the clips scored *this run*.
    """

    scored: int = 0
    skipped: int = 0
    failed: int = 0
    renewals: int = 0
    wer: dict[str, float] = field(default_factory=dict)
    cer: dict[str, float] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def top_failures(self, n: int = 5) -> list[tuple[str, int]]:
        """The ``n`` most common failure messages with counts — the at-a-glance 'what broke'."""
        return Counter(msg for _, msg in self.failures).most_common(n)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (``pct`` in ``[0, 100]``); ``0.0`` for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _distribution(values: list[float]) -> dict[str, float]:
    """``mean``/``median``/``p90`` of a list (all ``0.0`` when empty)."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 90.0),
    }


def _pending(store: CuratorStore, *, key: str | None, force: bool) -> list[Sample]:
    """Samples to score this run: filtered by ``key`` (source), un-scored unless ``force``."""
    samples = store.iter_samples(source=key) if key is not None else store.iter_samples()
    if force:
        return list(samples)
    return [s for s in samples if s.scribe_wer is None]


def _score_one(sample: Sample, scribe_fn: Any) -> dict[str, object]:
    """Run one Scribe pass over a clip and score its label against the hypothesis.

    Returns the ``meta["scribe"]`` detail dict: ``wer``/``cer``, the word/char S/D/I/H breakdown,
    the Scribe ``hypothesis`` text, plus the ``reference`` label. Raises on a Scribe / audio error
    (the superwhisper fn returns a ``Failure`` whose dict carries ``error`` but no ``transcript``) —
    the caller turns the raised error into a counted failure rather than scoring an empty string.
    """
    # The superwhisper process fn takes a Path, not a str.
    result = scribe_fn(Path(sample.audio_path)).as_dict()
    raise_for_scribe_error(result)
    if "transcript" not in result:
        raise RuntimeError("scribe returned no transcript")
    hypothesis = str(result.get("transcript") or "").strip()
    detail = score_pair(sample.text, hypothesis)
    detail["hypothesis"] = hypothesis
    detail["reference"] = sample.text
    return detail


def verify_store(
    store: CuratorStore,
    *,
    key: str | None = None,
    scribe_language: str | None = None,
    model: str = "scribe-v2",
    workers: int = 100,
    force: bool = False,
    breaker_threshold: int = 50,
    max_renewals: int = 3,
) -> VerifyStats:
    """Score every (un-scored) clip's label against a fresh Scribe pass; persist the full result.

    Iterates the store (optionally restricted to ``source == key``), and for each clip with no score
    yet (or all clips when ``force=True``) runs ONE Scribe pass on ``sample.audio_path``, computes
    jiwer ``process_words`` + ``process_characters`` of (label vs Scribe hypothesis) with the
    standard scoring normalization, and writes the result via :meth:`CuratorStore.set_score`
    (``scribe_wer``/``scribe_cer`` columns + the full breakdown in ``meta["scribe"]``).

    ``scribe_language`` selects the Scribe language setting; ``None`` -> ``"auto"`` (detect /
    code-switch). NOTE: for languages Scribe can render in more than one script (Tajik -> Cyrillic
    vs Persian-Arabic), ``auto`` makes WER meaningless — force the Scribe ISO code that matches the
    stored labels' script. ``key`` resolves via the default key chain when not passed.

    Failure policy (the run must be impossible to turn into a request spray):
    - an auth error (dead key) triggers an in-run key renewal via the Superwhisper proxy, at most
      ``max_renewals`` times; in-flight calls made with the dead key are absorbed, then aborted if
      the renewed key still fails;
    - ``breaker_threshold`` consecutive non-auth failures abort the run (queued work cancelled).
    An aborted run loses nothing: failed clips stay un-scored and the next run picks them up.
    """
    from tqdm import tqdm

    pending = _pending(store, key=key, force=force)
    total = store_total(store, key=key)
    stats = VerifyStats(skipped=total - len(pending))
    if not pending:
        return stats

    lang = scribe_language or "auto"
    # Mutable holder: queued tasks look the fn up at call time, so a key renewal applies to
    # everything not yet started — only the in-flight calls still hit the dead key.
    holder = {"fn": make_scribe_fns(key=default_key(), langs=(lang,), model=model)[lang]}

    def _score(sample: Sample) -> dict[str, object]:
        return _score_one(sample, holder["fn"])

    wers: list[float] = []
    cers: list[float] = []
    consecutive = 0
    grace = 0  # dead-key in-flight calls still allowed to fail after a renewal
    queued = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded in-flight window: never more than 2x workers submitted ahead of the consumer.
        # Submitting the whole pending list upfront would let fast failures (a 401 returns in
        # ~100 ms) race far past the breaker before the consumer loop ever sees them.
        futures = {pool.submit(_score, s): s for s in _take(queued, workers * 2)}
        progress = tqdm(total=len(pending), desc="scribe-verify", unit="clip")
        while futures:
            future = next(as_completed(futures))
            sample = futures.pop(future)
            progress.update(1)
            for nxt in _take(queued, 1):
                futures[pool.submit(_score, nxt)] = nxt
            try:
                detail = future.result()
            except Exception as exc:  # noqa: BLE001 — counted; the breaker bounds the damage
                stats.failed += 1
                stats.failures.append((sample.id, f"{type(exc).__name__}: {exc}"))
                if isinstance(exc, ScribeError) and exc.auth:
                    if grace > 0:
                        grace -= 1
                        continue
                    if stats.renewals >= max_renewals:
                        _abort(pool, f"key renewal exhausted ({stats.renewals}x)", stats, exc)
                    progress.write(f"auth failure — renewing key ({stats.renewals + 1})")
                    holder["fn"] = make_scribe_fns(
                        key=renew_scribe_key(), langs=(lang,), model=model
                    )[lang]
                    stats.renewals += 1
                    grace = workers * 2  # absorb calls already in flight on the dead key
                    consecutive = 0
                    continue
                consecutive += 1
                if consecutive >= breaker_threshold:
                    _abort(pool, f"{consecutive} consecutive failures", stats, exc)
                continue
            consecutive = 0
            _record_score(store, stats, sample, detail, wers, cers)
        progress.close()

    stats.wer = _distribution(wers)
    stats.cer = _distribution(cers)
    return stats


def _take(source: Iterator[Sample], n: int) -> list[Sample]:
    """Up to ``n`` next items from ``source`` (the bounded-submission window refill)."""
    return list(islice(source, n))


def _record_score(
    store: CuratorStore,
    stats: VerifyStats,
    sample: Sample,
    detail: dict[str, object],
    wers: list[float],
    cers: list[float],
) -> None:
    """Persist one clip's fresh score and fold it into the run's distributions."""
    wer = float(cast("float", detail["wer"]))
    cer = float(cast("float", detail["cer"]))
    store.set_score(sample.id, scribe_wer=wer, scribe_cer=cer, detail=detail)
    stats.scored += 1
    wers.append(wer)
    cers.append(cer)


def _abort(pool: ThreadPoolExecutor, reason: str, stats: VerifyStats, exc: Exception) -> None:
    """Cancel all queued work and abort the run — failures must never become a request spray."""
    pool.shutdown(wait=False, cancel_futures=True)
    msg = (
        f"verify aborted: {reason}; scored {stats.scored}, failed {stats.failed}; "
        f"top failures: {stats.top_failures(3)}; last: {exc}"
    )
    raise RuntimeError(msg)


def store_total(store: CuratorStore, *, key: str | None = None) -> int:
    """Total clips in the pool (or in ``source == key``) — the denominator for ``skipped``."""
    return sum(1 for _ in (store.iter_samples(source=key) if key else store.iter_samples()))


def scribe_summary(store: CuratorStore) -> dict[str, dict[str, object]]:
    """Per-source "how clean is this set" numbers: count + mean/median Scribe WER/CER.

    Aggregates the persisted ``scribe_wer``/``scribe_cer`` columns by source. Each source entry has
    ``count`` (clips in the source), ``scored`` (clips with a Scribe score), and ``wer``/``cer``
    dicts (``mean``/``median`` over the scored clips, ``None`` when none are scored yet). An extra
    ``"__all__"`` entry rolls the same numbers up over the whole pool.
    """
    by_source: dict[str, list[Sample]] = {}
    for sample in store.iter_samples():
        by_source.setdefault(sample.source, []).append(sample)

    def _summarize(samples: Iterable[Sample]) -> dict[str, object]:
        items = list(samples)
        wers = [s.scribe_wer for s in items if s.scribe_wer is not None]
        cers = [s.scribe_cer for s in items if s.scribe_cer is not None]
        return {
            "count": len(items),
            "scored": len(wers),
            "wer": _mean_median(wers),
            "cer": _mean_median(cers),
        }

    summary: dict[str, dict[str, object]] = {
        source: _summarize(samples) for source, samples in sorted(by_source.items())
    }
    summary["__all__"] = _summarize(s for samples in by_source.values() for s in samples)
    return summary


def _mean_median(values: list[float]) -> dict[str, float | None]:
    """``mean``/``median`` of ``values``, or ``None`` for both when empty."""
    if not values:
        return {"mean": None, "median": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values)}
