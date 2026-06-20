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
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, cast

from omni_curator.benchmark import dominant_script, normalize, score_pair
from omni_curator.create.transcribe import make_scribe_fns, raise_for_scribe_error

#: Human-readable script names for the transliteration prompt, by FLORES script code.
_SCRIPT_NAMES = {
    "Cyrl": "Cyrillic",
    "Arab": "Perso-Arabic script",
    "Latn": "Latin script",
    "Geor": "Georgian script",
}

class _ThreadState(threading.local):
    """Per-thread SuperWhisper text client for transliteration (not assumed thread-safe)."""

    def __init__(self) -> None:
        self.client: SuperwhisperClient | None = None


_state = _ThreadState()


def _text_client() -> SuperwhisperClient:
    if _state.client is None:
        from omni_curator.swservice import SuperwhisperClient

        _state.client = SuperwhisperClient()
    return _state.client

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from omni_curator.create.transcribe import ScribeFn
    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore
    from omni_curator.swservice import SuperwhisperClient


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
    unscoreable: int = 0  # label normalizes to nothing (e.g. '♪') — never scoreable, no call made
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


def _pending(store: CuratorStore, *, key: str | None, force: bool) -> tuple[list[Sample], int]:
    """Samples to score this run: filtered by ``key`` (source), un-scored unless ``force``.

    Returns ``(scoreable, unscoreable)``. A label that normalizes to nothing (``.``, ``...``,
    ``♪`` — Scribe's silence/music markers) can never be scored (jiwer rejects an empty
    reference), so those rows are dropped here BEFORE any Scribe call is spent on them; they
    stay ``scribe_wer IS NULL`` and the export junk filter is what removes them from training.
    """
    samples = store.iter_samples(source=key) if key is not None else store.iter_samples()
    chosen = samples if force else (s for s in samples if s.scribe_wer is None)
    scoreable: list[Sample] = []
    unscoreable = 0
    for sample in chosen:
        if normalize(sample.text):
            scoreable.append(sample)
        else:
            unscoreable += 1
    return scoreable, unscoreable


def _score_one(sample: Sample, scribe_fn: ScribeFn) -> dict[str, object]:
    """Run one Scribe pass over a clip and score its label against the hypothesis.

    Returns the ``meta["scribe"]`` detail dict: ``wer``/``cer``, the word/char S/D/I/H breakdown,
    the Scribe ``hypothesis`` text, plus the ``reference`` label. Raises on a Scribe / audio error
    (the service returns a result dict carrying ``error`` but no ``transcript``) — the caller turns
    the raised error into a counted failure rather than scoring an empty string.

    A hypothesis in a different script than the label (Scribe rendering Tajik speech in
    Perso-Arabic) is transliterated to the label's script first — WER across scripts compares
    alphabets, not speech. The raw hypothesis is preserved as ``hypothesis_raw``.
    """
    # The scribe fn takes a Path and returns the service result dict.
    result = scribe_fn(Path(sample.audio_path))
    raise_for_scribe_error(result)
    if "transcript" not in result:
        raise RuntimeError("scribe returned no transcript")
    hypothesis = str(result.get("transcript") or "").strip()
    return _score_hypothesis(sample, hypothesis)


def _score_hypothesis(sample: Sample, hypothesis: str) -> dict[str, object]:
    """Score one (label, hypothesis) pair, transliterating a cross-script hypothesis first."""
    detail_extra: dict[str, object] = {}
    ref_script = dominant_script(sample.text)
    hyp_script = dominant_script(hypothesis)
    if hypothesis and ref_script and hyp_script and hyp_script != ref_script:
        from omni_curator.create.fuse import transliterate

        raw = hypothesis
        hypothesis = transliterate(
            raw,
            language=sample.language,
            script=_SCRIPT_NAMES.get(ref_script, ref_script),
            client=_text_client(),
        )
        detail_extra = {"hypothesis_raw": raw, "scoring": "transliterated"}
    detail = score_pair(sample.text, hypothesis)
    detail["hypothesis"] = hypothesis
    detail["reference"] = sample.text
    detail.update(detail_extra)
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
    stored labels' script. ``key`` filters the store to ``source == key`` when passed.

    Failure policy: the deployed service owns ASR key rotation, so there is no in-run key
    renewal here. ``breaker_threshold`` consecutive failures abort the run (queued work
    cancelled). An aborted run loses nothing: failed clips stay un-scored and the next run
    picks them up.
    """
    from tqdm import tqdm

    pending, unscoreable = _pending(store, key=key, force=force)
    total = store_total(store, key=key)
    stats = VerifyStats(skipped=total - len(pending) - unscoreable, unscoreable=unscoreable)
    if not pending:
        return stats

    lang = scribe_language or "auto"
    scribe_fn = make_scribe_fns((lang,), model=model)[lang]

    def _score(sample: Sample) -> dict[str, object]:
        return _score_one(sample, scribe_fn)

    wers: list[float] = []
    cers: list[float] = []
    consecutive = 0
    queued = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded in-flight window: never more than 2x workers submitted ahead of the consumer.
        # Submitting the whole pending list upfront would let fast failures race far past the
        # breaker before the consumer loop ever sees them.
        futures = {pool.submit(_score, s): s for s in _take(queued, workers * 2)}
        progress = tqdm(total=len(pending), desc="scribe-verify", unit="clip")
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                sample = futures.pop(future)
                progress.update(1)
                try:
                    detail = future.result()
                except Exception as exc:  # noqa: BLE001 — counted; the breaker bounds the damage
                    stats.failed += 1
                    stats.failures.append((sample.id, f"{type(exc).__name__}: {exc}"))
                    consecutive += 1
                    if consecutive >= breaker_threshold:
                        _abort(pool, f"{consecutive} consecutive failures", stats, exc)
                    continue
                consecutive = 0
                _record_score(store, stats, sample, detail, wers, cers)
            # Refill AFTER processing, so an abort decision never races extra submissions.
            for nxt in _take(queued, len(finished)):
                futures[pool.submit(_score, nxt)] = nxt
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


def _cross_script_candidates(store: CuratorStore, *, key: str | None) -> list[tuple[Sample, str]]:
    """Scored rows whose saved hypothesis is in a different script than the label.

    Skips rows already scored via transliteration (``hypothesis_raw`` present) — idempotency.
    """
    candidates: list[tuple[Sample, str]] = []
    samples = store.iter_samples(source=key) if key is not None else store.iter_samples()
    for sample in samples:
        if sample.scribe_wer is None:
            continue
        scribe_raw = sample.meta.get("scribe")
        if not isinstance(scribe_raw, dict):
            continue
        scribe = cast("dict[str, object]", scribe_raw)
        if "hypothesis_raw" in scribe:
            continue
        hypothesis = str(scribe.get("hypothesis") or "")
        ref_script = dominant_script(sample.text)
        hyp_script = dominant_script(hypothesis)
        if hypothesis and ref_script and hyp_script and hyp_script != ref_script:
            candidates.append((sample, hypothesis))
    return candidates


def rescore_cross_script(
    store: CuratorStore,
    *,
    key: str | None = None,
    workers: int = 50,
    breaker_threshold: int = 50,
    on_progress: Callable[[int], None] | None = None,
) -> VerifyStats:
    """Re-score already-verified rows whose stored hypothesis is in a different script.

    Earlier verify runs scored the raw hypothesis even when Scribe rendered it in another
    script than the label (WER ~1.0 on correct content). This pass reuses the hypothesis
    saved in ``meta["scribe"]`` — NO Scribe calls — transliterates it to the label's script,
    and overwrites the score. Idempotent: rows already carrying ``hypothesis_raw`` (scored
    via transliteration) are skipped. The only network work is one free text-LLM call per
    mismatched row.
    """
    from tqdm import tqdm

    candidates = _cross_script_candidates(store, key=key)
    stats = VerifyStats()
    if not candidates:
        return stats
    wers: list[float] = []
    cers: list[float] = []
    consecutive = 0
    queued = iter(candidates)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_score_hypothesis, s, h): s for s, h in islice(queued, workers * 2)
        }
        progress = tqdm(total=len(candidates), desc="rescore-script", unit="clip")
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                sample = futures.pop(future)
                progress.update(1)
                try:
                    detail = future.result()
                except Exception as exc:  # noqa: BLE001 — counted; the breaker bounds the damage
                    stats.failed += 1
                    stats.failures.append((sample.id, f"{type(exc).__name__}: {exc}"))
                    consecutive += 1
                    if consecutive >= breaker_threshold:
                        _abort(pool, f"{consecutive} consecutive failures", stats, exc)
                    continue
                consecutive = 0
                _record_score(store, stats, sample, detail, wers, cers)
                if on_progress:
                    on_progress(stats.scored)
            for nxt, hyp in islice(queued, len(finished)):
                futures[pool.submit(_score_hypothesis, nxt, hyp)] = nxt
        progress.close()

    stats.wer = _distribution(wers)
    stats.cer = _distribution(cers)
    return stats


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
