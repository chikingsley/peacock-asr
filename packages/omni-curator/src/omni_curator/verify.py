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
failures (missing audio, a transient Scribe error) are counted and skipped, never aborting the run.
``verify_store`` is idempotent: it only scores rows with ``scribe_wer IS NULL`` unless ``force``.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_curator.benchmark import score_pair
from omni_curator.create.transcribe import default_key, make_scribe_fns

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omni_curator.sample import Sample
    from omni_curator.store.sqlite import CuratorStore


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
    wer: dict[str, float] = field(default_factory=dict)
    cer: dict[str, float] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)


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
    if "transcript" not in result:
        msg = str(result.get("error") or "scribe returned no transcript")
        raise RuntimeError(msg)
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
) -> VerifyStats:
    """Score every (un-scored) clip's label against a fresh Scribe pass; persist the full result.

    Iterates the store (optionally restricted to ``source == key``), and for each clip with no score
    yet (or all clips when ``force=True``) runs ONE Scribe pass on ``sample.audio_path``, computes
    jiwer ``process_words`` + ``process_characters`` of (label vs Scribe hypothesis) with the
    standard scoring normalization, and writes the result via :meth:`CuratorStore.set_score`
    (``scribe_wer``/``scribe_cer`` columns + the full breakdown in ``meta["scribe"]``).

    ``scribe_language`` selects the Scribe language setting; ``None`` -> ``"auto"`` (detect /
    code-switch), which is the safe default since the curator's FLORES codes are not Scribe ISO
    codes. ``key`` resolves via :func:`omni_curator.create.transcribe.default_key` when not passed.

    The run is parallelized across ``workers`` threads (Scribe is I/O-bound and free). Per-clip
    errors are caught, counted, and skipped — the run never aborts on one bad clip. A ``tqdm`` bar
    tracks progress. Returns :class:`VerifyStats` (scored / skipped / failed + WER/CER spread).
    """
    from tqdm import tqdm

    pending = _pending(store, key=key, force=force)
    total = store_total(store, key=key)
    stats = VerifyStats(skipped=total - len(pending))
    if not pending:
        return stats

    lang = scribe_language or "auto"
    scribe_fn = make_scribe_fns(key=default_key(), langs=(lang,), model=model)[lang]

    wers: list[float] = []
    cers: list[float] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, s, scribe_fn): s for s in pending}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="scribe-verify", unit="clip"
        ):
            sample = futures[future]
            try:
                detail = future.result()
            except Exception as exc:  # noqa: BLE001 — one clip's failure must not abort the run
                stats.failed += 1
                stats.failures.append((sample.id, f"{type(exc).__name__}: {exc}"))
                continue
            wer = float(detail["wer"])  # type: ignore[arg-type]
            cer = float(detail["cer"])  # type: ignore[arg-type]
            store.set_score(sample.id, scribe_wer=wer, scribe_cer=cer, detail=detail)
            stats.scored += 1
            wers.append(wer)
            cers.append(cer)

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
