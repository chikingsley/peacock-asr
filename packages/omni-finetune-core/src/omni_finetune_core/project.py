"""The per-language model-side CLI: train + eval, parameterized by a :class:`FinetuneProject`.

The model-side counterpart of ``omni_curator.project``: a language project's ``train.py`` builds
a :class:`FinetuneProject` (typed presets from :mod:`omni_finetune_core.presets`, eval defaults,
the scoring normalizer) and delegates to :func:`train_main`; its ``eval.py`` delegates to
:func:`eval_main`. Presets are *typed Python* (``TrainingConfig`` factories), not hand-copied
YAML — proven field-equivalent to the recipe configs they replace, and every project gets the
same train/eval surface.

Eval scores one or more registered model cards over an export's ``split=test`` partitions
(corpus-level jiwer via :mod:`omni_finetune_core.metrics`), with per-corpus breakdowns and
corpus filters (``--only-corpus-prefix youtube-`` = the conversational held-out alone).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

from omni_finetune_core.metrics import compute_measures
from omni_finetune_core.train import configure_environment, run_recipe, train

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from omni_finetune_core.config import TrainingConfig

SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 40.0  # omni ASRInferencePipeline hard cap (assert_max_length)


@dataclass(frozen=True)
class TrainingPreset:
    """One runnable training recipe: a lazy typed config + where the run lives."""

    config: Callable[[], TrainingConfig]
    output_dir: Path


@dataclass(frozen=True, kw_only=True)
class FinetuneProject:
    """Everything language-specific for training + eval, in one frozen config object.

    ``normalize`` is the scoring normalization (bind the language into
    ``omni_curator.process.normalize`` — injected so this package never depends on the
    curator). ``eval_dataset_root`` is the default ``version=N`` export root whose
    ``split=test`` partitions eval reads; ``eval_models`` are the default ``label -> card``
    pairs scored when ``--models`` is not given.
    """

    name: str  # short project name for messages, e.g. "tajik"
    language: str  # curator language code, e.g. "tgk_Cyrl" (also the pipeline lang hint)
    root: Path  # project root: caches + runs/ hang off it
    presets: Mapping[str, TrainingPreset] = field(default_factory=dict)
    # The generic-regime path (--regime gpu_max|1b|warm_restart): card names + the mixture TSV
    # + where runs land. A project can use named presets, the generic regimes, or both.
    model_card: str | None = None
    dataset_card: str | None = None
    tokenizer_card: str | None = None
    dataset_summary_path: Path | None = None  # mixture TSV (weighted, if the project has one)
    fragment_cache_dir: Path | None = None  # real disk! the /tmp default overflows tmpfs
    default_output_dir: Path | None = None  # generic-regime runs land here
    eval_dataset_root: Path | None = None
    eval_models: Mapping[str, str] = field(default_factory=dict)
    normalize: Callable[[str], str] = lambda text: text

    def __post_init__(self) -> None:
        object.__setattr__(self, "presets", dict(self.presets))
        object.__setattr__(self, "eval_models", dict(self.eval_models))

    def corpus_hours(self) -> float:
        """TRUE audio hours of the export the mixture TSV belongs to (for step budgeting).

        Reads ``export_summary.json`` next to the TSV — never the TSV itself, whose hours may
        be sampling WEIGHTS (see ``language_distribution_weighted.tsv``), not provenance.
        """
        if self.dataset_summary_path is None:
            msg = "no dataset_summary_path on this project"
            raise SystemExit(msg)
        summary = self.dataset_summary_path.parent / "export_summary.json"
        if not summary.exists():
            msg = f"export summary not found (needed for the step budget): {summary}"
            raise SystemExit(msg)
        return float(json.loads(summary.read_text(encoding="utf-8"))["hours"])


# -- train --------------------------------------------------------------------------------------


REGIMES = ("gpu_max", "1b", "warm_restart")


def build_train_parser(project: FinetuneProject) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run an Omnilingual ASR training recipe for {project.name}."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(project.presets),
        default=None,
        help="a pinned, named recipe (exactly reproducible)",
    )
    parser.add_argument(
        "--regime",
        choices=REGIMES,
        default=None,
        help="a generic recipe built from the project's cards",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="regime step budget (default: ~30 epochs from the export's hours)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="regime peak LR (warm_restart: peak_lr)"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="ad-hoc recipe YAML (escape hatch; needs --output-dir)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("recipe_args", nargs=argparse.REMAINDER)
    return parser


def _regime_config(
    project: FinetuneProject, regime: str, num_steps: int, lr: float | None
) -> TrainingConfig:
    """Build a generic-regime config from the project's cards (fail fast on missing config)."""
    from omni_finetune_core import presets

    model = project.model_card
    dataset = project.dataset_card
    tokenizer = project.tokenizer_card
    summary = project.dataset_summary_path
    if model is None or dataset is None or tokenizer is None or summary is None:
        msg = (
            "--regime needs model_card, dataset_card, tokenizer_card and "
            "dataset_summary_path on the project config"
        )
        raise SystemExit(msg)
    summary_path = str(summary)
    cache_dir = str(project.fragment_cache_dir) if project.fragment_cache_dir else None
    if regime == "gpu_max":
        return presets.gpu_max_finetune(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            dataset_summary_path=summary_path,
            num_steps=num_steps,
            lr=lr if lr is not None else 1e-5,
            fragment_cache_dir=cache_dir,
        )
    if regime == "1b":
        return presets.gpu_max_finetune_1b(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            dataset_summary_path=summary_path,
            num_steps=num_steps,
            lr=lr if lr is not None else 1e-5,
            fragment_cache_dir=cache_dir,
        )
    return presets.warm_restart(
        checkpoint_card=model,
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_summary_path=summary_path,
        num_steps=num_steps,
        peak_lr=lr if lr is not None else 2e-6,
        fragment_cache_dir=cache_dir,
    )


def train_main(project: FinetuneProject, argv: list[str] | None = None) -> int:
    """Run a pinned preset, a generic regime, or an ad-hoc ``--config-file``."""
    args = build_train_parser(project).parse_args(argv)
    if args.preset is not None and args.regime is not None:
        raise SystemExit("--preset and --regime are mutually exclusive")
    configure_environment(project.root)
    recipe_args = list(args.recipe_args)
    if recipe_args and recipe_args[0] == "--":
        recipe_args = recipe_args[1:]

    if args.preset is not None:
        preset = project.presets[args.preset]
        output_dir = (args.output_dir or preset.output_dir).resolve()
        train(preset.config(), output_dir, extra_args=recipe_args)
        return 0

    if args.regime is not None:
        from omni_finetune_core.presets import recommend_num_steps

        num_steps = args.num_steps
        if num_steps is None:
            num_steps = recommend_num_steps(project.corpus_hours(), target_epochs=30)
        output_dir = args.output_dir or project.default_output_dir
        if output_dir is None:
            raise SystemExit("--regime needs --output-dir (or default_output_dir on the project)")
        config = _regime_config(project, args.regime, num_steps, args.lr)
        print(
            f"regime={args.regime} num_steps={num_steps} output_dir={output_dir.resolve()}",
            flush=True,
        )
        train(config, output_dir.resolve(), extra_args=recipe_args)
        return 0

    if args.config_file is None or args.output_dir is None:
        raise SystemExit(
            "pick one: --preset NAME | --regime gpu_max|1b|warm_restart | "
            "--config-file F --output-dir D"
        )
    run_recipe(args.config_file.resolve(), args.output_dir.resolve(), extra_args=recipe_args)
    return 0


# -- eval ---------------------------------------------------------------------------------------


def _test_parquets(version_root: Path, language: str) -> list[Path]:
    return sorted(version_root.glob(f"corpus=*/split=test/language={language}/*.parquet"))


def _load_test(
    version_root: Path, language: str, limit: int, max_dur: float
) -> tuple[list[np.ndarray], list[str], list[str], int]:
    """Audio (FLAC int8 arrays), reference texts, and corpus tags from the test partitions."""
    audio: list[np.ndarray] = []
    refs: list[str] = []
    corpora: list[str] = []
    excluded = 0
    for parquet_path in _test_parquets(version_root, language):
        corpus = next(p.split("=", 1)[1] for p in parquet_path.parts if p.startswith("corpus="))
        # Stream batches so --limit stops early instead of reading the whole partition.
        for batch in pq.ParquetFile(str(parquet_path)).iter_batches(
            batch_size=256, columns=["text", "audio_bytes", "audio_size"]
        ):
            for text, ab, size in zip(
                batch.column("text").to_pylist(),
                batch.column("audio_bytes").to_pylist(),
                batch.column("audio_size").to_pylist(),
                strict=True,
            ):
                if size / SAMPLE_RATE > max_dur:
                    excluded += 1
                    continue
                audio.append(np.asarray(ab, dtype=np.int8))
                refs.append(text)
                corpora.append(corpus)
                if limit and len(audio) >= limit:
                    return audio, refs, corpora, excluded
    return audio, refs, corpora, excluded


def _load_manifest_test(
    manifest: Path, limit: int, max_dur: float
) -> tuple[list[np.ndarray], list[str], list[str], int]:
    """Load materialized encoded audio without expanding Parquet list columns to Python ints."""
    audio: list[np.ndarray] = []
    refs: list[str] = []
    corpora: list[str] = []
    excluded = 0
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            duration = float(row.get("duration", 0.0))
            if duration > max_dur:
                excluded += 1
                continue
            audio_path = Path(row["audio_filepath"])
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            audio.append(np.fromfile(audio_path, dtype=np.int8))
            refs.append(str(row["text"]))
            corpora.append(str(row.get("corpus", "manifest")))
            if limit and len(audio) >= limit:
                break
    return audio, refs, corpora, excluded


def _measures(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Corpus-level metrics via the shared scorer. WER/CER/MER/WIL + S/D/I/H per alignment."""
    m = compute_measures(refs, hyps)
    return {
        "wer": m.wer,
        "cer": m.cer,
        "mer": m.mer,
        "wil": m.wil,
        "sub": float(m.substitutions),
        "del": float(m.deletions),
        "ins": float(m.insertions),
        "hits": float(m.hits),
    }


def _parse_models(project: FinetuneProject, values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        if not project.eval_models:
            raise SystemExit("no eval models: pass --models label=card_name")
        return list(project.eval_models.items())
    out = []
    for value in values:
        label, _, card = value.partition("=")
        out.append((label, card) if card else (label, label))
    return out


def _safe_output_label(label: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in label
    )
    return safe.strip("_-") or "model"


def _write_eval_predictions(
    path: Path,
    refs: list[str],
    hyps: list[str],
    refs_norm: list[str],
    hyps_norm: list[str],
    corpora: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (ref, hyp, ref_norm, hyp_norm, corpus) in enumerate(
            zip(refs, hyps, refs_norm, hyps_norm, corpora, strict=True)
        ):
            handle.write(
                json.dumps(
                    {
                        "row_index": index,
                        "corpus": corpus,
                        "text": ref,
                        "hypothesis": hyp,
                        "normalized_text": ref_norm,
                        "normalized_hypothesis": hyp_norm,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_eval_parser(project: FinetuneProject) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Score {project.name} model cards on an export's test split."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=project.eval_dataset_root,
        help="version=N root of an export (its split=test partitions are scored)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="materialized JSONL audio manifest; takes precedence over --dataset-root",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="label=card_name pairs (default: the project's eval_models)",
    )
    p.add_argument(
        "--exclude-corpus",
        nargs="+",
        default=None,
        help="drop these corpora (e.g. fleurs -> conversational-only)",
    )
    p.add_argument(
        "--only-corpus-prefix",
        default=None,
        help="keep only corpora starting with this (e.g. youtube-)",
    )
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-duration", type=float, default=MAX_AUDIO_SEC, help="drop clips longer")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write per-model predictions plus one summary JSON",
    )
    return p


def eval_main(project: FinetuneProject, argv: list[str] | None = None) -> int:
    """Transcribe + score the test partitions with each model card; print the comparison."""
    args = build_eval_parser(project).parse_args(argv)
    if args.manifest is None and args.dataset_root is None:
        raise SystemExit("pass --manifest or --dataset-root (or set a project dataset root)")
    os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(project.root / ".fairseq2-cache/assets"))

    import torch

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    models = _parse_models(project, args.models)
    if args.manifest is not None:
        audio, refs, corpora, excluded = _load_manifest_test(
            args.manifest, args.limit, args.max_duration
        )
    else:
        audio, refs, corpora, excluded = _load_test(
            args.dataset_root, project.language, args.limit, args.max_duration
        )
    keep = [
        i
        for i, c in enumerate(corpora)
        if (args.exclude_corpus is None or c not in args.exclude_corpus)
        and (args.only_corpus_prefix is None or c.startswith(args.only_corpus_prefix))
    ]
    if len(keep) != len(corpora):
        audio = [audio[i] for i in keep]
        refs = [refs[i] for i in keep]
        corpora = [corpora[i] for i in keep]
    print(
        f"test rows: {len(audio)} (excluded {excluded} > {args.max_duration:.0f}s) | "
        f"corpora: {sorted(set(corpora))} | device: {args.device}",
        flush=True,
    )
    refs_norm = [project.normalize(r) for r in refs]

    summary: dict[str, dict[str, float]] = {}
    for label, card in models:
        print(f"\n=== {label} ({card}) ===", flush=True)
        pipe = ASRInferencePipeline(card, device=args.device, dtype=dtype)
        hyps_raw = pipe.transcribe(
            audio, lang=[project.language] * len(audio), batch_size=args.batch_size
        )
        del pipe
        hyps = [project.normalize(h) for h in hyps_raw]
        m = _measures(refs_norm, hyps)
        summary[label] = m
        if args.output_dir is not None:
            output_path = args.output_dir / f"{_safe_output_label(label)}-predictions.jsonl"
            _write_eval_predictions(output_path, refs, hyps_raw, refs_norm, hyps, corpora)
            print(f"wrote {output_path}", flush=True)
        print(
            f"{label}: WER {m['wer']:.2f}%  CER {m['cer']:.2f}%  MER {m['mer']:.2f}%  "
            f"WIL {m['wil']:.2f}%  |  sub {m['sub']:.0f} del {m['del']:.0f} "
            f"ins {m['ins']:.0f} hits {m['hits']:.0f}",
            flush=True,
        )
        for corp in sorted(set(corpora)):
            idx = [i for i, x in enumerate(corpora) if x == corp]
            cm = _measures([refs_norm[i] for i in idx], [hyps[i] for i in idx])
            print(
                f"    {corp:<28} WER {cm['wer']:6.2f}%  CER {cm['cer']:6.2f}%  (n={len(idx)})",
                flush=True,
            )

    print(f"\n=== SUMMARY ({project.name} test split, corpus-level, jiwer) ===")
    print(f"{'model':<16}{'WER':>9}{'CER':>9}{'MER':>9}")
    for label, m in summary.items():
        print(f"{label:<16}{m['wer']:>8.2f}%{m['cer']:>8.2f}%{m['mer']:>8.2f}%")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "project": project.name,
                    "rows": len(audio),
                    "corpora": sorted(set(corpora)),
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "models": dict(models),
                    "metrics": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {summary_path}", flush=True)
    return 0
