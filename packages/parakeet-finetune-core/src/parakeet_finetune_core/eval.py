"""Shared Parakeet checkpoint and model evaluation CLI."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parakeet_finetune_core.project import ParakeetProject

Normalizer = Callable[[str], str]
MIN_POSITIONAL_WITH_LANGUAGE = 2


@dataclass(frozen=True)
class EvalRow:
    audio_filepath: str
    text: str
    duration: float | None = None


def default_model_for_kind(project: ParakeetProject, kind: str) -> str | Path | None:
    # prefer the promoted final eval model; fall back to the training base
    if kind == "ctc":
        return project.default_eval_ctc_model or project.default_ctc_model
    return (
        project.default_eval_tdt_model or project.default_tdt_model or project.default_hybrid_model
    )


def default_checkpoint_for_kind(project: ParakeetProject, kind: str) -> Path | None:
    if kind == "ctc":
        return project.default_ctc_checkpoint
    return project.default_tdt_checkpoint


def build_parser(project: ParakeetProject) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate a Parakeet CTC or TDT checkpoint for {project.name}."
    )
    parser.add_argument("--kind", choices=["ctc", "tdt"], default=project.default_eval_kind)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=project.default_validation_manifest)
    parser.add_argument("--tokenizer-dir", type=Path, default=project.default_tokenizer_dir)
    parser.add_argument("--tokenizer-type", default="bpe")
    parser.add_argument("--audio-field", default="audio_filepath")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--duration-field", default="duration")
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--normalizer", default=project.default_eval_normalizer)
    parser.add_argument(
        "--normalizer-language",
        default=project.default_eval_normalizer_language or project.language,
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require(value: Any, label: str) -> Any:
    if value in (None, ""):
        raise SystemExit(f"{label} is required")
    return value


def load_manifest(
    manifest: Path,
    *,
    audio_field: str,
    text_field: str,
    duration_field: str,
    max_duration: float | None,
    limit: int,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            duration = item.get(duration_field)
            if duration is not None:
                duration = float(duration)
            if max_duration is not None and duration is not None and duration > max_duration:
                continue
            rows.append(
                EvalRow(
                    audio_filepath=str(item[audio_field]),
                    text=str(item[text_field]),
                    duration=duration,
                )
            )
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def make_normalizer(spec: str | None, language: str | None) -> Normalizer:
    if not spec:
        return str
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("normalizer must use module:function format")
    normalizer = getattr(importlib.import_module(module_name), function_name)

    try:
        signature = inspect.signature(normalizer)
    except (TypeError, ValueError):
        accepts_language = language is not None
    else:
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        accepts_language = language is not None and (
            len(positional) >= MIN_POSITIONAL_WITH_LANGUAGE
            or any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
        )

    def normalize(text: str) -> str:
        result = normalizer(text, language) if accepts_language else normalizer(text)
        return "" if result is None else str(result)

    return normalize


def compute_wer_percent(refs: list[str], hyps: list[str], normalizer: Normalizer) -> float:
    from jiwer import process_words

    normalized_refs = [normalizer(ref) for ref in refs]
    normalized_hyps = [normalizer(hyp) for hyp in hyps]
    return float(process_words(normalized_refs, normalized_hyps).wer * 100)


def coerce_hypotheses(raw_hypotheses: list[Any]) -> list[str]:
    return [str(hyp.text if hasattr(hyp, "text") else hyp) for hyp in raw_hypotheses]


def load_model(args: argparse.Namespace, model_name: str | Path) -> Any:
    import torch  # ty: ignore[unresolved-import]
    from nemo.collections.asr.models import ASRModel  # ty: ignore[unresolved-import]

    model_path = Path(str(model_name)).expanduser()
    if model_path.exists():
        model = ASRModel.restore_from(str(model_path), map_location=args.device)
    else:
        model = ASRModel.from_pretrained(str(model_name))

    if args.tokenizer_dir is not None:
        model.change_vocabulary(
            new_tokenizer_dir=str(Path(args.tokenizer_dir).resolve()),
            new_tokenizer_type=args.tokenizer_type,
        )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"loaded checkpoint {args.checkpoint}: missing={len(missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )

    return model.to(args.device).eval()


def write_predictions(path: Path, rows: list[EvalRow], hyps: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, hyp in zip(rows, hyps, strict=True):
            handle.write(
                json.dumps(
                    {
                        "audio_filepath": row.audio_filepath,
                        "text": row.text,
                        "hypothesis": hyp,
                        "duration": row.duration,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run(project: ParakeetProject, args: argparse.Namespace) -> None:
    manifest = Path(require(args.manifest, "--manifest"))
    model_name = args.model_name or default_model_for_kind(project, args.kind)
    model_name = require(model_name, "--model-name")
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_for_kind(project, args.kind)

    rows = load_manifest(
        manifest,
        audio_field=args.audio_field,
        text_field=args.text_field,
        duration_field=args.duration_field,
        max_duration=args.max_duration,
        limit=args.limit,
    )
    print(
        f"kind={args.kind} model={model_name} checkpoint={args.checkpoint} "
        f"manifest={manifest} rows={len(rows)} device={args.device}",
        flush=True,
    )
    if args.dry_run:
        return
    if not rows:
        raise SystemExit("manifest produced no rows")

    model = load_model(args, model_name)
    paths = [row.audio_filepath for row in rows]
    refs = [row.text for row in rows]
    raw_hyps = model.transcribe(paths, batch_size=args.batch_size)
    hyps = coerce_hypotheses(raw_hyps)
    normalizer = make_normalizer(args.normalizer, args.normalizer_language)
    wer = compute_wer_percent(refs, hyps, normalizer)

    print("\n=== samples ===", flush=True)
    for row, hyp in list(zip(rows, hyps, strict=True))[: args.sample_count]:
        print(f"REF: {row.text[:120]}\nHYP: {hyp[:120]}\n", flush=True)
    print(f"=== WER ({len(rows)} clips): {wer:.2f}% ===", flush=True)

    if args.output_jsonl is not None:
        write_predictions(args.output_jsonl, rows, hyps)
        print(f"wrote {args.output_jsonl}", flush=True)


def eval_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    project.configure_environment()
    run(project, build_parser(project).parse_args(argv))
    return 0
