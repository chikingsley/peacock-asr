"""Shared Parakeet checkpoint and model evaluation CLI."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
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
    parser.add_argument(
        "--replace-tokenizer",
        action="store_true",
        help="Replace the base model tokenizer before loading --checkpoint. "
        "Never use this for an already fine-tuned .nemo model.",
    )
    parser.add_argument(
        "--ngram-lm",
        type=Path,
        default=None,
        help="Token-level ARPA/.nemo n-gram LM for NGPU-LM greedy fusion (hybrid models).",
    )
    parser.add_argument("--ngram-lm-alpha", type=float, default=0.0)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=0,
        help="0 uses batched greedy; a positive value uses batched GPU beam decoding.",
    )
    parser.add_argument(
        "--beam-beta",
        type=float,
        default=0.0,
        help="CTC word insertion bonus for batched beam decoding; ignored for TDT.",
    )
    parser.add_argument("--audio-field", default="audio_filepath")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--duration-field", default="duration")
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--inference-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Model dtype used for inference. The official NeMo leaderboard uses bf16 on CUDA.",
    )
    parser.add_argument(
        "--longform-attention-context",
        type=int,
        default=0,
        help=(
            "Use NeMo rel_pos_local_attn with this symmetric context for long recordings; "
            "0 preserves the model's serialized attention configuration."
        ),
    )
    parser.add_argument(
        "--load-model-on-cpu",
        action="store_true",
        help=(
            "Restore local model/checkpoint weights on CPU, cast there, and move the final model "
            "to --device. This avoids retaining the original GPU weight allocation."
        ),
    )
    parser.add_argument(
        "--disable-cuda-graph-decoder",
        action="store_true",
        help="Disable NeMo's greedy CUDA graph decoder to reduce inference-only GPU state.",
    )
    parser.add_argument(
        "--memory-efficient-subsampling",
        action="store_true",
        help=(
            "Use an allocation-efficient equivalent of NeMo's channel-chunked FastConformer "
            "subsampler for recordings that exceed available GPU memory."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=8,
        help="CUDA clips to transcribe before timing; ignored on CPU.",
    )
    parser.add_argument("--normalizer", default=project.default_eval_normalizer)
    parser.add_argument(
        "--normalizer-language",
        default=project.default_eval_normalizer_language or project.language,
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary-json", type=Path, default=None)
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
    return float(compute_error_rates(refs, hyps, normalizer)["wer_percent"])


def compute_error_rates(
    refs: list[str], hyps: list[str], normalizer: Normalizer
) -> dict[str, float | int]:
    from jiwer import process_characters, process_words

    normalized_refs = [normalizer(ref) for ref in refs]
    normalized_hyps = [normalizer(hyp) for hyp in hyps]
    scored = [
        (reference, hypothesis)
        for reference, hypothesis in zip(normalized_refs, normalized_hyps, strict=True)
        if reference.strip()
    ]
    if not scored:
        raise ValueError("normalization removed every reference")
    scored_refs, scored_hyps = (list(values) for values in zip(*scored, strict=True))
    return {
        "wer_percent": float(process_words(scored_refs, scored_hyps).wer * 100),
        "cer_percent": float(process_characters(scored_refs, scored_hyps).cer * 100),
        "empty_hypotheses": sum(not hypothesis.strip() for hypothesis in scored_hyps),
        "scored_rows": len(scored),
        "excluded_empty_references": len(normalized_refs) - len(scored),
    }


def coerce_hypotheses(raw_hypotheses: list[Any]) -> list[str]:
    return [str(hyp.text if hasattr(hyp, "text") else hyp) for hyp in raw_hypotheses]


def replacement_tokenizer_dir(args: argparse.Namespace) -> Path | None:
    if not args.replace_tokenizer:
        return None
    if args.checkpoint is None:
        raise SystemExit(
            "--replace-tokenizer requires --checkpoint; an already fine-tuned .nemo "
            "must be evaluated without changing its vocabulary"
        )
    return Path(require(args.tokenizer_dir, "--tokenizer-dir"))


def configured_decoding(source: Any, args: argparse.Namespace, decoder_type: str) -> Any | None:
    """Build the requested greedy/beam NGPU-LM config for a CTC or transducer decoder."""
    if getattr(args, "ngram_lm", None) is None and getattr(args, "beam_size", 0) <= 0:
        return None

    from copy import deepcopy

    from omegaconf import open_dict

    decoding_cfg = deepcopy(source)
    with open_dict(decoding_cfg):
        if args.beam_size > 0:
            decoding_cfg.strategy = "malsd_batch" if decoder_type == "rnnt" else "beam_batch"
            decoding_cfg.beam.beam_size = args.beam_size
            decoding_cfg.beam.return_best_hypothesis = True
            decoding_cfg.beam.ngram_lm_model = (
                str(args.ngram_lm) if args.ngram_lm is not None else None
            )
            decoding_cfg.beam.ngram_lm_alpha = args.ngram_lm_alpha
            if decoder_type == "rnnt":
                # Some NVIDIA checkpoints serialize these PrettyStrEnum defaults as their
                # uppercase member names, while current NeMo accepts their lowercase values.
                decoding_cfg.beam.pruning_mode = "late"
                decoding_cfg.beam.blank_lm_score_mode = "lm_weighted_full"
            else:
                decoding_cfg.beam.beam_beta = args.beam_beta
        else:
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.greedy.ngram_lm_model = str(args.ngram_lm)
            decoding_cfg.greedy.ngram_lm_alpha = args.ngram_lm_alpha
    print(
        f"decoding strategy={decoding_cfg.strategy} beam_size={args.beam_size} "
        f"NGPU-LM={args.ngram_lm} alpha={args.ngram_lm_alpha}",
        flush=True,
    )
    return decoding_cfg


def configure_inference_runtime(model: Any, args: argparse.Namespace, torch: Any) -> Any:
    """Apply explicit long-form attention and inference dtype choices."""
    longform_context = getattr(args, "longform_attention_context", 0)
    if longform_context < 0:
        raise ValueError("--longform-attention-context must be non-negative")
    if longform_context:
        if not hasattr(model, "change_attention_model"):
            raise TypeError("model does not support NeMo long-form local attention")
        model.change_attention_model("rel_pos_local_attn", [longform_context, longform_context])
        print(
            "long-form attention="
            f"rel_pos_local_attn context=[{longform_context}, {longform_context}]",
            flush=True,
        )

    if getattr(args, "memory_efficient_subsampling", False):
        enable_memory_efficient_subsampling(model, torch)
        print("memory-efficient channel-chunked subsampling enabled", flush=True)

    inference_dtype = getattr(args, "inference_dtype", "fp32")
    if inference_dtype == "bf16":
        model = model.to(torch.bfloat16)
    elif inference_dtype != "fp32":
        raise ValueError(f"unsupported inference dtype: {inference_dtype}")
    model = model.to(args.device).eval()
    if str(args.device).startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return model


def disable_cuda_graph_decoder(decoding_cfg: Any) -> Any:
    """Return a NeMo decoding config with the greedy CUDA graph path disabled."""
    from copy import deepcopy

    from omegaconf import OmegaConf

    decoding_cfg = deepcopy(decoding_cfg)
    OmegaConf.update(
        decoding_cfg,
        "greedy.use_cuda_graph_decoder",
        value=False,
        force_add=True,
    )
    return decoding_cfg


def enable_memory_efficient_subsampling(model: Any, torch: Any) -> None:
    """Avoid NeMo's duplicate full-output allocation while preserving chunked convolution math."""
    try:
        subsampler = model.encoder.pre_encode
    except AttributeError as error:
        raise TypeError("model does not expose a FastConformer pre-encoder") from error
    if not hasattr(subsampler, "channel_chunked_conv"):
        raise TypeError("model does not expose NeMo channel-chunked convolution")

    def channel_chunked_conv(self: Any, conv: Any, chunk_size: int, x: Any) -> Any:
        channel_offset = 0
        output = None
        for chunk in torch.split(x, chunk_size, 1):
            channels = chunk.size(1)
            if self.is_causal:
                padded_chunk = torch.nn.functional.pad(
                    chunk,
                    pad=(
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                chunk_output = torch.nn.functional.conv2d(
                    padded_chunk,
                    conv.weight[channel_offset : channel_offset + channels],
                    bias=conv.bias[channel_offset : channel_offset + channels],
                    stride=self._stride,
                    padding=0,
                    groups=channels,
                )
            else:
                chunk_output = torch.nn.functional.conv2d(
                    chunk,
                    conv.weight[channel_offset : channel_offset + channels],
                    bias=conv.bias[channel_offset : channel_offset + channels],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=channels,
                )
            if output is None:
                output = chunk_output.new_empty(
                    (
                        chunk_output.size(0),
                        conv.out_channels,
                        chunk_output.size(2),
                        chunk_output.size(3),
                    )
                )
            output[:, channel_offset : channel_offset + channels].copy_(chunk_output)
            channel_offset += channels
            del chunk_output
        if output is None:
            raise ValueError("channel-chunked convolution received an empty tensor")
        return output

    subsampler.channel_chunked_conv = MethodType(channel_chunked_conv, subsampler)


def load_model(args: argparse.Namespace, model_name: str | Path) -> Any:  # noqa: C901, PLR0912
    tokenizer_dir = replacement_tokenizer_dir(args)

    import torch  # ty: ignore[unresolved-import]
    from nemo.collections.asr.models import ASRModel  # ty: ignore[unresolved-import]

    load_device = "cpu" if getattr(args, "load_model_on_cpu", False) else args.device
    model_path = Path(str(model_name)).expanduser()
    if model_path.exists():
        model = ASRModel.restore_from(str(model_path), map_location=load_device)
    else:
        model = ASRModel.from_pretrained(str(model_name))

    if tokenizer_dir is not None:
        model.change_vocabulary(
            new_tokenizer_dir=str(tokenizer_dir.resolve()),
            new_tokenizer_type=args.tokenizer_type,
        )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=load_device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(f"loaded checkpoint {args.checkpoint} with an exact state-dict match", flush=True)

    # Hybrid TDT+CTC models decode with whichever head is current; --kind must select it,
    # otherwise both kinds silently score the same lane.
    if hasattr(model, "cur_decoder"):
        decoder_type = "ctc" if args.kind == "ctc" else "rnnt"
        wants_decode_change = (
            getattr(args, "ngram_lm", None) is not None
            or getattr(args, "beam_size", 0) > 0
            or getattr(args, "disable_cuda_graph_decoder", False)
        )
        if wants_decode_change:
            source = model.cfg.decoding if decoder_type == "rnnt" else model.cfg.aux_ctc.decoding
            decoding_cfg = configured_decoding(source, args, decoder_type)
            if decoding_cfg is None:
                decoding_cfg = source
            if getattr(args, "disable_cuda_graph_decoder", False):
                decoding_cfg = disable_cuda_graph_decoder(decoding_cfg)
        else:
            decoding_cfg = None
        model.change_decoding_strategy(decoding_cfg, decoder_type=decoder_type)
        print(f"hybrid model: decoding with the {decoder_type} head", flush=True)
    else:
        wants_decode_change = (
            getattr(args, "ngram_lm", None) is not None
            or getattr(args, "beam_size", 0) > 0
            or getattr(args, "disable_cuda_graph_decoder", False)
        )
        if wants_decode_change:
            decoder_type = "ctc" if args.kind == "ctc" else "rnnt"
            decoding_cfg = configured_decoding(model.cfg.decoding, args, decoder_type)
            if decoding_cfg is None:
                decoding_cfg = model.cfg.decoding
            if getattr(args, "disable_cuda_graph_decoder", False):
                decoding_cfg = disable_cuda_graph_decoder(decoding_cfg)
            model.change_decoding_strategy(decoding_cfg)
            print(f"standalone model: configured the {decoder_type} decoder", flush=True)

    return configure_inference_runtime(model, args, torch)


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


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def transcribe_timed(
    model: Any, rows: list[EvalRow], args: argparse.Namespace
) -> tuple[list[str], dict[str, float | int | None]]:
    import torch  # ty: ignore[unresolved-import]

    paths = [row.audio_filepath for row in rows]
    use_cuda_metrics = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda_metrics and args.warmup_count > 0 and len(paths) > args.warmup_count:
        model.transcribe(paths[: args.warmup_count], batch_size=args.batch_size)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    raw_hypotheses = model.transcribe(paths, batch_size=args.batch_size)
    if use_cuda_metrics:
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    audio_seconds = sum(row.duration or 0.0 for row in rows)
    rtfx = audio_seconds / elapsed_seconds if audio_seconds and elapsed_seconds else None
    performance = {
        "warmup_count": args.warmup_count if use_cuda_metrics else 0,
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed_seconds,
        "rtfx": rtfx,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if use_cuda_metrics else None,
    }
    return coerce_hypotheses(raw_hypotheses), performance


def run(project: ParakeetProject, args: argparse.Namespace) -> None:
    manifest = Path(require(args.manifest, "--manifest"))
    model_name = args.model_name or default_model_for_kind(project, args.kind)
    model_name = require(model_name, "--model-name")
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_for_kind(project, args.kind)
    replacement_tokenizer_dir(args)

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
        f"replace_tokenizer={args.replace_tokenizer} manifest={manifest} "
        f"rows={len(rows)} device={args.device}",
        flush=True,
    )
    if args.dry_run:
        return
    if not rows:
        raise SystemExit("manifest produced no rows")

    model = load_model(args, model_name)
    refs = [row.text for row in rows]
    hyps, performance = transcribe_timed(model, rows, args)
    normalizer = make_normalizer(args.normalizer, args.normalizer_language)
    normalized = compute_error_rates(refs, hyps, normalizer)
    raw = compute_error_rates(refs, hyps, str)
    summary = {
        "kind": args.kind,
        "model": str(model_name),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "manifest": str(manifest),
        "rows": len(rows),
        "device": args.device,
        "inference_dtype": args.inference_dtype,
        "longform_attention_context": args.longform_attention_context,
        "load_model_on_cpu": args.load_model_on_cpu,
        "disable_cuda_graph_decoder": args.disable_cuda_graph_decoder,
        "memory_efficient_subsampling": args.memory_efficient_subsampling,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "beam_beta": args.beam_beta,
        "ngram_lm": str(args.ngram_lm) if args.ngram_lm is not None else None,
        "ngram_lm_alpha": args.ngram_lm_alpha,
        **performance,
        "raw": raw,
        "normalized": normalized,
    }

    print("\n=== samples ===", flush=True)
    for row, hyp in list(zip(rows, hyps, strict=True))[: args.sample_count]:
        print(f"REF: {row.text[:120]}\nHYP: {hyp[:120]}\n", flush=True)
    normalized_cer = normalized["cer_percent"]
    normalized_cer_text = "n/a" if normalized_cer is None else f"{normalized_cer:.2f}%"
    print(
        f"=== normalized WER/CER ({normalized['scored_rows']}/{len(rows)} clips): "
        f"{normalized['wer_percent']:.2f}% / {normalized_cer_text} ===",
        flush=True,
    )
    if normalized["excluded_empty_references"]:
        print(
            "normalized scoring excluded "
            f"{normalized['excluded_empty_references']} empty references",
            flush=True,
        )
    print(
        f"raw WER/CER: {raw['wer_percent']:.2f}% / {raw['cer_percent']:.2f}% | "
        f"empty hypotheses: {normalized['empty_hypotheses']}",
        flush=True,
    )
    if performance["rtfx"] is not None:
        print(
            f"RTFx={performance['rtfx']:.1f} "
            f"({performance['audio_seconds']:.1f}s audio / "
            f"{performance['elapsed_seconds']:.2f}s, batch={args.batch_size}, "
            f"warmup={performance['warmup_count']})",
            flush=True,
        )
    if performance["peak_vram_bytes"] is not None:
        peak_gib = float(performance["peak_vram_bytes"]) / (1024**3)
        print(f"peak CUDA allocation={peak_gib:.2f} GiB", flush=True)

    if args.output_jsonl is not None:
        write_predictions(args.output_jsonl, rows, hyps)
        print(f"wrote {args.output_jsonl}", flush=True)
    if args.output_summary_json is not None:
        write_summary(args.output_summary_json, summary)
        print(f"wrote {args.output_summary_json}", flush=True)


def eval_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    project.configure_environment()
    run(project, build_parser(project).parse_args(argv))
    return 0
