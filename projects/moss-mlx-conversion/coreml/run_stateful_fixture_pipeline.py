from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "artifacts/cache/huggingface/models--OpenMOSS-Team--MOSS-Transcribe-preview-2B"
    / "snapshots/c98175cb20e48bd9be4e95f6c85f2af18899f780"
)
DEFAULT_CONFIG = SNAPSHOT_DIR / "config.json"
DEFAULT_REFERENCE_TENSORS = (
    PROJECT_ROOT / "artifacts/reference/libri1-pytorch-bf16/reference_tensors.npz"
)
DEFAULT_PACKAGES_DIR = PROJECT_ROOT / "coreml/build"
DEFAULT_OUTPUT = PROJECT_ROOT / "coreml/build/moss_coreml_stateful_fixture_pipeline.json"
DEFAULT_EMBEDDING_PACKAGE = "moss_token_embedding.mlpackage"
DEFAULT_AUDIO_PACKAGE = "moss_audio_encoder_adapter_fixture.mlpackage"
DEFAULT_DECODER_PACKAGE = "moss_decoder_stateful_fused.mlpackage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MOSS CoreML token embedding, audio encoder+adapter, "
            "and stateful decoder against the LibriSpeech fixture."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-tensors", type=Path, default=DEFAULT_REFERENCE_TENSORS)
    parser.add_argument("--packages-dir", type=Path, default=DEFAULT_PACKAGES_DIR)
    parser.add_argument("--token-package", default=DEFAULT_EMBEDDING_PACKAGE)
    parser.add_argument("--audio-package", default=DEFAULT_AUDIO_PACKAGE)
    parser.add_argument("--decoder-package", default=DEFAULT_DECODER_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-max-seq-len", type=int, default=512)
    parser.add_argument(
        "--merged-source",
        choices=["coreml-components", "reference"],
        default="coreml-components",
        help=(
            "Use CoreML token/audio outputs for the decoder input, or use the "
            "saved reference merged embeddings to isolate decoder wiring."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def topk(values: np.ndarray, k: int = 5) -> dict[str, Any]:
    flat = values.reshape(-1)
    indices = np.argpartition(flat, -k)[-k:]
    ordered = indices[np.argsort(flat[indices])[::-1]]
    return {
        "indices": [int(index) for index in ordered],
        "values": [float(flat[index]) for index in ordered],
    }


def padded_ids(ids: np.ndarray, *, max_seq_len: int) -> np.ndarray:
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"expected ids with shape [1, seq], got {ids.shape}")
    if ids.shape[1] > max_seq_len:
        raise ValueError(f"ids length {ids.shape[1]} exceeds max_seq_len={max_seq_len}")
    padded = np.zeros((1, max_seq_len), dtype=np.int32)
    padded[:, : ids.shape[1]] = ids.astype(np.int32)
    return padded


def qwen3_rope(
    *,
    positions: np.ndarray,
    head_dim: int,
    rope_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim))
    )
    freqs = np.einsum("i,j->ij", positions.astype(np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb)[None, :, :].astype(np.float32), np.sin(emb)[None, :, :].astype(np.float32)


def causal_mask(length: int) -> np.ndarray:
    mask = np.triu(np.ones((length, length), dtype=np.float32), k=1) * -1e9
    return mask.reshape(1, 1, length, length)


def timed_predict(
    model: Any,
    inputs: dict[str, np.ndarray],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    output = model.predict(inputs) if state is None else model.predict(inputs, state=state)
    return output, time.perf_counter() - start


def load_coreml_model(path: Path) -> Any:
    import coremltools as ct

    return ct.models.MLModel(str(path))


def run_pipeline(  # noqa: PLR0915
    *,
    config: Path,
    reference_tensors: Path,
    packages_dir: Path,
    token_package: str,
    audio_package: str,
    decoder_package: str,
    token_max_seq_len: int,
    merged_source: str,
) -> dict[str, Any]:
    config = require_path(config)
    reference_tensors = require_path(reference_tensors)
    packages_dir = require_path(packages_dir)
    token_path = require_path(packages_dir / token_package)
    audio_path = require_path(packages_dir / audio_package)
    decoder_path = require_path(packages_dir / decoder_package)

    config_data = load_json(config)
    language_config = config_data["language_config"]
    head_dim = int(language_config["head_dim"])
    rope_theta = float(language_config["rope_theta"])
    hidden_size = int(language_config["hidden_size"])

    refs = np.load(reference_tensors)
    input_ids = refs["input_ids"].astype(np.int64)
    audio_input_mask = refs["audio_input_mask"].astype(bool)
    generated_ids = refs["generated_ids"].astype(np.int64)
    prompt_len = int(input_ids.shape[1])
    first_token_id = int(generated_ids[0, 0])
    second_token_id = int(generated_ids[0, 1])
    audio_token_count = int(audio_input_mask.sum())

    token_model = load_coreml_model(token_path)
    audio_model = load_coreml_model(audio_path)
    decoder_model = load_coreml_model(decoder_path)

    token_prediction, token_seconds = timed_predict(
        token_model,
        {"input_ids": padded_ids(input_ids, max_seq_len=token_max_seq_len)},
    )
    token_embeddings = np.asarray(token_prediction["token_embeddings"], dtype=np.float32)[
        :, :prompt_len, :
    ]

    audio_prediction, audio_seconds = timed_predict(
        audio_model,
        {
            "audio_data": refs["audio_data"].astype(np.float32),
            "audio_data_seqlens": refs["audio_data_seqlens"].astype(np.int32),
        },
    )
    audio_embeddings = np.asarray(audio_prediction["audio_embeddings"], dtype=np.float32)
    if audio_embeddings.shape[0] != audio_token_count:
        raise ValueError(
            f"audio output length {audio_embeddings.shape[0]} "
            f"!= audio mask count {audio_token_count}"
        )
    if token_embeddings.shape[-1] != hidden_size or audio_embeddings.shape[-1] != hidden_size:
        raise ValueError(
            "unexpected hidden size: "
            f"token={token_embeddings.shape[-1]}, audio={audio_embeddings.shape[-1]}, "
            f"config={hidden_size}"
        )

    component_merged_embeddings = token_embeddings.copy()
    component_merged_embeddings[0, audio_input_mask[0], :] = audio_embeddings
    if merged_source == "reference":
        decoder_input_embeddings = refs["merged_embeds"].astype(np.float32)
    else:
        decoder_input_embeddings = component_merged_embeddings

    prefill_positions = np.arange(prompt_len, dtype=np.int64)
    prefill_cos, prefill_sin = qwen3_rope(
        positions=prefill_positions,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    step_cos, step_sin = qwen3_rope(
        positions=np.array([prompt_len], dtype=np.int64),
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    prefill_mask = causal_mask(prompt_len)
    step_mask = np.zeros((1, 1, 1, prompt_len + 1), dtype=np.float32)

    state = decoder_model.make_state()
    prefill_prediction, prefill_seconds = timed_predict(
        decoder_model,
        {
            "inputs_embeds": decoder_input_embeddings.astype(np.float32),
            "cos": prefill_cos,
            "sin": prefill_sin,
            "attention_mask": prefill_mask,
        },
        state=state,
    )
    prefill_logits = np.asarray(prefill_prediction["logits"], dtype=np.float32)

    first_token_input = np.array([[first_token_id]], dtype=np.int64)
    first_token_prediction, first_token_seconds = timed_predict(
        token_model,
        {"input_ids": padded_ids(first_token_input, max_seq_len=token_max_seq_len)},
    )
    first_token_embeddings = np.asarray(
        first_token_prediction["token_embeddings"],
        dtype=np.float32,
    )[:, :1, :]

    step_prediction, step_seconds = timed_predict(
        decoder_model,
        {
            "inputs_embeds": first_token_embeddings.astype(np.float32),
            "cos": step_cos,
            "sin": step_sin,
            "attention_mask": step_mask,
        },
        state=state,
    )
    step_logits = np.asarray(step_prediction["logits"], dtype=np.float32)

    prefill_topk = topk(prefill_logits)
    step_topk = topk(step_logits)
    return {
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "packages": {
            "token_embedding": str(token_path),
            "audio_encoder_adapter": str(audio_path),
            "stateful_decoder": str(decoder_path),
        },
        "prompt_len": prompt_len,
        "audio_token_count": audio_token_count,
        "first_token_id": first_token_id,
        "second_token_id": second_token_id,
        "token_embedding_shape": list(token_embeddings.shape),
        "audio_embedding_shape": list(audio_embeddings.shape),
        "decoder_input_source": merged_source,
        "merged_embedding_shape": list(decoder_input_embeddings.shape),
        "audio_embeddings_vs_reference": diff_stats(audio_embeddings, refs["audio_embeds"]),
        "component_merged_embeddings_vs_reference": diff_stats(
            component_merged_embeddings,
            refs["merged_embeds"],
        ),
        "decoder_input_embeddings_vs_reference": diff_stats(
            decoder_input_embeddings,
            refs["merged_embeds"],
        ),
        "prefill_topk": prefill_topk,
        "step_topk": step_topk,
        "prefill_top1_matches_first_token": prefill_topk["indices"][0] == first_token_id,
        "step_top1_matches_second_token": step_topk["indices"][0] == second_token_id,
        "prefill_logits_vs_reference": diff_stats(prefill_logits, refs["prefill_last_logits"]),
        "timing_seconds": {
            "token_embedding_prompt": token_seconds,
            "audio_encoder_adapter": audio_seconds,
            "stateful_decoder_prefill": prefill_seconds,
            "token_embedding_first_token": first_token_seconds,
            "stateful_decoder_step": step_seconds,
            "total": token_seconds
            + audio_seconds
            + prefill_seconds
            + first_token_seconds
            + step_seconds,
        },
    }


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        config=args.config.resolve(),
        reference_tensors=args.reference_tensors.resolve(),
        packages_dir=args.packages_dir.resolve(),
        token_package=args.token_package,
        audio_package=args.audio_package,
        decoder_package=args.decoder_package,
        token_max_seq_len=args.token_max_seq_len,
        merged_source=args.merged_source,
    )
    write_json(args.output.resolve(), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
