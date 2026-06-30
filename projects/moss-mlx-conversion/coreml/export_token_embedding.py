from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/weights.safetensors"
)
DEFAULT_EXTRACTED_WEIGHTS = PROJECT_ROOT / "artifacts/coreml/moss-token-embedding-fp16.safetensors"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "coreml/build"
DEFAULT_WEIGHT_KEY = "model.embed_tokens.weight"


class TokenEmbedding(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids.to(torch.long))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MOSS token embedding to CoreML.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--weight-key", default=DEFAULT_WEIGHT_KEY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_token_embedding.mlpackage")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--extracted-weights", type=Path, default=DEFAULT_EXTRACTED_WEIGHTS)
    parser.add_argument("--validate-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_embedding_weight(path: Path, key: str) -> tuple[str, torch.Tensor]:
    tensors = load_file(str(path), device="cpu")
    if key in tensors:
        weight_key = key
    elif len(tensors) == 1:
        weight_key = next(iter(tensors))
    else:
        available = ", ".join(sorted(tensors)[:20])
        raise KeyError(f"missing {key!r} in {path}; first available keys: {available}")
    return weight_key, tensors[weight_key].to(dtype=torch.float16).contiguous()


def extract_embedding_weight(*, weights: Path, key: str, output: Path) -> dict[str, Any]:
    weight_key, weight = load_embedding_weight(weights, key)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({weight_key: weight}, str(output))
    return {
        "source_weights": str(weights),
        "output": str(output),
        "weight_key": weight_key,
        "weight_dtype": str(weight.dtype),
        "weight_shape": list(weight.shape),
        "bytes": output.stat().st_size,
    }


def example_input(*, max_seq_len: int, vocab_size: int) -> torch.Tensor:
    ids = torch.arange(max_seq_len, dtype=torch.int32).reshape(1, max_seq_len)
    return ids.remainder(vocab_size)


def export_embedding(
    *,
    weights: Path,
    weight_key: str,
    output_dir: Path,
    package_name: str,
    max_seq_len: int,
    validate_predict: bool,
    overwrite: bool,
) -> dict[str, Any]:
    import coremltools as ct

    resolved_key, weight = load_embedding_weight(weights, weight_key)
    module = TokenEmbedding(weight).eval()
    inputs = example_input(max_seq_len=max_seq_len, vocab_size=weight.shape[0])
    with torch.no_grad():
        traced = torch.jit.trace(module, inputs, strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / package_name
    if package_path.exists():
        if not overwrite:
            raise FileExistsError(f"{package_path} exists; pass --overwrite to replace it")
        shutil.rmtree(package_path)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=inputs.shape,
                dtype=np.int32,
            )
        ],
        outputs=[ct.TensorType(name="token_embeddings")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(str(package_path))

    validation: dict[str, Any] | None = None
    if validate_predict:
        torch_output = module(inputs).detach().cpu().numpy()
        prediction = mlmodel.predict({"input_ids": inputs.numpy().astype(np.int32)})
        output_key = (
            "token_embeddings" if "token_embeddings" in prediction else next(iter(prediction))
        )
        coreml_output = np.asarray(prediction[output_key])
        diff = np.abs(coreml_output.astype(np.float32) - torch_output.astype(np.float32))
        validation = {
            "output_key": output_key,
            "coreml_shape": list(coreml_output.shape),
            "torch_shape": list(torch_output.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        }

    return {
        "source_weights": str(weights),
        "output_package": str(package_path),
        "weight_key": resolved_key,
        "weight_dtype": str(weight.dtype),
        "weight_shape": list(weight.shape),
        "input_shape": list(inputs.shape),
        "minimum_deployment_target": "macOS14",
        "compute_precision": "FLOAT16",
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    if args.extract_only:
        manifest = extract_embedding_weight(
            weights=args.weights.resolve(),
            key=args.weight_key,
            output=args.extracted_weights.resolve(),
        )
        write_json(args.extracted_weights.resolve().with_suffix(".json"), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest = export_embedding(
        weights=args.weights.resolve(),
        weight_key=args.weight_key,
        output_dir=args.output_dir.resolve(),
        package_name=args.package_name,
        max_seq_len=args.max_seq_len,
        validate_predict=args.validate_predict,
        overwrite=args.overwrite,
    )
    manifest_path = Path(manifest["output_package"]).with_suffix(".json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
