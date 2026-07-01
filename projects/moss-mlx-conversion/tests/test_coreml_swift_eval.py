import argparse
from pathlib import Path

from moss_mlx_conversion.coreml.swift_eval import safe_stem, swift_batch_command, swift_command
from moss_mlx_conversion.runtime.eval import StreamingExample


def test_safe_stem_uses_row_and_sanitized_id() -> None:
    example = StreamingExample(
        row_idx=7,
        example_id="abc/def ghi",
        reference="hello",
        audio_url="https://example.test/audio.flac",
    )

    assert safe_stem(example) == "000007-abc_def_ghi"


def test_swift_command_uses_absolute_runtime_inputs() -> None:
    args = argparse.Namespace(
        swift_package_path=Path("swift/MossCoreMLFixture"),
        packages_dir=Path("coreml/build"),
        fixture=Path("artifacts/coreml/moss_swift_fixture_compact.json"),
        runtime_manifest=None,
        audio_max_frames=3000,
        audio_package="compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
        decoder_package="compiled_stateful/moss_decoder_stateful_fused.mlmodelc",
        prefill_cache_package=None,
        prefill_cache_seq_len=None,
        step_package=None,
        cache_len=768,
        compute_units="cpu-gpu",
        max_new_tokens=160,
    )

    command = swift_command(
        project_root=Path("/example/moss"),
        args=args,
        audio_path=Path("/example/audio.wav"),
        reference_path=Path("/example/reference.txt"),
        output_path=Path("/example/out.json"),
    )

    assert command[:6] == [
        "swift",
        "run",
        "--package-path",
        "/example/moss/swift/MossCoreMLFixture",
        "-c",
        "release",
    ]
    assert "--audio-max-frames" in command
    assert "3000" in command
    assert "--reference-text-file" in command
    assert "/example/reference.txt" in command
    assert "--decoder-package" in command
    assert "compiled_stateful/moss_decoder_stateful_fused.mlmodelc" in command
    assert "--prefill-cache-package" not in command


def test_swift_command_can_use_external_cache_packages() -> None:
    args = argparse.Namespace(
        swift_package_path=Path("swift/MossCoreMLFixture"),
        packages_dir=Path("coreml/build"),
        fixture=Path("artifacts/coreml/moss_swift_fixture_compact.json"),
        runtime_manifest=None,
        audio_max_frames=3000,
        audio_package="compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
        decoder_package="compiled_stateful/moss_decoder_stateful_fused.mlmodelc",
        prefill_cache_package="compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
        prefill_cache_seq_len=512,
        step_package="compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
        cache_len=768,
        compute_units="cpu-gpu",
        max_new_tokens=160,
    )

    command = swift_command(
        project_root=Path("/example/moss"),
        args=args,
        audio_path=Path("/example/audio.wav"),
        reference_path=Path("/example/reference.txt"),
        output_path=Path("/example/out.json"),
    )

    assert "--prefill-cache-package" in command
    assert "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc" in command
    assert "--prefill-cache-seq-len" in command
    assert "512" in command
    assert "--step-package" in command
    assert "compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc" in command
    assert "--cache-len" in command
    assert "768" in command


def test_swift_batch_command_uses_manifest_and_external_cache() -> None:
    args = argparse.Namespace(
        swift_package_path=Path("swift/MossCoreMLFixture"),
        packages_dir=Path("coreml/build"),
        fixture=Path("artifacts/coreml/moss_swift_fixture_compact.json"),
        runtime_manifest=Path("runtime/moss_runtime_manifest.json"),
        audio_max_frames=3000,
        audio_package="compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
        decoder_package="compiled_stateful/moss_decoder_stateful_fused.mlmodelc",
        prefill_cache_package="compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
        prefill_cache_seq_len=512,
        step_package="compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
        cache_len=768,
        compute_units="cpu-gpu",
        max_new_tokens=160,
    )

    command = swift_batch_command(
        project_root=Path("/example/moss"),
        args=args,
        manifest_path=Path("/example/manifest.jsonl"),
        batch_output_path=Path("/example/batch-results.jsonl"),
    )

    assert "--batch-manifest" in command
    assert "/example/manifest.jsonl" in command
    assert "--batch-output-jsonl" in command
    assert "/example/batch-results.jsonl" in command
    assert "--audio" not in command
    assert "--reference-text-file" not in command
    assert "--runtime-manifest" in command
    assert "/example/moss/runtime/moss_runtime_manifest.json" in command
    assert "--prefill-cache-seq-len" in command
    assert "512" in command
